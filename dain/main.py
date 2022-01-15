import shutil
import os
import sys
import scipy.signal
import scipy.io.wavfile
import librosa
import threading
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from multiprocessing import Pool, Process
from queue import Queue
from copy import deepcopy
from keras import Model, layers
from keras.callbacks import Callback
import sys
import matplotlib.pyplot as plt
import keras.backend as K
import tensorlayer as tl


#file = "out.flac"
file = "/mnt/maestro-v3.0.0/repeat.flac"
DATA_DURATION = 2
SAMPLE_RATE = 20480
duration = 731119
num_samples = duration // DATA_DURATION
COMPRESSION_RATE = 16


def init_stream():
	return librosa.stream(
		file, 
		block_length=1, 
		frame_length=DATA_DURATION * SAMPLE_RATE, 
		hop_length=DATA_DURATION * SAMPLE_RATE,
	)


stream_lock = threading.Lock()
class Dataset(Sequence):
	def __init__(self, batch_size):
		self.stream = init_stream()
		self.batch_size = batch_size


	def raw(self):
		stream_lock.acquire()
		try:
			wav = next(self.stream)
		except RuntimeError:
			print("failed")
			if stream_lock.locked(): stream_lock.release()
			return self.raw()
		except StopIteration:
			print("DONE: RESTARTING")
			if stream_lock.locked(): stream_lock.release()
			self.stream = init_stream()
			return self.raw()
		if len(wav) != SAMPLE_RATE * DATA_DURATION:
			if stream_lock.locked(): stream_lock.release()
			return self.raw()
		if stream_lock.locked(): stream_lock.release()
		return wav

	def get(self):
		raw = self.raw()
		#resampled = raw[::COMPRESSION_RATE]
		resampled = np.mean(raw.reshape(-1, COMPRESSION_RATE), axis=1)
		#resampled = np.array([resampled.T]).T
		#`raw = np.array([raw.T]).T
		return resampled[..., None], raw[..., None]
	
	def __len__(self):
		return num_samples // self.batch_size
	
	def __getitem__(self, i):
		x = []
		y = []

		for _ in range(self.batch_size):
			gx, gy = self.get()
			x.append(gx)
			y.append(gy)
		
		x = np.array(x)
		y = np.array(y)

		return x, y


def _phase_shift(I, r):
    bsize, a, b, c = I.get_shape().as_list()
    bsize = tf.shape(I)[0] # Handling Dimension(None) type for undefined batch dim
    X = tf.reshape(I, (bsize, a, b, r, r))
    X = tf.transpose(X, (0, 1, 2, 4, 3))  # bsize, a, b, 1, 1
    X = tf.split(1, a, X)  # a, [bsize, b, r, r]
    X = tf.concat(2, [tf.squeeze(x, axis=1) for x in X])  # bsize, b, a*r, r
    X = tf.split(1, b, X)  # b, [bsize, a*r, r]
    X = tf.concat(2, [tf.squeeze(x, axis=1) for x in X])  # bsize, a*r, b*r
    return tf.reshape(X, (bsize, a*r, b*r, 1))


def PS(X, r, color=False):
    if color:
        Xc = tf.split(3, 3, X)
        X = tf.concat(3, [_phase_shift(x, r) for x in Xc])
    else:
        X = _phase_shift(X, r)
    return X



class SubPixel1D(layers.Layer):
	def __init__(self, r):
		super(SubPixel1D, self).__init__()
		self.r = r
	def call(self, I):
		with tf.compat.v1.name_scope('subpixel'):
			X = tf.transpose(a=I, perm=[2,1,0]) # (r, w, b)
			X = tf.batch_to_space(X, [self.r], [[0,0]]) # (1, r*w, b)
			X = tf.transpose(a=X, perm=[2,1,0])
			return X



from tensorflow.python.keras import backend as K
from keras.layers import merge
from keras.layers.core import Activation, Dropout
from keras.layers import Conv1D
from keras.layers import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.initializers import RandomNormal, Orthogonal

class Autoencoder(Model):
	def __init__(self):
		super(Autoencoder, self).__init__()
		"""
			layers.Input(shape=(2560, 1)),
			layers.Conv1D(filters=16, kernel_size=64+1, activation=None, padding="same"),
			layers.Dropout(rate=0.5),
			layers.Activation("relu"),
			SubPixel1D(r=2),

			layers.Conv1D(filters=32, kernel_size=32+1, activation=None, padding="same"),
			layers.Dropout(rate=0.5),
			layers.Activation("relu"),
			SubPixel1D(r=2),

			layers.Conv1D(filters=64, kernel_size=16+1, activation=None, padding="same"),
			layers.Dropout(rate=0.5),
			layers.Activation("relu"),
			SubPixel1D(r=2),

			layers.Conv1D(filters=128, kernel_size=8+1, activation=None, padding="same"),
			layers.Dropout(rate=0.5),
			layers.Activation("relu"),
			SubPixel1D(r=2),
		"""
		scale = 1
		self.model = tf.keras.Sequential([
			layers.Input(shape=(2560, 1)),

			Conv1D(filters=2*128//scale, kernel_size=65, activation=None, padding="same"),
			Dropout(rate=0.5),
			SubPixel1D(r=2),
			Conv1D(filters=128//scale, kernel_size=65, activation=None, padding="same"),
			LeakyReLU(0.2),

			Conv1D(filters=2*384//scale, kernel_size=33, activation=None, padding="same"),
			Dropout(rate=0.5),
			SubPixel1D(r=2),
			Conv1D(filters=384//scale, kernel_size=33, activation=None, padding="same"),
			LeakyReLU(0.2),

			Conv1D(filters=2*512//scale, kernel_size=17, activation=None, padding="same"),
			Dropout(rate=0.5),
			SubPixel1D(r=2),
			Conv1D(filters=512//scale, kernel_size=17, activation=None, padding="same"),
			LeakyReLU(0.2),



			Conv1D(filters=2, kernel_size=9,
					activation=None, padding='same'),
			SubPixel1D(r=2)

		])

	def call(self, x):
		return self.model(x)	




class EvaluationCallback(Callback):
	def __init__(self):
		self.epoch = 0
		if os.path.exists("eval"):
			shutil.rmtree("eval")
		os.mkdir("eval")

	def on_epoch_begin(self, epoch, logs=None):
		self.epoch = epoch

	def on_train_batch_end(self, batch, logs=None):
		if batch % (100) != 0:
			return

		point = Dataset(1)[100]

		piano = librosa.load("../piano.wav", sr=SAMPLE_RATE)[0]
		real = piano[:SAMPLE_RATE * DATA_DURATION]

		small = np.mean(real.reshape(-1, COMPRESSION_RATE), axis=1)
		small = np.squeeze(small)
		small = small[None, ..., None]


		pred = tf.squeeze(self.model(small))

		time = np.linspace(0, len(pred) / 44100, num=len(pred))

		plt.clf()
		plt.plot(time, real)
		plt.plot(time, pred, alpha=0.5)
		os.makedirs(f"eval/{self.epoch}/{batch}")
		os.chdir(f"eval/{self.epoch}/{batch}")

		scipy.io.wavfile.write(f"real.mp3", SAMPLE_RATE, np.array(real))
		scipy.io.wavfile.write(f"pred.mp3", SAMPLE_RATE, np.array(pred))

		plt.savefig(f"fig")

		os.chdir("../../..")


def loss(a, b):
	#err = tf.signal.stft(tf.squeeze(a), 512, 512) - tf.signal.stft(tf.squeeze(b), 512, 512)
	err = a - b
	return K.mean(K.square(err))

ds = Dataset(4)

autoencoder = Autoencoder()
schedule = tf.keras.optimizers.schedules.ExponentialDecay(
	0.000025,
	decay_steps=25000,
	decay_rate=0.996,
)
autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.000025, decay=0.01), loss=loss)
autoencoder.fit(ds, epochs=10, callbacks=[EvaluationCallback()])

#lower lr
#easing model

# try rmse instead of mse