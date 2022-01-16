import shutil
from threaded_fn import threaded_fn
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


class Dataset(Sequence):
	def __init__(self, batch_size):
		self.stream = init_stream()
		self.batch_size = batch_size

	def raw(self):
		try:
			wav = next(self.stream)
		except RuntimeError:
			print("failed")
			return self.raw()
		except StopIteration:
			print("DONE: RESTARTING")
			self.stream = init_stream()
			return self.raw()
		if len(wav) != SAMPLE_RATE * DATA_DURATION:
			return self.raw()
		return wav

	def __len__(self):
		return num_samples // self.batch_size
	
	def __getitem__(self, i):
		batch = np.array([self.raw() for i in range(self.batch_size)])
		return batch, batch

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
from keras.layers import Conv1D, MaxPool1D, LeakyReLU, Dropout

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
		self.encoder = tf.keras.Sequential([
			layers.Input(shape=(40960, 1)),

			Conv1D(filters=32, kernel_size=33, activation=None, padding="same"),
			Dropout(rate=0.25),
			MaxPool1D(pool_size=2, padding="same"),
			LeakyReLU(0.2),

			Conv1D(filters=16, kernel_size=17, activation=None, padding="same"),
			Dropout(rate=0.25),
			MaxPool1D(pool_size=2, padding="same"),
			LeakyReLU(0.2),

			Conv1D(filters=8, kernel_size=9, activation=None, padding="same"),
			Dropout(rate=0.25),
			MaxPool1D(pool_size=2, padding="same"),
			LeakyReLU(0.2),

			Conv1D(filters=4, kernel_size=5, activation=None, padding="same"),
			Dropout(rate=0.25),
			MaxPool1D(pool_size=2, padding="same"),
			LeakyReLU(0.2),

			Conv1D(filters=2, kernel_size=5, activation=None, padding="same"),
			Dropout(rate=0.25),
			LeakyReLU(0.2),

			Conv1D(filters=1, kernel_size=5, activation=None, padding="same"),
			Dropout(rate=0.25),
			LeakyReLU(0.2),



		])


		self.decoder = tf.keras.Sequential([
			layers.Input(shape=(2560, 1)),

			Conv1D(filters=2*32, kernel_size=33, activation=None, padding="same"),
			Dropout(rate=0.25),
			SubPixel1D(r=2),
			Conv1D(filters=32, kernel_size=33, activation=None, padding="same"),
			LeakyReLU(0.2),

			Conv1D(filters=2*96, kernel_size=17, activation=None, padding="same"),
			Dropout(rate=0.25),
			SubPixel1D(r=2),
			Conv1D(filters=96, kernel_size=17, activation=None, padding="same"),
			LeakyReLU(0.2),

			Conv1D(filters=2*128, kernel_size=9, activation=None, padding="same"),
			Dropout(rate=0.25),
			SubPixel1D(r=2),
			Conv1D(filters=128, kernel_size=9, activation=None, padding="same"),
			LeakyReLU(0.2),



			Conv1D(filters=2, kernel_size=9,
					activation=None, padding='same'),
			SubPixel1D(r=2)

		])

	def call(self, x):
		return self.decoder(self.encoder(x))	




class EvaluationCallback(Callback):
	def __init__(self):
		self.epoch = 0
		if os.path.exists("eval"):
			shutil.rmtree("eval")
		os.mkdir("eval")

	def on_epoch_begin(self, epoch, logs=None):
		self.epoch = epoch

	@threaded_fn
	def on_train_batch_end(self, batch, logs=None):
		if batch % (500) != 0:
			return


		piano = librosa.load("../piano.wav", sr=SAMPLE_RATE)[0]
		piano = np.squeeze(piano)

		plt.clf()
		fig, axs = plt.subplots(5, figsize=(15, 30))
		for i in range(5):

			real = np.squeeze(piano[SAMPLE_RATE*DATA_DURATION*i:SAMPLE_RATE * DATA_DURATION*(i+1)])
			real = np.squeeze(real)[None, ...]


			pred = np.squeeze(self.model(real))
			real = np.squeeze(real)

			time = np.linspace(0, len(pred) / 44100, num=len(pred))


			axs[i].plot(time, real, label="real")
			axs[i].plot(time, pred, alpha=0.5, label="pred")
		plt.legend(["real", "small", "pred"])
		os.makedirs(f"eval/{self.epoch}/{batch}")
		os.chdir(f"eval/{self.epoch}/{batch}")

		scipy.io.wavfile.write(f"real.mp3", SAMPLE_RATE, np.array(real))
		scipy.io.wavfile.write(f"pred.mp3", SAMPLE_RATE, np.array(pred))

		plt.savefig(f"fig")
		plt.close()

		os.chdir("../../..")


def loss(a, b):
	err = a - b
	return K.mean(K.square(err))

ds = Dataset(4)

autoencoder = Autoencoder()
schedule = tf.keras.optimizers.schedules.ExponentialDecay(
	0.0001,
	decay_steps=5000,
	decay_rate=0.996,
)
autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=schedule), loss=loss)
autoencoder.fit(ds, epochs=10, callbacks=[EvaluationCallback()])

#lower lr
#easing model

# try rmse instead of mse
