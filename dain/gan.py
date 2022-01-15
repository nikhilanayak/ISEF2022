import multiprocessing
from keras.layers.pooling import AveragePooling1D
from tensorflow.python.ops.gen_batch_ops import batch
from tqdm import tqdm
import shutil
import os
import sys
from keras.layers import Conv1D, LeakyReLU, Dropout, Dense, Flatten, BatchNormalization, Reshape, ReLU
import scipy.signal
import scipy.io.wavfile
import librosa
import threading
from librosa.core.spectrum import stft
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


#file = "out.flac"
file = "/mnt/maestro-v3.0.0/repeat.flac"
DATA_DURATION = 2
SAMPLE_RATE = 40960
duration = 731119
num_samples = duration // DATA_DURATION
COMPRESSION_RATE = 32
LATENT_DIM = SAMPLE_RATE * DATA_DURATION // COMPRESSION_RATE


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
		return resampled, raw
	
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

def generator():

	decoder = tf.keras.Sequential([
			layers.Input(shape=(SAMPLE_RATE * DATA_DURATION // COMPRESSION_RATE, 1)),
			layers.Conv1DTranspose(128, 2, strides=2, activation="relu"),
			layers.Conv1DTranspose(64, 2, strides=2, activation="relu"),
			layers.Conv1DTranspose(32, 2, strides=2, activation="relu"),
			layers.Conv1DTranspose(16, 2, strides=2, activation="relu"),
			layers.Conv1DTranspose(8, 2, strides=2, activation="relu"),
			layers.Conv1D(1, 2, activation=None, padding="same")
			])

	return decoder

def discriminator(InputShape):
	model = tf.keras.Sequential()

	model.add(Reshape((InputShape, 1), input_shape=(InputShape,)))
	model.add(Conv1D(32, 100, strides=7, padding="valid"))
	model.add(ReLU())
	model.add(AveragePooling1D(4))
	model.add(BatchNormalization(momentum=0.9))
	model.add(Dropout(rate=0.1))

	model.add(Conv1D(16, 50, strides=5, padding="valid"))
	model.add(ReLU())
	model.add(BatchNormalization(momentum=0.9))
	model.add(Dropout(rate=0.1))

	model.add(Conv1D(8, 25, strides=3, padding="valid"))
	model.add(ReLU())
	model.add(BatchNormalization(momentum=0.9))
	model.add(Dropout(rate=0.1))

	model.add(Flatten())
	model.add(Dense(4096))
	model.add(LeakyReLU(alpha=0.01))
	model.add(BatchNormalization(momentum=0.9))
	model.add(Dense(1))
	return model


cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
cross_entropy = tf.keras.losses.MeanSquaredError()
def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

gen = generator()
disc = discriminator(DATA_DURATION * SAMPLE_RATE)


BATCH_SIZE = 32
ds = Dataset(BATCH_SIZE)

gen_losses = []
disc_losses = []

@tf.function
def train_step(data):
	noise = tf.random.normal([BATCH_SIZE, LATENT_DIM])

	with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
		generated = gen(noise, training=True)

		real_output = disc(data, training=True)
		fake_output = disc(generated, training=True)

		gen_loss = generator_loss(fake_output)
		disc_loss = discriminator_loss(real_output, fake_output)

		#gen_losses.append(gen_loss)
		#disc_losses.append(disc_loss)
	
	gradients_of_generator = gen_tape.gradient(gen_loss, gen.trainable_variables)
	gradients_of_discriminator = disc_tape.gradient(disc_loss, disc.trainable_variables)

	generator_optimizer.apply_gradients(zip(gradients_of_generator, gen.trainable_variables))
	discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, disc.trainable_variables))

	return gen_loss, disc_loss

shutil.rmtree("eval")
os.mkdir("eval")

for epoch in range(10):
	for i in tqdm(range(len(ds))):
		gl, dl = train_step(ds[0][1])

		gen_losses.append(gl)
		disc_losses.append(dl)

		if i % 100 == 0:
			print(sum(gen_losses)/len(gen_losses))
			print(sum(disc_losses)/len(disc_losses))

			piano = librosa.load("../piano.wav", sr=SAMPLE_RATE)[0]
			real = piano[:SAMPLE_RATE * DATA_DURATION]
			small = np.array([real[::COMPRESSION_RATE]])
			pred = tf.squeeze(gen(small))

			time = np.linspace(0, len(pred) / 44100, num=len(pred))
			plt.clf()
			plt.plot(time, real)
			plt.plot(time, pred, alpha=0.5)
			os.makedirs(f"eval/{epoch}/{i}")
			os.chdir(f"eval/{epoch}/{i}")

			scipy.io.wavfile.write(f"real.mp3", SAMPLE_RATE, np.array(real))
			scipy.io.wavfile.write(f"pred.mp3", SAMPLE_RATE, np.array(pred))

			plt.savefig(f"fig")

			os.chdir("../../..")


"""
def stacked_G_D(g, d):
	model = tf.keras.Sequential()
	model.add(g)
	model.add(d)
	return model

gen = generator(LATENT_DIM, DATA_DURATION * SAMPLE_RATE)
disc = discriminator(DATA_DURATION * SAMPLE_RATE)
schedule = tf.keras.optimizers.schedules.ExponentialDecay(
	0.001,
	decay_steps=25000,
	decay_rate=0.996,
)

optimizer = tf.keras.optimizers.Adam(learning_rate=schedule)

gen.compile(loss="mse", optimizer=optimizer)
disc.compile(loss="binary_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.9))

epochs = 10
batch_size = 32
ds = Dataset(batch_size)

d_losses = []
g_losses = []
for cnt in range(epochs):
	for i in tqdm(range(len(ds))):
		small, real = ds[0]
		#gen_noise = np.random.normal(0, 1, (batch_size, LATENT_DIM))
		synthetic_audios = np.squeeze(gen.predict(small))

		x_combined_batch = np.concatenate((real, synthetic_audios))
		y_combined_batch = np.concatenate((np.ones((batch_size, 1)), np.zeros((batch_size, 1))))

		d_loss = disc.train_on_batch(x_combined_batch, y_combined_batch)


		d_loss_mean = np.mean(d_loss)
		d_losses.append(d_loss_mean)


		#noise = np.random.normal(0, 1, (batch_size, LATENT_DIM))
		#y_mislabeled = np.ones((batch_size, 1))
		#g_loss = gen.train_on_batch(noise, y_mislabeled)
		g_loss = gen.train_on_batch(small, real)


		g_loss_mean = np.mean(g_loss)
		g_losses.append(g_loss_mean)

		if i % 10 == 0 :
			print(f"{np.mean(g_losses)},{np.mean(d_losses)}")

		if i % 100 == 0:
			piano = librosa.load("../piano.wav", sr=SAMPLE_RATE)[0]
			real = piano[:SAMPLE_RATE * DATA_DURATION]
			small = np.array([real[::COMPRESSION_RATE]])
			pred = tf.squeeze(gen(small))

			time = np.linspace(0, len(pred) / 44100, num=len(pred))
			plt.clf()
			plt.plot(time, real)
			plt.plot(time, pred, alpha=0.5)
			os.makedirs(f"eval/{cnt}/{i}")
			os.chdir(f"eval/{cnt}/{i}")

			scipy.io.wavfile.write(f"real.mp3", SAMPLE_RATE, np.array(real))
			scipy.io.wavfile.write(f"pred.mp3", SAMPLE_RATE, np.array(pred))

			plt.savefig(f"fig")

			os.chdir("../../..")

"""





"""
class EvaluationCallback(Callback):
	def __init__(self):
		self.epoch = 0
		if os.path.exists("eval"):
			shutil.rmtree("eval")
		os.mkdir("eval")

	def on_epoch_begin(self, epoch, logs=None):
		self.epoch = epoch

	def on_train_batch_end(self, batch, logs=None):
		if batch % 10000 != 0:
			return

		#eval_ds = loader.Dataset(1)

		point = Dataset(1)[100]

		piano = librosa.load("../piano.wav", sr=SAMPLE_RATE)[0]
		real = piano[:SAMPLE_RATE * DATA_DURATION]
		small = np.array([real[::COMPRESSION_RATE]])




		pred = tf.squeeze(self.model(small))



		time = np.linspace(0, len(pred) / 44100, num=len(pred))

		plt.clf()
		plt.plot(time, real)
		plt.plot(time, pred, alpha=0.5)
		os.makedirs(f"eval/{self.epoch}/{batch}")
		os.chdir(f"eval/{self.epoch}/{batch}")

		scipy.io.wavfile.write(f"real.mp3", sample_rate, np.array(real))
		scipy.io.wavfile.write(f"pred.mp3", sample_rate, np.array(pred))

		plt.savefig(f"fig")

		os.chdir("../../..")
"""


ds = Dataset(1)

