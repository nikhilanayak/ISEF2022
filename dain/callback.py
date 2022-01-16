import os
import shutil
from keras.callbacks import Callback
import dataset
import librosa

class EvaluationCallback(Callback):
	def __init__(self):
		self.epoch = 0
		if os.path.exists("eval"):
			shutil.rmtree("eval")
		os.mkdir("eval")

		if os.path.exists("model"):
			shutil.rmtree("model")
		os.mkdir("model")

	def on_epoch_begin(self, epoch, logs=None):
		self.epoch = epoch

	def on_epoch_end(self, epoch):
		self.model.save(f"model/{epoch}")

	def on_train_batch_end(self, batch, logs=None):
		if batch % (100) != 0:
			return


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

