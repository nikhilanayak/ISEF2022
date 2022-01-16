import librosa
import numpy as np
import threading
from tensorflow.keras.utils import Sequence

file = "/mnt/maestro-v3.0.0/repeat.flac"
DATA_DURATION = 2
SAMPLE_RATE = 20480
duration = 731119
num_samples = duration // DATA_DURATION
COMPRESSION_RATE = 32


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

		resampled = np.mean(raw.reshape(-1, COMPRESSION_RATE), axis=1)
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

