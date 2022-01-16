import librosa
import matplotlib.pyplot as plt
import scipy.io.wavfile
import numpy as np
import tensorflow as tf

DATA_DURATION = 2
SAMPLE_RATE = 20480
COMPRESSION_RATE = 16

piano = librosa.load("../piano.wav", sr=SAMPLE_RATE)[0]
real = piano[:SAMPLE_RATE * DATA_DURATION]

small = np.mean(real.reshape(-1, COMPRESSION_RATE), axis=1)
small = np.squeeze(small)
small = small[None, ..., None]


pred = f.squeeze(self.model(small))

time = np.linspace(0, len(pred) / 44100, num=len(pred))

plt.clf()
plt.plot(time, real)
plt.plot(time, pred, alpha=0.5)
os.makedirs(f"eval/{self.epoch}/{batch}")
os.chdir(f"eval/{self.epoch}/{batch}")

scipy.io.wavfile.write(f"real.mp3", SAMPLE_RATE, np.array(real))
scipy.io.wavfile.write(f"pred.mp3", SAMPLE_RATE, np.array(pred))

plt.savefig(f"fig")