import os
#os.environ["TF_GPU_ALLOCATOR"]="cuda_malloc_async"
import tensorflow as tf
import loader
from scipy.io.wavfile import write
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras import mixed_precision


from keras import backend as K

def mse(y_true, y_pred):

	y_true = tf.cast(y_true, "float32")
	y_pred = tf.cast(y_pred, "float32")

	#diff = (y_pred - y_true).astype("float32")
	diff = K.cast(y_pred - y_true, "float32")

	sq = K.cast(K.square(diff), "float32")

	mean = K.mean(sq)
	return mean

	return K.sqrt(
		K.mean(
			K.square(K.cast(y_pred, "float32") - K.cast(y_true, "float32")), axis=-1
		)
			
	)


policy = mixed_precision.Policy("mixed_float16")
mixed_precision.set_global_policy(policy)


#physical_devices = tf.config.list_physical_devices("GPU") 
#tf.config.experimental.set_memory_growth(physical_devices[0], True)

import warnings
warnings.filterwarnings("ignore")

"""
class Autoencoder(Model):
	def __init__(self):
		super(Autoencoder, self).__init__()
		self.encoder = tf.keras.Sequential([
			layers.Input(shape=(2176, 1056, 2, 1)),

			layers.Conv3D(2, (3, 3, 1), activation="relu", padding="same"),
			layers.MaxPooling3D(pool_size=(2,2, 1), padding="same"),

			layers.Conv3D(4, (3, 3, 1), activation="relu", padding="same"),
			layers.MaxPooling3D(pool_size=(2,2, 1), padding="same"),

			layers.Conv3D(8, (3, 3, 1), activation="relu", padding="same"),
			layers.MaxPooling3D(pool_size=(2,2, 1)),

			layers.Conv3D(16, (3, 3, 1), activation="relu", padding="same"),
			layers.MaxPooling3D(pool_size=(2,2, 1)),

			layers.Conv3D(32, (3, 3, 1), activation="relu", padding="same"),
			layers.MaxPooling3D(pool_size=(2,2, 1)),
			#layers.Activation("linear", dtype="float32"),
		])

		self.decoder = tf.keras.Sequential([
			layers.Conv3DTranspose(32, kernel_size=(3, 3, 1), strides=(2, 2, 1), activation="relu", padding="same"),
			layers.Conv3DTranspose(16, kernel_size=(3, 3, 1), strides=(2, 2, 1), activation="relu", padding="same"),
			layers.Conv3DTranspose(8, kernel_size=(3, 3, 1), strides=(2, 2, 1), activation="relu", padding="same"),
			layers.Conv3DTranspose(4, kernel_size=(3, 3, 1), strides=(2, 2, 1), activation="relu", padding="same"),
			layers.Conv3DTranspose(2, kernel_size=(3, 3, 1), strides=(2, 2, 1), activation="relu", padding="same"),
			layers.Conv3D(1, (3, 3, 1), activation=None, padding="same")
		])

	def call(self, x):
		encoded = self.encoder(x)
		decoded = self.decoder(encoded)
		return decoded
"""

class Autoencoder(Model):
  def __init__(self):
    super(Autoencoder, self).__init__()
    self.encoder = tf.keras.Sequential([
      layers.Input(shape=(1088, 464, 2, 1)),
	  #layers.Input(shape=(200, 712, 2, 1)),
      layers.Conv3D(2, (3, 3, 1), activation="relu", padding="same"),
      layers.MaxPooling3D(pool_size=(2,2, 1), padding="same"),
      layers.Conv3D(4, (3, 3, 1), activation="relu", padding="same"),
      layers.MaxPooling3D(pool_size=(2,2, 1), padding="same"),
      layers.Conv3D(6, (3, 3, 1), activation="relu", padding="same"),
      layers.MaxPooling3D(pool_size=(2,2, 1)),
	  #layers.Flatten(),
	  #layers.Dense(63104//4)
    ])

    self.decoder = tf.keras.Sequential([
	  layers.Dense(31552),
	  layers.Reshape((None, 68, 29, 2, 8)),
      layers.Conv3DTranspose(8, kernel_size=(3, 3, 1), strides=(2, 2, 1), activation="relu", padding="same"),
      layers.Conv3DTranspose(6, kernel_size=(3, 3, 1), strides=(2, 2, 1), activation="relu", padding="same"),
      layers.Conv3DTranspose(4, kernel_size=(3, 3, 1), strides=(2, 2, 1), activation="relu", padding="same"),
      layers.Conv3D(1, (3, 3, 1), activation=None, padding="same")
    ])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded


autoencoder = Autoencoder()
autoencoder.encoder.summary()
quit()

LEARNING_RATE = 0.01
EPOCHS = 15
opt = tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE)
autoencoder.compile(optimizer=opt, loss="mse")

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
	filepath="models/checkpoints")

print(tf.config.list_physical_devices("GPU"))


#ds = dataloader.MetaBatchLoader(meta_batches=8, batch_size=16)
ds = loader.Dataset(64)
#autoencoder.fit(ds, epochs=10, batch_size=ds.batch_size, shuffle=False, callbacks=[model_checkpoint_callback, WandbCallback()])


autoencoder.fit(
	ds, 
	epochs=EPOCHS, 
	batch_size=ds.batch_size, 
	shuffle=False, 
	callbacks=[model_checkpoint_callback]
)