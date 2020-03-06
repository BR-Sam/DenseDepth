# %% md
from model import DepthEstimate
from data import DataLoader
import tensorflow
from loss import depth_loss_function
import os

# Parameters

# Allows more memory usage - Still not enough
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
# End of memory patch

batch_size = 8
learning_rate = 0.0001
epochs = 10

# strategy = tensorflow.distribute.OneDeviceStrategy(device="/gpu:0")
# with strategy.scope():
model = DepthEstimate()
dl = DataLoader()
train_generator = dl.get_batched_dataset(batch_size)

print('Data loader ready.')

optimizer = tensorflow.keras.optimizers.Adam(lr=learning_rate, amsgrad=True)
model.compile(loss=depth_loss_function, optimizer=optimizer)

# Create checkpoint callback
checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tensorflow.keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=True, verbose=1)

# Start training
# model.fit(train_generator, epochs=5, steps_per_epoch=dl.length // batch_size, shuffle=True, callbacks=[cp_callback])
model.fit(train_generator, epochs=3, steps_per_epoch=100, shuffle=True, callbacks=[cp_callback])

converter = tensorflow.lite.TFLiteConverter.from_keras_model(model)
tfl_model = converter.convert()

open("./depth_trained2.tflite", "wb").write(tfl_model)

