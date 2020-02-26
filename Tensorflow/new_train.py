#%% md

# Parameters

#%%

batch_size     = 8
learning_rate  = 0.0001
epochs         = 10

#%% md

# Define model

#%%

from model import DepthEstimate

model = DepthEstimate()

#%% md

# Data loader

#%%

from data import DataLoader

dl = DataLoader()
train_generator = dl.get_batched_dataset(batch_size)

print('Data loader ready.')

#%% md

# Compile & Train

#%%

import tensorflow
from loss import depth_loss_function

optimizer = tensorflow.keras.optimizers.Adam(lr=learning_rate, amsgrad=True)

model.compile(loss=depth_loss_function, optimizer=optimizer)

#%%

# Create checkpoint callback
import os
checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tensorflow.keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=True, verbose=1)

#%%

# Start training
model.fit(train_generator, epochs=5, steps_per_epoch=dl.length//batch_size, shuffle=True, callbacks=[cp_callback])
