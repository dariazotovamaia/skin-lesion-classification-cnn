import tensorflow as tf
from keras import backend as k

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.23
k.tensorflow_backend.set_session(tf.Session(config=config))

from keras.applications import VGG16

# Get VGG16 model as a baseline
conv_base = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(224, 224, 3))
				  
import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import CSVLogger
from keras import models
from keras import layers
from keras import optimizers


model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))

# Freezing all layers up to a specific one
conv_base.trainable = True

set_trainable = False
for layer in conv_base.layers:
  if layer.name == 'block5_conv1':
    set_trainable = True
  if set_trainable:
    layer.trainable = True
  else:
    layer.trainable = False

train_dir = '/challenge1/set6000'

batch_size = 20
n_epochs = 50

# Training the model end to end with a frozen convolutional base
# Training with augmentation
train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')


train_generator = train_datagen.flow_from_directory(
        # Target directory
        train_dir,
        # All images will be resized to 224x224
        target_size=(224, 224),
        batch_size=20,
        # For binary labels
        class_mode='binary')


model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-5),
              metrics=['acc
			  

csv_logger = CSVLogger('/challenge1/model_50epochs_6000_loss_acc.log')
history = model.fit_generator(
      train_generator,
      steps_per_epoch=100,
      epochs=n_epochs,
      verbose=2,
	  callbacks=[csv_logger])
	  
	  
model.save('/challenge1/vgg16_finetuning_50epoches_6000.h5')
