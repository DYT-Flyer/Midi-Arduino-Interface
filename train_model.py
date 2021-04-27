import pretty_midi
from glob import glob
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.utils import plot_model

#Sources:
#https://medium.com/@anuj_shah/creating-custom-data-generator-for-training-deep-learning-models-part-2-be9ad08f3f0e

l = 1
if l == 1:
    fmidi = 'C:/Users/remove/Documents/GitHub/Midi-Arduino-Interface/Dataset/'
    genres = glob(fmidi+'*')
    
    files = []
    labels = []
    for i,genre in enumerate(genres):
        print(i)
        count = 0
        songs = glob(genre+'/*')
        for song in songs:
            print(count)
            try:
                if count < 2200:
                    song = pretty_midi.PrettyMIDI(song)
                    song = song.get_piano_roll(5)
                    song = song[46:71,:]
                    files.append(song)
                    labels.append(i)
                    count += 1
            except:
                print(count)

def generator(files,labels,batch_size=32,shuffle_data=True,resize=224):
    count = 0
    num_files = len(files)
    while True: # Loop forever so the generator never terminates
        temp = list(zip(files, labels))
        random.shuffle(temp)
        files, labels = zip(*temp)
        
        for i in range(0, num_files, batch_size):
            # Get the samples you'll use in this batch
            try:
                batch_samples = files[i:i+batch_size]
            except:
                batch_samples = files[i:num_files-1]
    
            # Initialise X_train and Y_train arrays for this batch
            X_train = []
            Y_train = []

            # For each example
            for j,batch_sample in enumerate(batch_samples):
                # Load song data
                try:
                    #midi_data = pretty_midi.PrettyMIDI(batch_sample)
                    #midi_data = midi_data.get_piano_roll(10);
                    midi_data = files[i+j]
                    interval = int(random.random()*midi_data.shape[1]-26)
                    midi_data = midi_data[:,interval:interval+25]
                    midi_data[midi_data>0]=1
                    midi_data = np.expand_dims(midi_data, axis=0)
                    midi_data = np.expand_dims(midi_data, axis=3)
                    if midi_data.shape[2]!=25:
                        1/0
                    # Add example to arrays
                    X_train.append(midi_data)
                    Y_train.append(labels[j])
                except:
                    count += 1
                    
            # Make sure they're numpy arrays (as opposed to lists)
            X_train = np.concatenate(X_train, axis=0)
            Y_train = np.array(Y_train)
            # The generator-y part: yield the next training batch            
            yield X_train, Y_train

temp = list(zip(files, labels))
random.shuffle(temp)
files, labels = zip(*temp)

#shape = 128 by 50        
n = int(len(files)*.8)
labels = to_categorical(labels)
train_generator = generator(files[1:n], labels[1:n], batch_size=32)
validation_generator = generator(files[n:len(files)-50], labels[n:len(files)-50], batch_size=32)


callback_folder = 'C:/Users/remove/Documents/GitHub/Midi Models/'

callback = tf.keras.callbacks.ModelCheckpoint(
    filepath = callback_folder+'model.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss', 
    verbose=0, save_best_only=False,
    save_weights_only=False, mode='auto', 
    save_freq='epoch', options=None
)

model = models.Sequential()
model.add(layers.Conv2D(128, (3, 3), activation='relu', input_shape=(25,25,1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(5))
 
# function for creating a vgg block
def vgg_block(layer_in, n_filters, n_conv):
     # add convolutional layers
     for _ in range(n_conv):
        layer_in = Conv2D(n_filters, (3,3), padding='same', activation='relu')(layer_in)
     # add max pooling layer
     layer_in = MaxPooling2D((2,2), strides=(2,2))(layer_in)
     return layer_in
 
# define model input
visible = Input(shape=(25, 25, 1))
# add vgg module
layer = vgg_block(visible, 64, 2)
# add vgg module
layer = vgg_block(layer, 128, 2)
# add vgg module
layer = vgg_block(layer, 256, 4)

x = layers.Flatten()(layer)
x = (layers.Dense(256, activation='relu'))(x)
x = (layers.Dropout(0.5))(x)
x = (layers.Dense(32, activation='relu'))(x)
x = (layers.Dropout(0.5))(x)
x = (layers.Dense(5, activation='softmax'))(x)

# create model
model = Model(inputs=visible, outputs=x)
# summarize model
model.summary()

import keras

# example of a CNN model with an identity or projection residual module
from keras.models import Model
from keras.layers import Input
from keras.layers import Activation
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import add
from keras.utils import plot_model

# function for creating an identity or projection residual module
def residual_module(layer_in, n_filters):
	merge_input = layer_in
	# check if the number of filters needs to be increase, assumes channels last format
	if layer_in.shape[-1] != n_filters:
		merge_input = Conv2D(n_filters, (1,1), padding='same', activation='relu', kernel_initializer='he_normal')(layer_in)
	# conv1
	conv1 = Conv2D(n_filters, (3,3), padding='same', activation='relu', kernel_initializer='he_normal')(layer_in)
	# conv2
	conv2 = Conv2D(n_filters, (3,3), padding='same', activation='linear', kernel_initializer='he_normal')(conv1)
	# add filters, assumes filters/channels last
	layer_out = add([conv2, merge_input])
	# activation function
	layer_out = Activation('relu')(layer_out)
	return layer_out

import cv2
import numpy as np
import os
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
import keras
from keras.models import Sequential, Model,load_model
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D,MaxPool2D
from keras.preprocessing import image
from keras.initializers import glorot_uniform

def identity_block(X, f, filters, stage, block):
   
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    F1, F2, F3 = filters

    X_shortcut = X
   
    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2a', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    X = Add()([X, X_shortcut])# SKIP Connection
    X = Activation('relu')(X)

    return X

def convolutional_block(X, f, filters, stage, block, s=2):
   
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    F1, F2, F3 = filters

    X_shortcut = X

    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base + '2a', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    X_shortcut = Conv2D(filters=F3, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base + '1', kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis=3, name=bn_name_base + '1')(X_shortcut)

    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X

def ResNet50(input_shape=(224, 224, 3)):

    X_input = Input(input_shape)

    X = ZeroPadding2D((3, 3))(X_input)

    X = Conv2D(64, (7, 7), strides=(2, 2), name='conv1', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name='bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    X = convolutional_block(X, f=3, filters=[64, 64, 256], stage=2, block='a', s=1)
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='c')


    X = convolutional_block(X, f=3, filters=[128, 128, 512], stage=3, block='a', s=2)
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='b')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='c')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='d')

    X = convolutional_block(X, f=3, filters=[256, 256, 1024], stage=4, block='a', s=2)
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='b')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='c')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='d')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='e')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='f')

    X = X = convolutional_block(X, f=3, filters=[512, 512, 2048], stage=5, block='a', s=2)
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='b')
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='c')

    X = AveragePooling2D(pool_size=(2, 2), padding='same')(X)
    
    model = Model(inputs=X_input, outputs=X, name='ResNet50')

    return model


# base_model = ResNet50(input_shape=(25, 25, 1))
# #headModel = base_model.output
# headModel = Flatten()(base_model)
# headModel=Dense(256, activation='relu', name='fc1',kernel_initializer=glorot_uniform(seed=0))(headModel)
# headModel =layers.Dropout(0.5)(headModel)
# headModel=Dense(128, activation='relu', name='fc2',kernel_initializer=glorot_uniform(seed=0))(headModel)
# headModel=layers.Dropout(0.5)(headModel)
# headModel = Dense(5,activation='softmax', name='fc3',kernel_initializer=glorot_uniform(seed=0))(headModel)

# model = headModel()

# # define model input
# visible = Input(shape=(25,25,1))
# # add vgg module
# layer = residual_module(visible, 64)
# # create model
# model = Model(inputs=visible, outputs=layer)
# # summarize model
# model.summary()
# # plot model architecture
# plot_model(model, show_shapes=True, to_file='residual_module.png')

sgd = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer='sgd',
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-5)

model.fit_generator(
        train_generator,
        steps_per_epoch=n/32,
        epochs=250,
        validation_data=validation_generator,
        validation_steps=(len(files)-n)/32,
        verbose=1,
        callbacks=[callback])

print('end')
test_loss, test_acc = model.evaluate(files[len(files)-50:len(files)],  labels[len(files)-50:len(files)], verbose=2)

plt.plot(model.history['accuracy'], label='accuracy')
plt.plot(model.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')