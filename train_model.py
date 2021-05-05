import pretty_midi
import os
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
from vgg_cnn import *

#Sources:
#https://medium.com/@anuj_shah/creating-custom-data-generator-for-training-deep-learning-models-part-2-be9ad08f3f0e

#There are more efficient ways to load these files - I just loaded them as is
#Control allows the user to toggle loading when debugging
l = 1
if l == 1:
    fmidi = os.cwd() + '/Dataset/'
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

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-5)

callback = tf.keras.callbacks.ModelCheckpoint(
    filepath = callback_folder+'model.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss', 
    verbose=0, save_best_only=False,
    save_weights_only=False, mode='auto', 
    save_freq='epoch', options=None
)

model = vgg_cnn()
# summarize model
model.summary()

sgd = tf.keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer='sgd',
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit_generator(
        train_generator,
        steps_per_epoch=n/32,
        epochs=250,
        validation_data=validation_generator,
        validation_steps=(len(files)-n)/32,
        verbose=1,
        callbacks=[callback])

test_loss, test_acc = model.evaluate(files[len(files)-50:len(files)],  labels[len(files)-50:len(files)], verbose=2)