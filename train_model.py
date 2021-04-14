import pretty_midi
from glob import glob
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models
from keras.utils import to_categorical

#Sources:
#https://medium.com/@anuj_shah/creating-custom-data-generator-for-training-deep-learning-models-part-2-be9ad08f3f0e

l = 1
if l == 1:
    fmidi = 'C:/Users/sherktho/Documents/GitHub/Midi-Arduino-Interface/Dataset/'
    genres = glob(fmidi+'*')
    
    files = []
    labels = []
    count = 0
    for i,genre in enumerate(genres):
        songs = glob(genre+'/*')
        for song in songs:
            try:
                song = pretty_midi.PrettyMIDI(song)
                song = song.get_piano_roll(10)
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
                    interval = int(random.random()*midi_data.shape[1]-51)
                    midi_data = midi_data[:,interval:interval+50]
                    midi_data[midi_data>0]=1
                    midi_data = np.expand_dims(midi_data, axis=0)
                    midi_data = np.expand_dims(midi_data, axis=3)
                    if midi_data.shape[2]!=50:
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


callback_folder = 'C:/Users/sherktho/Documents/GitHub/Midi Models/'

callback = tf.keras.callbacks.ModelCheckpoint(
    filepath = callback_folder+'model.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss', 
    verbose=0, save_best_only=False,
    save_weights_only=False, mode='auto', 
    save_freq='epoch', options=None
)


model = models.Sequential()
model.add(layers.Conv2D(128, (3, 3), activation='relu', input_shape=(25,50,1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(4))

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit_generator(
        train_generator,
        steps_per_epoch=n/32,
        epochs=50,
        validation_data=validation_generator,
        validation_steps=(len(files)-n)/32,
        verbose=1,
        callbacks=callback)

print('end')
test_loss, test_acc = model.evaluate(files[len(files)-50:len(files)],  labels[len(files)-50:len(files)], verbose=2)

plt.plot(model.history['accuracy'], label='accuracy')
plt.plot(model.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')