import csv
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
import data_gen

# Define hyperparameter 
BATCH_SIZE = 32
SAMPLES_PER_EPOCH = 100
NB_VAL_SAMPLES = 20
EPOCH = 5

# Read in the driving_log.csv 
samples = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)
    samples.pop(0) # Remove the first row, which is title of the form
num_samples = len(samples)
# Compile and train the model using the generator function

train_generator = data_gen.generate_batch(samples, batch_size=BATCH_SIZE)
validation_generator = data_gen.generate_batch(samples, batch_size=BATCH_SIZE)

model = Sequential()
model.add(Lambda(lambda x : x / 127.5 - 1.0, input_shape=(64 ,64 ,3)))
model.add(Convolution2D(24, 5 , 5, border_mode='same', subsample=(2, 2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

model.add(Convolution2D(36, 5 , 5, border_mode='same', subsample=(2, 2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

model.add(Convolution2D(48, 5 , 5, border_mode='same', subsample=(2, 2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

model.add(Convolution2D(64, 3 , 3, border_mode='same', subsample=(2, 2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

model.add(Convolution2D(64, 3 , 3, border_mode='same', subsample=(2, 2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

model.add(Flatten())

model.add(Dense(1164))
model.add(Activation('relu'))

model.add(Dense(100))
model.add(Activation('relu'))

model.add(Dense(50))
model.add(Activation('relu'))

model.add(Dense(10))
model.add(Activation('relu'))

model.add(Dense(1))
model.summary()

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, 
        samples_per_epoch=SAMPLES_PER_EPOCH,
        validation_data=validation_generator, 
        nb_val_samples=NB_VAL_SAMPLES,
        nb_epoch=EPOCH)


model.save('model.h5')
