import csv
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Activation, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
import data_gen

# Define hyperparameter 
BATCH_SIZE = 64
SAMPLES_PER_EPOCH = 25600
NB_VAL_SAMPLES = 5000
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
model.add(Lambda(lambda x : x / 255.0 - 0.5, input_shape=(64, 64 ,3)))
model.add(Convolution2D(32 ,5 ,5, subsample=(2, 2), border_mode='same'))
model.add(LeakyReLU())

model.add(Convolution2D(32, 3, 3, subsample=(1, 1), border_mode='same'))
model.add(LeakyReLU())
model.add(Dropout(.3))
model.add(MaxPooling2D((2, 2), border_mode='valid'))

model.add(Convolution2D(16, 3, 3, subsample=(1, 1), border_mode='same'))
model.add(LeakyReLU())
model.add(Dropout(.3))

model.add(Flatten())
model.add(Dense(1024))
model.add(Dense(256))
model.add(Dense(64))
model.add(Dense(16))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, 
        samples_per_epoch=SAMPLES_PER_EPOCH,
        validation_data=validation_generator, 
        nb_val_samples=NB_VAL_SAMPLES,
        nb_epoch=EPOCH)


model.save('model.h5')
