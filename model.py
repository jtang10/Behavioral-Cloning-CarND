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
model.add(Lambda(lambda x : x / 255.0 - 0.5, input_shape=(160 ,320 ,3)))
model.add(Cropping2D(cropping=((70, 25), (0, 0))))
model.add(Convolution2D(6 ,5 ,5,activation="relu"))
model.add(MaxPooling2D())
model.add(Convolution2D(6, 5, 5,activation="relu"))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, 
        samples_per_epoch=SAMPLES_PER_EPOCH,
        validation_data=validation_generator, 
        nb_val_samples=NB_VAL_SAMPLES,
        nb_epoch=EPOCH)


model.save('model.h5')
