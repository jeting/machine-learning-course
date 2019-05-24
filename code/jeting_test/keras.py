import keras

import numpy as np
import matplotlib.pyplot as plt

# import data from mnist
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x_train = mnist.train.images
y_train = mnist.train.labels

#print(type(x_train))
#print(type(y_train))

# design model (1 input, 2 hidden, and 1 output layers)
model = keras.Sequential()

model.add( keras.layers.Dense(input_dim = 28*28, output_dim = 500) )
model.add( keras.layers.Activation('sigmoid') )

model.add( keras.layers.Dense(output_dim = 500) )
model.add( keras.layers.Activation('sigmoid') )

model.add( keras.layers.Dense(output_dim = 10) )
model.add( keras.layers.Activation('softmax') )

# set cost function
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

# training
model.fit(x_train, y_train, batch_size = 100, nb_epoch = 20)


