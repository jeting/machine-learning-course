from keras.models import Sequential
from keras.layers import Dense 

# import data from mnist
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x_train = mnist.train.images
y_train = mnist.train.labels

# design model (1 input, 2 hidden, and 1 output layers)
model = Sequential()

model.add( Dense(input_dim = 28*28, units = 500, activation = 'sigmoid') )

model.add( Dense(units = 500, activation = 'sigmoid') )

model.add( Dense(units = 10, activation = 'softmax') )

# set cost function
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

# training
# batch_size indicates how many training data is used in each batch
print('Start training')
model.fit(x_train, y_train, batch_size = 100, nb_epoch = 20)
model.save('my_model.h5')
print('End training')
