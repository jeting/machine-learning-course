# Import Keras libraries and packages
from keras.models import Sequential  #用來啟動 NN
from keras.layers import Conv2D  # Convolution Operation
from keras.layers import MaxPooling2D # Pooling
from keras.layers import Flatten
from keras.layers import Dense # Fully Connected Networks

# import data from mnist
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x_train = mnist.train.images
y_train = mnist.train.labels

# initializing CNN
model = Sequential()  
model.add(Conv2D(32, 3, 3, input_shape = (1,28, 28), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

# Second convolutional layer
model.add(Conv2D(32, 3, 3, activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

# Third convolutional layer
model.add(Conv2D(64, 3, 3, activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(output_dim = 128, activation = 'relu'))
model.add(Dense(output_dim = 1, activation = 'sigmoid'))

# set cost function
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# training
# batch_size indicates how many training data is used in each batch
print('Start training')
model.fit(x_train, y_train, batch_size = 100, nb_epoch = 20)
model.save('my_model.h5')
print('End training')

