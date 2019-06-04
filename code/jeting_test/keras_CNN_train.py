# Data preprocessing
# import data from mnist
from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# plot image
import matplotlib.pyplot as plt
plt.imshow(x_train[0])
print(y_train[0])

# reshape data
x_size = x_train.shape
x_train = x_train.reshape(x_size[0], x_size[1], x_size[2], 1)

x_size = x_test.shape
x_test = x_test.reshape(x_size[0], x_size[1], x_size[2], 1)

# one-hot encode target column
from keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Builing model
from keras.models import Sequential  #用來啟動 NN
from keras.layers import Conv2D  # Convolution Operation
from keras.layers import MaxPooling2D # Pooling
from keras.layers import Flatten
from keras.layers import Dense # Fully Connected Networks

# Initializing CNN
model = Sequential()  
model.add(Conv2D(64, kernel_size = 3, input_shape = (28, 28, 1), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

# Second convolutional layer
model.add(Conv2D(32, kernel_size = 3, activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Flatten())

model.add(Dense(output_dim = 10, activation = 'softmax'))

# Set cost function
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# train the model
model.fit(x_train, y_train, validation_data = (x_test, y_test), epochs = 3)

model.save('my_CNN_model.h5')
  



