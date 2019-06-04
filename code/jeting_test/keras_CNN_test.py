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

x_test = mnist.test.images
y_test = mnist.test.labels


print(x_train.shape)
print(y_train.shape)

print(x_test.shape)
print(y_test.shape)

## Initializing CNN
#model = Sequential()  
#model.add(Conv2D(32, 3, 3, input_shape = (128, 128, 3), activation = 'relu'))
#model.add(MaxPooling2D(pool_size = (2, 2)))
#
## Second convolutional layer
#model.add(Conv2D(32, 3, 3, activation = 'relu'))
#model.add(MaxPooling2D(pool_size = (2, 2)))
#
## Third convolutional layer
#model.add(Conv2D(64, 3, 3, activation = 'relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#
#model.add(Flatten())
#
#model.add(Dense(output_dim = 128, activation = 'relu'))
#model.add(Dense(output_dim = 1, activation = 'sigmoid'))
#
## Set cost function
#model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#training_set = train_datagen.flow_from_directory('dataset/training_set', target_size = (128, 128), batch_size = 32, class_mode = 'binary')
#test_set = test_datagen.flow_from_directory('dataset/test_set', target_size = (128, 128), batch_size = 32, class_mode = 'binary')
#model.fit_generator(training_set, samples_per_epoch = 400, nb_epoch = 30, validation_data = test_set, nb_val_samples = 100)
#
#import numpy as np
#from keras.preprocessing import image
#
#test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size = (128, 128))
#test_image = image.img_to_array(test_image)
#test_image = np.expand_dims(test_image, axis = 0)
#result = model.predict(test_image)
#training_set.class_indices
#if result[0][0] == 1:
#  prediction = 'dog'
#else:
#  prediction = 'cat'
  
  
  