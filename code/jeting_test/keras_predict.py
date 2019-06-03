from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt

# load model
model = load_model('my_model.h5')

# import data from mnist
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x_train = mnist.train.images
y_train = mnist.train.labels

x_test = mnist.test.images
y_test = mnist.test.labels

score = model.evaluate(x_train, y_train)
print("Total loss on training set: {}".format(score[0]))
print("Accurary of training set: {}".format(score[1]))

score = model.evaluate(x_test, y_test)
print("Total loss on testing set: {}".format(score[0]))
print("Accurary of testing set: {}".format(score[1]))

#image_id = 5521
#curr_img   = np.reshape(x_test[image_id,:], (28, 28)) # 28 by 28 matrix 
#curr_label = np.argmax(y_test[image_id,:]) # Label
#plt.matshow(curr_img, cmap=plt.get_cmap('gray'))
#plt.title("The Training Data " 
#              + "Label is " + str(curr_label)) 
#
#y_predict = model.predict(x_test[image_id,:].reshape(1,-1))
#
#val = np.argmax(y_predict)
#
#print("The input image is {}".format(val))
