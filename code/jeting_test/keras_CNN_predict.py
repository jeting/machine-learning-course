from keras.models import load_model
import numpy as np

# load model
model = load_model('my_CNN_model.h5')

from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_size = x_test.shape
x_test = x_test.reshape(x_size[0], x_size[1], x_size[2], 1)

# predict 
y_predict = model.predict(x_test[0:10,:])
for t in y_predict:
    print(np.argmax(t))
    
print(y_test[0:10])
