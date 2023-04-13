from keras.datasets import mnist
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression


(x_train, y_train), (x_test, y_test) = mnist.load_data()

# show the shape of dataset
print('CONJUNTO ORIGINAL: ')
print('X_train: ' + str(x_train.shape))
print('Y_train: ' + str(y_train.shape))
print('X_test:  '  + str(x_test.shape))
print('Y_test:  '  + str(y_test.shape))

# test the dataset
print(y_test[0])
plt.imshow(x_test[0])
plt.show()

# subset of dataset (0 and 8)
train_filter = np.where((y_train == 0) | (y_train == 8))
test_filter = np.where((y_test == 0) | (y_test == 8))
x_train, y_train = x_train[train_filter], y_train[train_filter]
x_test, y_test = x_test[test_filter], y_test[test_filter]

# test the y_train : just 0 and 8
print(y_train)

# show the shape of dataset
print('SOLO 8 Y 0: ')
print('X_train: ' + str(x_train.shape))
print('Y_train: ' + str(y_train.shape))
print('X_test:  '  + str(x_test.shape))
print('Y_test:  '  + str(y_test.shape))
print(type(x_test))

# flat the dataset
num_pixels = x_train.shape[1] * x_train.shape[2] # find size of one-dimensional vector
x_train = x_train.reshape(x_train.shape[0], num_pixels).astype('float32') # flatten training images
x_test = x_test.reshape(x_test.shape[0], num_pixels).astype('float32') # flatten test images

# setup regression
regresion = LogisticRegression()
regresion.fit(x_train, y_train)
y_pred = regresion.predict(x_test)
print(y_pred)

