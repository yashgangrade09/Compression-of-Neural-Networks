## This is a utility file to load the MNIST Dataset correctly ## 


## Reference: https://keras.io/datasets/
from keras.datasets import mnist
from keras.utils import np_utils as util 
from numpy import transpose as T

def get_mnist():
	rows = 28
	cols = 28
	categories = 10
	(X_Train, Y_Train), (X_Test, Y_Test) = mnist.load_data()

	# Resizing the array and taking the Transpose of the array 
	X_Train = (T(X_Train.reshape(X_Train.shape[0], 1, rows, cols), axes= [0, 2, 3, 1])).astype('float32')
	X_Test = (T(X_Test.reshape(X_Test.shape[0], 1, rows, cols), axes= [0, 2, 3, 1])).astype('float32')

	X_Train /= 255
	X_Test /= 255

	# Get the Binary Class Matrices from the Class/Category Vectors
	Y_Train = util.to_categorical(Y_Train, categories)
	Y_Test = util.to_categorical(Y_Test, categories)

	return X_Train[0:10000], X_Test[0:1000], Y_Train[0:10000], Y_Test[0:1000], rows, cols, categories