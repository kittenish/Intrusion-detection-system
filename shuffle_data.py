import numpy as np
import random

def shuffle(x_train, y_train, num):

	x = np.linspace(0, num - 1, num)
	random.shuffle(x)

	X_train = np.zeros([num, 7])
	Y_train = np.zeros((num,), dtype=np.int)

	for i in range(num):
		X_train[i,:] = x_train[int(x[i]),:]
		Y_train[i] = y_train[int(x[i])]

	print 'finish train data shuffle'
	return X_train, Y_train