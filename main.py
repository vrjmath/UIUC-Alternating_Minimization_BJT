import numpy as np
import math
import numpy as np
import scipy.io

from lib import createA, createb, plot_data, descend

a = [0.2, 0.4, 0.2, 0.6, 0.2, 0.2]

dt = 0.0000002
x_path ='/home/viraj2/CAEML/v_50sinv3.mat'
y_path = '/home/viraj2/CAEML/i_50sinv3.mat'

x = scipy.io.loadmat(x_path)['datain']
y = scipy.io.loadmat(y_path)['dataout']

train_samples = int(0.9*x.shape[0])
test_samples = int(0.1*x.shape[0])

train_x = x[0:train_samples]
train_y = y[0:train_samples]
test_x = x[train_samples:train_samples + test_samples]
test_y = y[train_samples:train_samples + test_samples]

epochs = 100

for i in range(epochs):
	A = createA(train_x, dt, a).T
	b = createb(train_y).T
	alpha = np.linalg.lstsq(A, b, rcond=None)[0]
	#train_norm = np.linalg.norm(b - A @ alpha)
	#print("Train Euclidean norm", train_norm)

	test_A = createA(test_x, dt, a).T
	test_b = createb(test_y).T

	test_norm = np.linalg.norm(test_b - test_A @ alpha)
	print("Test Euclidean norm", test_norm)

	a = descend(A.T[1], A.T[2], A.T[3], A.T[4], alpha.T, b.T, a, learning_rate=0.9)
	print(a)
	if(i%5 == 0):
		plot_data(test_b.T, (test_A @ alpha).T)
