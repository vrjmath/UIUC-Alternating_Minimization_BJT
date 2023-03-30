import math
import numpy as np
import matplotlib.pyplot as plt
# Voltages
# First row: VCE
# Second row: VBE

# Currents
# First row: Ic
# Second row: Ib
# Third row: Ie


def gradient(x, dt):
	return np.gradient(x)/dt

def createA(x, dt, a):
	inputs = []
	samples = x.shape[0]
	length = x.shape[2]
	inputs.append(np.ones(samples*length))
	inputs.append(x[0][0])
	inputs.append(x[0][1])
	inputs.append(gradient(x[0][0], dt))
	inputs.append(gradient(x[0][1], dt))
	inputs.append(np.exp(a[0]*x[0][0]))
	inputs.append(np.exp(a[1]*x[0][0]))
	inputs.append(np.exp(a[2]*x[0][0]))
	inputs.append(np.exp(a[3]*x[0][1]))	
	inputs.append(np.exp(a[4]*x[0][1]))
	inputs.append(np.exp(a[5]*x[0][1]))

	for i in range(samples - 1):
		inputs[1] = np.concatenate((inputs[1], x[i+1][0]), 0)
		inputs[2] = np.concatenate((inputs[2], x[i+1][1]), 0)
		inputs[3] = np.concatenate((inputs[3], gradient(x[i+1][0], dt)), 0)
		inputs[4] = np.concatenate((inputs[4], gradient(x[i+1][1], dt)), 0)
		inputs[5] = np.concatenate((inputs[5], np.exp(a[0]*x[i+1][0])), 0)
		inputs[6] = np.concatenate((inputs[6], np.exp(a[1]*x[i+1][0])), 0)
		inputs[7] = np.concatenate((inputs[7], np.exp(a[2]*x[i+1][0])), 0)
		inputs[8] = np.concatenate((inputs[8], np.exp(a[3]*x[i+1][1])), 0)
		inputs[9] = np.concatenate((inputs[9], np.exp(a[4]*x[i+1][1])), 0)
		inputs[10] = np.concatenate((inputs[10], np.exp(a[5]*x[i+1][1])), 0)

	return np.vstack(inputs)

def createb(y):
	outputs = []
	samples = y.shape[0]
	outputs.append(y[0][0])
	outputs.append(y[0][1])

	for i in range(samples - 1):
		outputs[0] = np.concatenate((outputs[0], y[i+1][0]), 0)
		outputs[1] = np.concatenate((outputs[1], y[i+1][1]), 0)

	return np.vstack(outputs)

def plot_data(true, predicted):
	true = true[1][500:1000]
	predicted = predicted[1][500:1000]

	plt.plot(true)
	plt.plot(predicted)
	plt.show()

def descend(vbc, vbe, vbc_dot, vbe_dot, d, current, a, learning_rate):
	dl_da = [0, 0, 0, 0, 0, 0]
	for k in range(2):
		for i in range(vbc.shape[0]):
			dl_da[0] += 2*(current[k][i] - (d[k][0] + vbc[i]*d[k][1] + vbe[i]*d[k][2] + vbc_dot[i]*d[k][3] + 							vbe_dot[i]*d[k][4] + d[k][5]*np.exp(a[0]*vbc[i]) + d[k][6]*np.exp(a[1]*vbc[i]) + 
						d[k][7]*np.exp(a[2]*vbc[i]) + d[k][8]*np.exp(a[3]*vbe[i]) + d[k][9]*np.exp(a[4]*vbe[i]) + 
						d[k][10]*np.exp(a[5]*vbe[i])))*(-1)*vbc[i]*np.exp(a[0]*vbc[i])
			dl_da[1] += 2*(current[k][i] - (d[k][0] + vbc[i]*d[k][1] + vbe[i]*d[k][2] + vbc_dot[i]*d[k][3] + 							vbe_dot[i]*d[k][4] + d[k][5]*np.exp(a[0]*vbc[i]) + d[k][6]*np.exp(a[1]*vbc[i]) + 
						d[k][7]*np.exp(a[2]*vbc[i]) + d[k][8]*np.exp(a[3]*vbe[i]) + d[k][9]*np.exp(a[4]*vbe[i]) + 
						d[k][10]*np.exp(a[5]*vbe[i])))*(-1)*vbc[i]*np.exp(a[1]*vbc[i])
			dl_da[2] += 2*(current[k][i] - (d[k][0] + vbc[i]*d[k][1] + vbe[i]*d[k][2] + vbc_dot[i]*d[k][3] + 							vbe_dot[i]*d[k][4] + d[k][5]*np.exp(a[0]*vbc[i]) + d[k][6]*np.exp(a[1]*vbc[i]) + 
						d[k][7]*np.exp(a[2]*vbc[i]) + d[k][8]*np.exp(a[3]*vbe[i]) + d[k][9]*np.exp(a[4]*vbe[i]) + 
						d[k][10]*np.exp(a[5]*vbe[i])))*(-1)*vbc[i]*np.exp(a[2]*vbc[i])		
			dl_da[3] += 2*(current[k][i] - (d[k][0] + vbc[i]*d[k][1] + vbe[i]*d[k][2] + vbc_dot[i]*d[k][3] + 							vbe_dot[i]*d[k][4] + d[k][5]*np.exp(a[0]*vbc[i]) + d[k][6]*np.exp(a[1]*vbc[i]) + 
						d[k][7]*np.exp(a[2]*vbc[i]) + d[k][8]*np.exp(a[3]*vbe[i]) + d[k][9]*np.exp(a[4]*vbe[i]) + 
						d[k][10]*np.exp(a[5]*vbe[i])))*(-1)*vbc[i]*np.exp(a[3]*vbe[i])
			dl_da[4] += 2*(current[k][i] - (d[k][0] + vbc[i]*d[k][1] + vbe[i]*d[k][2] + vbc_dot[i]*d[k][3] + 							vbe_dot[i]*d[k][4] + d[k][5]*np.exp(a[0]*vbc[i]) + d[k][6]*np.exp(a[1]*vbc[i]) + 
						d[k][7]*np.exp(a[2]*vbc[i]) + d[k][8]*np.exp(a[3]*vbe[i]) + d[k][9]*np.exp(a[4]*vbe[i]) + 
						d[k][10]*np.exp(a[5]*vbe[i])))*(-1)*vbc[i]*np.exp(a[4]*vbe[i])
			dl_da[5] += 2*(current[k][i] - (d[k][0] + vbc[i]*d[k][1] + vbe[i]*d[k][2] + vbc_dot[i]*d[k][3] + 							vbe_dot[i]*d[k][4] + d[k][5]*np.exp(a[0]*vbc[i]) + d[k][6]*np.exp(a[1]*vbc[i]) + 
						d[k][7]*np.exp(a[2]*vbc[i]) + d[k][8]*np.exp(a[3]*vbe[i]) + d[k][9]*np.exp(a[4]*vbe[i]) + 
						d[k][10]*np.exp(a[5]*vbe[i])))*(-1)*vbc[i]*np.exp(a[5]*vbe[i])	
	for i in range(6):
		a[i] = a[i] - learning_rate*dl_da[i]
	
	return a

