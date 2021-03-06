#!/usr/bin/env Python3
'''
    This file will read in data and start your mlp network.
    You can leave this file mostly untouched and do your
    mlp implementation in mlp.py.
'''
# Feel free to use numpy in your MLP if you like to.
import numpy as np
import mlp
import random
import statistics
import math


filename = '../data/movements_day1-3.dat'

movements = np.loadtxt(filename,delimiter='\t')

# Subtract arithmetic mean for each sensor. We only care about how it varies:
movements[:,:40] = movements[:,:40] - movements[:,:40].mean(axis=0)

# Find maximum absolute value:
imax = np.concatenate(  ( movements.max(axis=0) * np.ones((1,41)) ,
                          np.abs( movements.min(axis=0) * np.ones((1,41)) ) ),
                          axis=0 ).max(axis=0)

# Divide by imax, values should now be between -1,1
movements[:,:40] = movements[:,:40]/imax[:40]

# Generate target vectors for all inputs 2 -> [0,1,0,0,0,0,0,0]
target = np.zeros((np.shape(movements)[0],8));
for x in range(1,9):
    indices = np.where(movements[:,40]==x)
    target[indices,x-1] = 1

# Randomly order the data
order = list(range(np.shape(movements)[0]))
np.random.shuffle(order)
movements = movements[order,:]
target = target[order,:]


# Try networks with different number of hidden nodes:
hidden = 6
iterations = 10
learning_rate = 0.01
beta = 1
momentum = 0.3
outtype = 'logistic'
k = 3
n_n = []


# Old routine, now its own function, everything else is the same
def normal(movements, target, iterations=10, hidden=6, outtype='logistic'):
	# Split data into 3 sets

	# Training updates the weights of the network and thus improves the network
	train = movements[::2,0:40]
	train_targets = target[::2]

	# Validation checks how well the network is performing and when to stop
	valid = movements[1::4,0:40]
	valid_targets = target[1::4]

	# Test data is used to evaluate how good the completely trained network is.
	test = movements[3::4,0:40]
	test_targets = target[3::4]

	# Initialize the network:
	net = mlp.mlp(train, train_targets, hidden, beta, momentum, outtype)

	# Run training:
	net.earlystopping(train, train_targets, valid, valid_targets,learning_rate,iterations)

	# Check how well the network performed:
	print("HN: {} - Iter: {} - LR: {}".format(hidden, iterations, learning_rate))
	print("Momentum: {} - Type: {}" .format(momentum, outtype))

	net.confusion(test,test_targets)



# K-folds cross-validation
def kfolds(movements, target, iterations=10, hidden=6, k=5, outtype='logistic'):

	n_networks = []

	for i in range(k):
		test = movements[i::k,0:40]
		test_targets = target[i::k]

		valid = movements[((i+1)%k)::k,0:40]
		valid_targets = target[((i+1)%k)::k]

		train = [item for sublist in [movements[((i + j) % k)::k,0:40] for j in range(2, k)] for item in sublist]
		train_targets = [item for sublist in [target[((i + j) % k)::k] for j in range(2, k)] for item in sublist]

		# Initialize the network:
		net = mlp.mlp(train, train_targets, hidden, beta, momentum, outtype)

		# Store error and percentage correct:
		net.earlystopping(train, train_targets, valid, valid_targets,learning_rate)
		net.confusion(test, valid_targets)

		# Store the neural network:
		n_networks.append(net)

	percentages = []
	errors = []
	for i in range(len(n_networks)):
		percentages.append(n_networks[i].percentage)
		errors.append(n_networks[i].errors)

	print("Average Percentage: {}".format(statistics.mean(percentages)))
	print("Standard deviation: {}".format(statistics.stdev(percentages)))


if __name__ == "__main__":
	#normal(movements, target, iterations, hidden, outtype)
	kfolds(movements, target, iterations, hidden, k, outtype)

