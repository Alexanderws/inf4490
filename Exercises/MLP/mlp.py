import numpy as np

class mlp:
    def __init__(self, inputs, targets, n_hidden_neurons):
    	self.n_inputs = len(inputs[0])
    	self.n_outputs = len(targets[0])
    	self.n_hidden_neurons = n_hidden_neurons
    	self.beta = 1
    	self.eta = 0.01
    	self.momentum = 0.0

    	self.weights_ih = np.random.random((self.n_inputs, self.n_hidden_neurons))
    	self.weights_ho = np.random.random((self.n_hidden_neurons, self.n_outputs))

    def __repr__(self):
    	return "MLP - Inputs: {} - Outputs: {} - Hidden Neurons: {}\nWeights ih: {} - Weights ho: {} \n".format(self.n_inputs, self.n_outputs, self.n_hidden_neurons, self.weights_ih, self.weights_ho)

    def forward(self, inputs):
    	sum_1 = sigmoid(np.dot(inputs, self.weights_ih))
    	sum_2 = sigmoid(np.dot(sum_1, self.weights_ho))
    	return sum_2

    def train(self, inputs, outputs, iterations=1):
    	for x in range(iterations):
	    	layer_0 = inputs
	    	print("Layer 0 (2 nums)- {}".format(layer_0))
	    	print("weights_ih (2 x 2)- {}".format(self.weights_ih))

	    	layer_1 = sigmoid(np.dot(layer_0, self.weights_ih))
	    	print("Layer 1 (2 nums)- {}".format(layer_1))
	    	print("weights_ho (2 x 1) - {}".format(self.weights_ho))
	    	layer_2 = sigmoid(np.dot(layer_1, self.weights_ho)) 
	    	print("Layer 2 (1 num)- {}".format(layer_2))

	    	layer_2_error = outputs - layer_2
	    	print("Layer 2 error (? num)- {}".format(layer_2_error))

	    	layer_2_delta = np.multiply(layer_2_error, sigmoid_der(layer_2))
	    	print("Layer 2 delta (? num)- {}".format(layer_2_delta))
	    	layer_1_error = np.dot(layer_2_delta, self.weights_ho.T)
	    	print("Layer 1 eror (? num)- {}".format(layer_1_error))
	    	layer_1_delta = np.multiply(layer_1_error, sigmoid_der(layer_1))
	    	print("Layer 1 delta (? num)- {}".format(layer_1_delta))
	    	print("weights_ih - {}".format(self.weights_ih))
	    	print("_____________")
	    	self.weights_ih += ((-1) * self.eta * layer_1_delta) #np.dot(layer_0.T, layer_1_delta)#
	    	print("weights_ih - {}".format(self.weights_ih))
	    	self.weights_ho += ((-1) * self.eta * layer_2_delta) #np.dot(layer_1.T, layer_2_delta) #


def sigmoid(x):
	return 1/(1 + np.exp(-x))

def sigmoid_der(x):
	return x*(1 - x)





