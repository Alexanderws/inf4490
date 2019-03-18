import numpy as np


class mlp:
	def __init__(self, inputs, targets, n_hidden):
		self.n_inputs = len(inputs[0])
		self.n_outputs = len(targets[0])
		self.n_hidden = n_hidden
		self.beta = 1
		self.eta = 0.01
		self.momentum = 0.0
		self.weights_ih = np.array([[0.2, 0.5], [0.8,0.4]])
		self.weights_ho = np.array([[0.2], [0.5]])
		#self.weights_ih = np.random.random((self.n_inputs, self.n_hidden))
		#self.weights_ho = np.random.random((self.n_hidden, self.n_outputs))
		print("weights_ih: {}".format(self.weights_ih))
		print("weights_ho: {}".format(self.weights_ho))

	#def forward(self, inputs):

	#def test(self, inputs):


def sigmoid(x, k=1):
	return 1/(1 + np.exp(-x*k))

#print(sigmoid(-1))

inp = np.array([[1, 0]])
out = np.array([[1]])
nn = mlp(inp, out, 2)





#self.weights_ih = np.array([[0.2, 0.5], [0.8,0.4]])
#self.weights_ho = np.array([[0.2], [0.5]])


