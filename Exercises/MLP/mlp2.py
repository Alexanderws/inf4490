import numpy as np

class mlp:
    """Multi-Layer Perceptron"""
    def __init__(self, inputs, targets, n_hidden, beta=1, momentum=0.9, outtype='linear'):
        self.n_outputs = np.shape(targets)[1]
        print("n_outputs (1)- {}".format(self.n_outputs))

        self.n_inputs = np.shape(inputs)[1]
        print("n_inputs (1)- {}".format(self.n_inputs))

        self.n_data = np.shape(inputs)[0]
        self.n_hidden = n_hidden

        self.beta = beta
        self.eta = 0.1
        self.outtype = outtype
        self.weights_ih = (np.random.rand(self.n_inputs+1,self.n_hidden)-0.5)*2/np.sqrt(self.n_inputs)
        self.weights_ho = (np.random.rand(self.n_hidden+1,self.n_outputs)-0.5)*2/np.sqrt(self.n_hidden)

    def train(self, inputs, targets, eta, niterations):
        inputs = np.concatenate((inputs,-np.ones((self.ndata,1))),axis=1)
        change = range(self.ndata)

        updatew1 = np.zeros((np.shape(self.weights1)))
        updatew2 = np.zeros((np.shape(self.weights2)))

        for n in range(niterations):



    def forward(self, inputs):
        """Run the neural network forward"""
        print("weights_ih (4,2)- {}".format(self.weights_ih))
        print("weights_ho (2,1)- {}".format(self.weights_ho))

        print("inputs ()- {}".format(inputs))

        self.hidden_sum = np.dot(inputs, self.weights_ih)
        self.hidden_sum = sigmoid(sum_1, self.beta)
        self.hidden_sum = np.concatenate((self.hidden, -np.ones((np.shape(inputs)[0],1))),axis=1)

        output_sum = np.dot(self.hidden_sum, self.weights_ho)

        if self.outtype == 'linear':
            return output_sum
        elif self.outtype == 'logistic':
            return sigmoid(output_sum, self.beta)
        elif self.outtype == 'softmax':
            normalisers = np.sum(np.exp(output_sum),axis=1)*np.ones((1,np.shape(output_sum)[0]))
            return np.transpose(np.transpose(np.exp(output_sum))/normalisers)


def sigmoid(x, k=1):
    return 1/(1 + np.exp(-k*x))