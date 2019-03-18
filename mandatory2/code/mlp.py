'''
    This pre-code is a nice starting point, but you can
    change it to fit your needs.
'''
import numpy as np
import matplotlib.pyplot as plt

class mlp:
    def __init__(self, inputs, targets, n_hidden, beta=1, momentum=0.9, outtype='linear'):
        self.n_outputs = np.shape(targets)[1]
        print("n_outputs (1)- {}".format(self.n_outputs))

        self.n_inputs = np.shape(inputs)[1]
        print("n_inputs (1)- {}".format(self.n_inputs))

        self.n_data = np.shape(inputs)[0]
        print("n_data (1)- {}".format(self.n_data))
        self.percentage = 0
        self.errors = 0
        self.n_hidden = n_hidden
        self.beta = beta
        self.eta = 0.1
        self.momentum = momentum
        self.outtype = outtype
        self.weights_ih = (np.random.rand(self.n_inputs+1,self.n_hidden)-0.5)*2/np.sqrt(self.n_inputs)
        self.weights_ho = (np.random.rand(self.n_hidden+1,self.n_outputs)-0.5)*2/np.sqrt(self.n_hidden)

    # You should add your own methods as well!

    def earlystopping(self, inputs, targets, valid, validtargets, eta=0.1, iterations=100):
        valid = np.concatenate((valid, -np.ones((np.shape(valid)[0],1))),axis=1)

        old_val_error1 = 100002
        old_val_error2 = 100001
        new_val_error = 100000

        count = 0

        while (((old_val_error1 - new_val_error) > 0.001) or ((old_val_error2 - old_val_error1)>0.001)):
            count+=1
            #print(count)
            self.train(inputs, targets, eta, iterations)
            old_val_error2 = old_val_error1
            old_val_error1 = new_val_error
            validout = self.forward(valid)
            new_val_error = 0.5*np.sum((validtargets-validout)**2)

        print("Stopped", new_val_error,old_val_error1, old_val_error2)
        self.errors = new_val_error
        return new_val_error

    def train(self, inputs, targets, eta, iterations=1):
        inputs = np.concatenate((inputs,-np.ones((self.n_data,1))),axis=1)
        change = range(self.n_data)
        update_w_ih = np.zeros((np.shape(self.weights_ih)))
        update_w_ho = np.zeros((np.shape(self.weights_ho)))

        for n in range(iterations):
            self.outputs = self.forward(inputs)
            error = 0.5*np.sum((self.outputs-targets)**2)
            if (np.mod(n,100)==0):
                print("Iteration: ",n, " Error: ",error)
            # Different types of output neurons
            if self.outtype == 'linear':
                deltao = (self.outputs-targets)/self.n_data
            elif self.outtype == 'logistic':
                deltao = self.beta*(self.outputs-targets)*self.outputs*(1.0-self.outputs)
            elif self.outtype == 'softmax':
                deltao = (self.outputs-targets)*(self.outputs*(-self.outputs)+self.outputs)/self.n_data 
            else:
                print("error")

            deltah = self.hidden_sum*self.beta*(1.0-self.hidden_sum)*(np.dot(deltao,np.transpose(self.weights_ho)))
            update_w_ih = eta*(np.dot(np.transpose(inputs),deltah[:,:-1])) + self.momentum*update_w_ih
            update_w_ho = eta*(np.dot(np.transpose(self.hidden_sum),deltao)) + self.momentum*update_w_ho
            self.weights_ih -= update_w_ih
            self.weights_ho -= update_w_ho



    def forward(self, inputs):
        #inputs = np.concatenate((inputs,-np.ones((self.n_data,1))),axis=1)

        self.hidden_sum = np.dot(inputs, self.weights_ih)
        self.hidden_sum = sigmoid(self.hidden_sum, self.beta)
        self.hidden_sum = np.concatenate((self.hidden_sum, -np.ones((np.shape(inputs)[0],1))),axis=1)

        output_sum = np.dot(self.hidden_sum, self.weights_ho)

        if self.outtype == 'linear':
            return output_sum
        elif self.outtype == 'logistic':
            return sigmoid(output_sum, self.beta)
        elif self.outtype == 'softmax':
            normalisers = np.sum(np.exp(output_sum),axis=1)*np.ones((1,np.shape(output_sum)[0]))
            return np.transpose(np.transpose(np.exp(output_sum))/normalisers)


    def confusion(self, inputs, targets):
        inputs = np.concatenate((inputs,-np.ones((np.shape(inputs)[0],1))),axis=1)
        outputs = self.forward(inputs)
        n_classes = np.shape(targets)[1]
        if n_classes == 1:
            n_classes = 2
            outputs = np.where(outputs>0.5,1,0)
        else:
            outputs = np.argmax(outputs,1)
            targets = np.argmax(targets,1)

        confusion_matrix = np.zeros((n_classes,n_classes))
        for i in range(n_classes):
            for j in range(n_classes):
                confusion_matrix[i,j] = np.sum(np.where(outputs==i,1,0)*np.where(targets==j,1,0))
        percentage_correct = np.trace(confusion_matrix)/np.sum(confusion_matrix)*100
        #print("Confusion matrix:")

        #print(confusion_matrix)
        #print("Percentage correct: ", percentage_correct)
        #plt.hist2d(targets, outputs, bins=30)
        plt.show()
        self.percentage = percentage_correct
        return percentage_correct


def sigmoid(x, k=1):
    return 1/(1 + np.exp(-k*x))