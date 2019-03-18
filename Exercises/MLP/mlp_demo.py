import mlp2
import mlp
import numpy as np


inputs = np.array([[0,0],[0,1],[1,0],[1,1]])
targets = np.array([[0],[1],[1],[0]])#,[1],[0]])

mlp_network = mlp.mlp(inputs,targets,2)
print(mlp_network)
print(mlp_network.forward(inputs))
mlp_network.train(inputs, targets, 50)
#print(mlp_network.forward(inputs))
