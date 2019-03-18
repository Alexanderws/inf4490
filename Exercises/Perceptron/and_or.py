from random import choice
from numpy import array, dot, random
import matplotlib.pyplot as plt





unit_step = lambda x: 0 if x < 0 else 1

training_data = [(array([0,0,1]), 0),
(array([1,0,1]), 1),
(array([0,1,1]), 1),
(array([1,1,1]), 1)]

w = random.rand(3)
errors = []
eta = 0.2
n = 100

def train(training_data, eta, errors, n, w):
	for i in range(n):
		x, expected = choice(training_data)
		result = dot(w, x)
		error = expected - unit_step(result)
		errors.append(error)
		w += eta * error * x

train(training_data, eta, errors, n, w)
plt.plot(errors)
plt.show()