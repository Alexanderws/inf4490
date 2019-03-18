import numpy as np
import matplotlib.pyplot as plt

def f(x):
	return -x**4 + 2*x**3 + 2*x**2 - x

def df(x):
	return -4*x**3 + 6*x**2 + 4*x - 1

def gradient_ascent(x, gamma = 0.001):
	print('Starting x: ' + str(x))
	while abs(df(x)) > 0.001:
		plt.plot(x, f(x), color='black', marker='o')
		x = x + gamma*df(x)
	plt.plot(x, f(x), color='red', marker='o')

def exhaustive_search(x_range):
	best_x = 0
	for x in x_range:
		if df(x) < 0.0001 and f(x) > f(best_x):
			best_x = x
	plt.plot(best_x, f(best_x), color='red', marker='o')
	print(best_x)

def main():

	x = np.linspace(-2, 3, 1000)

	# gradient_ascent(x[np.random.randint(1000)])
	exhaustive_search(x)
	plt.plot(x, f(x))
	plt.show()



main()