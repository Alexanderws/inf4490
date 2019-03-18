import numpy as np
import matplotlib.pyplot as plt
from integrator import timing

@timing
def numpy_integrate(f, a, b, n):
	interval = np.linspace(a, b, n+1)
	sum_of_integral = np.sum(f(interval))*b-a/n
	return sum_of_integral

def f(x):
	return x**2



if __name__ == '__main__':
	sum_of_integral1 = numpy_integrate(f, 0, 1, 200)
	plot_error(100)
	print(sum_of_integral1)
