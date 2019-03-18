import csv
import time
import numpy as np
from random import shuffle
from itertools import permutations 
import matplotlib.pyplot as plt 
import seaborn
with open("european_cities.csv", "r") as f:
    data = list(csv.reader(f, delimiter=';'))

cities = {}
for i in range(0,len(data[0])):
	cities[i] = data[0][i]
distances = data[1:][:]

# GLOBAL VARIABLES
city_size = 10


class Route():

	def __init__(self, route):
		self.route = list(route)
		self.distance = self.calculate_distance()
		self.fitness = self.calculate_fitness()

	def calculate_distance(self):
		distance = 0
		for i in range(0,len(self.route)-1):
			distance += float(distances[int(self.route[i])][int(self.route[i+1])])
		distance += float(distances[int(self.route[len(self.route)-1])][int(self.route[0])])
		return distance

	def calculate_fitness(self):
		fitness = float(1/self.distance)
		return fitness

	def set_route(self, route):
		self.route = route
		self.init_values()
		return self.distance

	def randomize_route(self):
		shuffle(self.route)
		self.init_values()
		return self.route

	def mutate_route(self, mutation_rate):
		if np.random.randint(0,1) < mutation_rate:
			i1 = np.random.randint(0,len(self.route))
			i2 = np.random.randint(0,len(self.route))
			self.route[i1], self.route[i2] = self.route[i2], self.route[i1]
			self.init_values()
		return self.route

	def compare_route(self, route):
		if self.distance < route.distance:
			return self
		else:
			return route

	def init_values(self):
		self.distance = self.calculate_distance()
		self.fitness = self.calculate_fitness()
		return True


class Population():

	def __init__(self, size, city_size):
		self.pool = {}
		self.size = size
		for i in range(size):
			self.pool[i] = Route(range(0,city_size))

	def get_fittest(self):
		fittest = self.pool[0]
		for i in self.pool:
			fittest = fittest.compare_route(self.pool[i])
		return fittest

class Data():

	def __init__(self, tours):
		self.tours = tours
		self.best = self.best_distance()
		self.worst = self.worst_distance()
		self.avg = self.average_distance()
		self.std_dev = self.standard_deviation()
		self.comp_time = 0

	def best_distance(self):
		return min(self.tours)

	def worst_distance(self):
		return max(self.tours)

	def average_distance(self):
		return np.mean(self.tours)

	def standard_deviation(self):
		return np.std(self.tours)

	def __repr__(self):
		string = "Best distance: " + str(self.best) + '\n'
		string += "Worst distance: " + str(self.worst) + '\n'
		string += "Average distance: " + str(self.avg) + '\n'
		string += "Standard deviation: " + str(self.std_dev) + '\n'
		string += "Computation time: " + str(self.comp_time)
		return string

def print_cities(tour):
	string = ''
	for i in range(len(tour)-1):
		string += cities[tour[i]] + ' - '
	string += cities[tour[len(tour)-1]]
	return string

def generate_distances(tours):
	distances = []
	for i in range(0, len(tours)):
		distances.append(tours[i].distance)
	return distances

def timing(f):
    def wrap(*args):
        time1 = time.time()
        ret = f(*args)
        time2 = time.time()
        print(f.__name__, 'function took',round((time2-time1)*1000.0,3), 'ms')
        return ret
    return wrap

def calculate_distance(route):
	distance = 0
	for i in range(0,len(route)-1):
			distance += float(distances[int(route[i])][int(route[i+1])])
	distance += float(distances[int(route[len(route)-1])][int(route[0])])
	return distance

@timing
def exhaustive_search(max_length):
	total_distance = 9000000
	best_route = Route(range(0,max_length))
	for item in permutations(range(0,max_length)):
		current_route = Route(item)
		best_route = current_route.compare_route(best_route)
	return best_route

@timing
def hill_climbing(max_length):
	total_distance = 9000000
	best_route = Route(range(0,max_length)) #list(range(1,max_length+1))
	routes = []
	for x in range(0,20):
		initial_route = Route(range(0,max_length)) #list(range(1,max_length+1))
		initial_route.randomize_route()
		current_best = Route(hill_climb_permutations(initial_route.route))
		routes.append(current_best)
		best_route = best_route.compare_route(routes[x])
	return best_route

def hill_climb_permutations(route):
	best_route = route[:]
	for i in range(len(route)):
		for x in range(0, len(route)):
			n_r = best_route
			n_r[i], n_r[x] = n_r[x], n_r[i]
			if calculate_distance(n_r) < calculate_distance(best_route):
				best_route = n_r
	return best_route



# GENETIC ALGORITHM FUNCTIONS
def crossover(parent1, parent2):
	size = len(parent1.route)
	child = [0]*size

	start = np.random.randint(size - 2)
	end = np.random.randint(start, size + 1)

	child[start:end] = parent1.route[start:end]
	for i in range(0, size):
		while parent2.route[i] not in child:
			for ii in range(0, size):
				if child[ii] == 0:
					child[ii] = parent2.route[i]
					break
	return Route(child)

def parent_selection(pop, tour_size, city_size):
	selection = Population(tour_size, city_size)
	for i in range(0, tour_size):
		selection.pool[i] = pop.pool[np.random.randint(0, pop.size)]
	return selection.get_fittest()

def mutate_population(pop, mutation_rate):
	for i in pop.pool:
		pop.pool[i].mutate_route(mutation_rate)
	return True

@timing
def evolve_population(pop):
	mutation_rate = 0.025
	tournament_size = int(pop.size/10)

	new_pop = Population(pop.size, city_size)

	for i in range (0, pop.size):
		parent1 = parent_selection(pop, tournament_size, city_size)
		parent2 = parent_selection(pop, tournament_size, city_size)
		child = crossover(parent1, parent2)

		new_pop.pool[i] = child
	
	mutate_population(new_pop, mutation_rate)

	return new_pop

@timing
def hybrid_evolve_population(pop):
	mutation_rate = 0.025
	tournament_size = int(pop.size/10)

	new_pop = Population(pop.size, city_size)

	for i in range (0, pop.size):
		parent1 = parent_selection(pop, tournament_size, city_size)
		parent2 = parent_selection(pop, tournament_size, city_size)
		child = crossover(parent1, parent2)
		hc_child = Route(hill_climb_permutations(child.route))
		child.fitness = hc_child.fitness
		new_pop.pool[i] = child
		# Lamarckian/Baldwin
		# new_pop.pool[i] = Route(hc_child)
		
	
	mutate_population(new_pop, mutation_rate)

	return new_pop


# GENETIC ALGORITHM / HYBRID
# Population sizes: 10, 100, 1000
# City: 10
print('GA 10')
population = Population(10, city_size)
ga_10 = []
ga_10_br = []
ga_10_f = []
ga_10_bd = 900000
for i in range(20):
	population = hybrid_evolve_population(population)
	ga_10.append(population.get_fittest())
	ga_10_f.append(ga_10[i].fitness)
	if ga_10[i].distance < ga_10_bd:
		ga_10_bd = ga_10[i].distance
		ga_10_br = ga_10[i].route

ga_10_data = Data(generate_distances(ga_10))
print(ga_10_data)
print(ga_10_br)
print(print_cities(ga_10_br), '\n')

print('GA 100')
population = Population(100, city_size)
ga_100 = []
ga_100_br = []
ga_100_f = []
ga_100_bd = 900000
for i in range(20):
	population = hybrid_evolve_population(population)
	ga_100.append(population.get_fittest())
	ga_100_f.append(ga_100[i].fitness)
	if ga_100[i].distance < ga_100_bd:
		ga_100_bd = ga_100[i].distance
		ga_100_br = ga_100[i].route

ga_100_data = Data(generate_distances(ga_100))
print(ga_100_data)
print(ga_100_br)
print(print_cities(ga_100_br), '\n')

print('GA 1000')
population = Population(1000, city_size)
ga_1000 = []
ga_1000_br = []
ga_1000_f = []
ga_1000_bd = 900000

for i in range(20):
	population = hybrid_evolve_population(population)
	ga_1000.append(population.get_fittest())
	ga_1000_f.append(ga_1000[i].fitness)
	if ga_1000[i].distance < ga_1000_bd:
		ga_1000_bd = ga_1000[i].distance
		ga_1000_br = ga_1000[i].route

ga_1000_data = Data(generate_distances(ga_1000))
print(ga_1000_data)
print(ga_1000_br)
print(print_cities(ga_1000_br), '\n')

plt.plot(range(1,21), ga_10_f, label = 'Hybrid - Avg Fitness of pop: 10')
plt.plot(range(1,21), ga_100_f, label = 'Hybrid - Avg Fitness of pop: 100')
plt.plot(range(1,21), ga_1000_f, label = 'Hybrid - Avg Fitness of pop: 1000')
plt.legend()
plt.show()

# HILL CLIMBING IN PRACTICE
""" 10 CITIES
hc_10_cities = []
best_distance = 900000
best_route = []
for i in range(20):
	hc_10_cities.append(hill_climbing(10))
	if hc_10_cities[i].distance < best_distance:
		best_distance = hc_10_cities[i].distance
		best_route = hc_10_cities[i].route
hc_10_data = Data(generate_distances(hc_10_cities))

hc_10_data.comp_time = '34.812 ms'
print(best_route)
print(print_cities(best_route))
print(hc_10_data)
"""

""" 24 CITIES
hc_24_cities = []
best_distance = 900000
best_route = []
for i in range(20):
	hc_24_cities.append(hill_climbing(24))
	if hc_24_cities[i].distance < best_distance:
		best_distance = hc_24_cities[i].distance
		best_route = hc_24_cities[i].route
hc_24_data = Data(generate_distances(hc_24_cities))

hc_24_data.comp_time = '34.812 ms'
print(best_route)
print(print_cities(best_route))
print(hc_24_data)

# route, distances = hill_climbing(10)
# print(calculate_distance(route), route)
"""

# EXHAUSTIVE SEARCH
"""
ex_6 = exhaustive_search(6) 
print(ex_6.route, ex_6.distance)
print(print_cities(ex_6.route))
"""