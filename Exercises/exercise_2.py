import numpy as np
# Exercise 2 - Representations

""" 
1) Permutation: PMX (Partial Map Crossover) - Copy a section over, fill remaining by comparing with section
				Edge Recombination - Combine based on neighbours
				Order Crossover - Copy a section over, fill remaining cells in order
				Cycle Crossover - Compare parents, find cycles of numbers
	Binary: 	1-point Crossover - Split at random points, create children from heads/tails
				n-point Crossover - Split at several random points, otherwise the same
				Uniform Crossover - Loop through parents, flip coin to inverse the gene

2) 	Probability of no bits being flipped: p(0) = (3/4)^4 = 0.316
	Probability of 1 bit being flipped: p(1) = (1/4 * 3/4 * 3/4 * 3/4) = 0.105
	Probabiltiy of more than 1 being flipped p(>1) = 1 - (p(1) + p(0)) = 0.579 

3) """

p1 = [2,4,7,1,3,6,8,9,5]
p2 = [5,9,8,6,2,4,1,3,7]

def pmx(p1, p2):
	c1 = c2 = [0]*len(p1)
	startPoint = np.random.randint(len(p1)-1)
	stopPoint = np.random.randint(startPoint, len(p1))+1
	print(startPoint, stopPoint)
	c1[startPoint:stopPoint] = p1[startPoint:stopPoint] # Copy segment to child
	print(c1)
	print(p1, p2)
	for i in p2[startPoint: stopPoint]:
		print("i in p2[start:stop]: " + str(i))
		if not i in p1[startPoint:stopPoint]: #If item is not in segment
			j = p1[p2.index(i)] # Item j
			while (j != 0):
				if p2[p1.index(j)] not in c1:
					c1[p1.index(j)] = j
					print(str(j) + " placed at index " + str(p1.index(j)))
					j = 0
				else:
					j = p2[c1.index[p1.index(j)]]

	print(c1)
	





pmx(p1, p2)