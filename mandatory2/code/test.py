import numpy as np

arr1 = np.array([[0,0,0],[1,1,1],[2,2,2],[3,3,3],[4,4,4,]])

print(len(arr1))

arr2 = arr1[0::3,:]
print(arr2)