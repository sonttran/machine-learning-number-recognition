import struct
import math
import numpy as np
from array import array
import matplotlib.pyplot as plt

#import numpy as np
import numpy.linalg as LA

X = np.zeros(12).reshape(4,3)
#X = np.zeros(9).reshape(3,3)
print('* * initialize X rows are samples, cols are features:')
print(X)
print('* * shape of X according to python numpy: 4 rows x 3 cols')
print(X.shape)
X[0] = [1,5,6]
X[1] = [2,9,6]
X[2] = [3,8,10]
X[3] = [7,8,9]
print('* * gives X some values:')
print(X)


muX=np.mean(X, axis=0)
print('* * muX mean of X of axis 0:')
print(muX)


Z = X - muX
print('* * Z move center of gravity of X to coordinate origin:')
print(Z)


C = np.cov(X,rowvar=False)
print('* * C Covariance matrix of X:')
print(C)


print(Z[:,0])
def calVariance(X1):
    X1 = X1 - np.mean(X1)
    variance = np.sum(np.square(X1))/(len(X1)-1)
    print('* * variance of X1:', variance)

calVariance(X[:,0])
#print(math.sqrt((1-muX[0])**2+(2-muX[0])**2+(3-muX[0])**2+(7-3.25)**2))


#print('C:', C[0][0])
#print('C:', C.shape) # show (dimension, # of sample)

