'''
Created on Feb 12, 2011

@author: anol
'''
from numpy import matlib as np

def perceptron(x, y, weight):
    '''Perceptron Algorithm'''
    N, l = x.shape

    rho = 0.02
    iter_max = 10000
    mis_class = N
    iter = 0
    w = weight
    while (mis_class > 0) and (iter < iter_max):
        mis_class = 0
        iter += 1
        gradient = np.zeros((1, l))
        for i in range(N):
            if((np.dot(x[i] , w.T)) * y[i] < 0):
                mis_class += 1
                gradient += (-y[i] * x[i])
        w -= rho * gradient[0]
    return w
