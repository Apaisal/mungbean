'''
Created on Feb 8, 2011

@author: anol
'''
import numpy as np
from numpy import matlib
from scipy import stats

def FDR_comp(X, y, ind):
    '''
    '''
    (l, N) = X.shape
    c = int(np.max(y))
    vari = []
    m = []

    for i in range(1, c + 1):
        y_temp = (y == i).nonzero()[1]
        X_temp = X[ind, y_temp]
        m.append(matlib.mean(X_temp))
        vari.append(matlib.var(X_temp))

    a = stats.binom_test(range(c), 2)
    q = ((m[0] - m[1]) ** 2) / (vari[0] ** 2 + vari[1] ** 2)
    return np.sum(q);


def scatter_mat(X, y):
    '''
    '''
    [l, N] = X.size()
    c = max(y)
#    %Computation of class mean vectors, a priori prob. and
#    %Sw
    m, P = []
    Sw = matlib.zeros(l)
    for i in range(1, c) :
        y_temp = (y == i)
        X_temp = X[:, y_temp]
        P[i] = sum(y_temp) / N
        m[:, i] = (matlib.mean(X_temp))
        Sw = Sw + P[i] * matlib.cov(X_temp)

#    %Computation of Sb
    m0 = (sum(((matlib.ones((l, 1)) * P) * m)))
    Sb = matlib.zeros(l)
    for i in range(1, c) :
        Sb = Sb + P(i) * ((m[:, i] - m0) * (m[:, i] - m0))
#    %Computation of Sm
    Sm = Sw + Sb;
    return [Sm, Sw, Sb]

