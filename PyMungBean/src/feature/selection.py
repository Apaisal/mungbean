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

#    a = stats.binom_test(range(c), c)
#    q = ((m[0] - m[1]) ** 2) / (vari[0] ** 2 + vari[1] ** 2)
    FDR = 0
    for i in range(c):
        for j in range(c):
            if i != j:
                FDR += pow(m[i] - m[j], 2) / float(pow(vari[i], 2) + pow(vari[j], 2))
#    var = 0
#    for i in vari: var += pow(i, 2)
#    q = pow(np.sum(m), 2) / var
    return FDR;


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

def choice_strongfeature(X, y, FDR, threshold = 100):
    '''
    '''
#    selected = []
#    c = int(np.max(y))
#    ind = []
#    ind.extend(FDR)
#    ind.sort()
#    ind = [v > threshold for v in ind]
    ind = (np.array(FDR) >= threshold).nonzero()[0]
#    ind = (np.array(FDR) >= max(FDR)).nonzero()[0]
#    for i in range(1, c + 1):
#        y_temp = (y == i).nonzero()[1]
    f_seled = X.take(ind, axis = 0)
#        selected.append(f_seled.take(y_temp, axis = 1))
    return list(ind) , f_seled #selected
