'''
Created on Feb 8, 2011

@author: anol
'''
from numpy import matlib
from scipy import stats

def FDR_comp(X, y, ind):
    '''
    '''
    [l, N] = matlib.size(X)
    c = max(y)
    vari, m = []
    
    for i in range(1, c):
        y_temp = (y == i)
        X_temp = X[ind, y_temp]
        m[i] = matlib.mean(X_temp)
        vari[i] = matlib.var(X_temp)
        
    a = stats.binom_test(range(1,c), 2)
    q = (m[a[:, 1]] - m[a[:, 2]]) ^ 2 / (vari(a[:, 1]) + vari[a[:, 2]])
    return sum(q);
    

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
        
