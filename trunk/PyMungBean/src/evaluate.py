'''
Created on Feb 6, 2012

@author: anol
'''
from PyML.utils import myio
from PyML.evaluators import roc as roc1
from PyML.evaluators import roc as roc2
from PyML import *

if __name__ == '__main__':
    rocN = 100
    color = ['b', 'g', 'r', 'c']
    linearResult = myio.load("./linear_result")
    with open("./result_linear_roc", "w") as fd:
        for i in range(len(linearResult)):
            folds1 = [(linearResult[i][j].decisionFunc, linearResult[i][j].Y) for j in range(len(linearResult[i]))]
            for k in range(4):
                rocFP1, rocTP1, area1 = roc1.roc_VA(folds1, rocN, n_samps = 100, selectClass = k)
                roc1.plotROC(rocFP1, rocTP1, 'roc_linear%d_%d.svg' % (i, k), numPoints = 100, show = False, plotStr = "-%s" % (color[k]))

#
#    nonlinearResult = myio.load("./nonlinear_result")

