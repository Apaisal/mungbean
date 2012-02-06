'''
Created on Feb 6, 2012

@author: anol
'''
from PyML.utils import myio
from PyML.evaluators import roc as roc1
from PyML.evaluators import roc as roc2
from PyML import *
#==============================================================================
# ==========  ========
#       character   color
#       ==========  ========
#       'b'         blue
#       'g'         green
#       'r'         red
#       'c'         cyan
#       'm'         magenta
#       'y'         yellow
#       'k'         black
#       'w'         white
#       ==========  ========
#==============================================================================
if __name__ == '__main__':
    rocN = 100
    color1 = ['b', 'g', 'r', 'c']
    color2 = ['m', 'y', 'k', '-b']
    linearResult = myio.load("./linear_result1")
#    with open("./result_linear_roc", "w") as fd:
    label1 = []
    dval1 = []
    for i in range(len(linearResult)):
        for j in range(len(linearResult[i])):
            label1.extend(linearResult[i][j].givenY)
            dval1.extend(linearResult[i][j].decisionFunc)

#        folds1 = [[i  for elem in dval for i in elem], [i  for elem in label for i in elem]]
    for k in range(4):
        rocFP1, rocTP1, area1 = roc1.roc(dval1, label1, rocN, selectClass = k)
#            rocFP1, rocTP1, area1 = roc1.roc_VA(folds1, rocN, n_samps = 100, selectClass = k)
        roc1.plotROC(rocFP1, rocTP1, 'roc_linear%d_%d.pdf' % (i, k), numPoints = 100, show = True, plotStr = "-%s" % (color1[k]))

#
    nonlinearResult = myio.load("./nonlinear_result1")
#    with open("./result_nonlinear_roc", "w") as fd:
    label2 = []
    dval2 = []
    for i in range(len(nonlinearResult)):
        for j in range(len(nonlinearResult[i])):
            label2.extend(nonlinearResult[i][j].givenY)
            dval2.extend(nonlinearResult[i][j].decisionFunc)

#        folds1 = [[i  for elem in dval for i in elem], [i  for elem in label for i in elem]]
    for k in range(4):
        rocFP2, rocTP2, area2 = roc2.roc(dval2, label2, rocN, selectClass = k)
#            rocFP1, rocTP1, area1 = roc1.roc_VA(folds1, rocN, n_samps = 100, selectClass = k)
        roc2.plotROC(rocFP2, rocTP2, 'roc_nonlinear%d_%d.pdf' % (i, k), numPoints = 100, show = True, plotStr = "-%s" % (color2[k]))

