'''
Created on Feb 8, 2011

@author: anol
'''
import feature
import glob
import csv
import numpy as np
from numpy.matlib import ones
import matplotlib.pyplot as plt
import PyML as ml
from PyML import * #@UnusedWildImport
#from mpl_toolkits.mplot3d.axes3d import Axes3D
import classifier


id = '4'
selected_file = 'selected%s.csv' % (id)
test_file = 'test%s.data' % (id)
svm_file = 'svm%s.data' % (id)

ext_fig1 = plt.figure(1)
ext_fig2 = plt.figure(2)
ext_fig3 = plt.figure(3)
sel_fig = plt.figure(4)
seled_fig = plt.figure(5)
final_fig = plt.figure(6)

fea1 = ext_fig1.add_subplot(111)
fea1.grid()
fea2 = ext_fig2.add_subplot(111)
fea2.grid()
fea3 = ext_fig3.add_subplot(111)
fea3.grid()
fx = sel_fig.add_subplot(111)
fx.grid()
sx = seled_fig.add_subplot(111)

def extraction(files):
    feature1 = feature.extraction.first_order_stat(files)
    feature2 = feature.extraction.moment_base(files)
#    feature3 = feature.extraction.fourier(files)
    features = []
    features.append(feature1.items() + feature2.items())#+ feature3.items())
    return features

if __name__ == '__main__':
    kfiles = glob.glob('../dataset/training_set%s/k*.jpg' % (id))
    cfiles = glob.glob('../dataset/training_set%s/c*.jpg' % (id))

#===============================================================================
# Feature Extraction class k
#===============================================================================
    c1 = extraction(kfiles)
    c2 = extraction(cfiles)

    kfiles = None
    cfiles = None

    x1 = []
    x2 = []
    xlabel = []
    for f in range(len(c1[0])):
        xlabel.append(c1[0][f][0])
        x1.append(c1[0][f][1])
    for f in range(len(c2[0])):
        x2.append(c2[0][f][1])

#    c1 = None
#    c2 = None

    y1 = ones((1, len(x1[0])))
    y2 = ones((1, len(x2[0]))) * 2

    X = np.concatenate((x1, x2), axis = 1)
    y = np.concatenate((y1, y2), axis = 1).A

# Plot mean and variance
    fea1.set_xlabel('Var')
    fea1.set_ylabel('Mean')
#    fea1.set_xlabel('STD')
#    fea1.set_ylabel('RMS')
    fea1.set_title('Color Statistic')
    fea1.plot(x1[0], x1[1], 'rx')
    fea1.plot(x2[0], x2[1], 'bo')
    fea1.legend(('KPS1', 'CN72'))
# Plot Area
    fea2.set_title('Contour Area')
    fea2.plot(x1[8], 'b-')
    fea2.plot(x2[8], 'g.-')
    fea2.legend(('KPS1', 'CN72'))

# Plot Hu
    fea3.set_title('Seven Moments of HU')
    fea3.plot(x1[2], 'b-')
    fea3.plot(x1[3], 'r-')
    fea3.plot(x1[4], 'g-')
    fea3.plot(x1[5], 'y-')
    fea3.plot(x1[6], 'c-')
    fea3.plot(x1[7], 'm-')
    fea3.plot(x1[9], 'k-')

    fea3.plot(x2[2], 'b--')
    fea3.plot(x2[3], 'r--')
    fea3.plot(x2[4], 'g--')
    fea3.plot(x2[5], 'y--')
    fea3.plot(x2[6], 'c--')
    fea3.plot(x2[7], 'm--')
    fea3.plot(x2[9], 'k--')

    fea3.legend(('I4', 'I5', 'I6', 'I7', 'I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7', 'I1', 'I2', 'I3'))
#===============================================================================
# Feature Selection
#===============================================================================
    FDR = []
    for i in range(len(X)):
        FDR.append(feature.selection.FDR_comp(X, y, i))
    fx.bar(range(len(X)), FDR, 0.05, color = 'r')
    fx.set_ylabel('Ratio')
    fx.set_xlabel('Feature')
    fx.set_title('Fisher\'s Discriminate Ratio')
    fx.set_xticks(range(len(X)))
    fx.set_xticklabels((xlabel))
    plt.draw()
    # Choice ratio more than 50
    ind , selected = feature.selection.choice_strongfeature(X, y, FDR, 50)

    with open(selected_file, "w") as fd:
        write = csv.writer(fd, delimiter = ' ')

        for c in range(len(y.T)):
            line = []
            line.append('%s,%s' % (c, int(y.T[c][0])))
            fea = selected.T[c]
            for i in range(len(ind)):
                line.append("%s:%s" % (ind[i], fea[i]))
            write.writerow(line)
#    FDR = None
#===============================================================================
# Machine Learning
#===============================================================================
    trainingset1 = ml.SparseDataSet(selected_file)
    trainingset2 = ml.SparseDataSet(selected_file)

    k2 = ker.Polynomial(2) #ker.Gaussian(gamma = 0.5)
    k1 = ker.Linear()
    snl = ml.SVM(k2)
    snl.C = 100
    sl = ml.SVM(k1)
    sl.C = 100

#===============================================================================
# Classifies
#===============================================================================
    files1 = glob.glob('../dataset/test_set%s/k*.jpg' % (id))
    files2 = glob.glob('../dataset/test_set%s/c*.jpg' % (id))
    c1 = extraction(files1)
    c2 = extraction(files2)

    f1 = []
    f2 = []
    for f in range(len(c1[0])):
        f1.append(c1[0][f][1])
    for f in range(len(c2[0])):
        f2.append(c2[0][f][1])

    files1 = None
    files2 = None
#    c1 = None
#    c2 = None

    g1 = ones((1, len(f1[0])))
    g2 = ones((1, len(f2[0]))) * 2

    X = np.concatenate((f1, f2), axis = 1)
    y = np.concatenate((g1, g2), axis = 1).A

    testdata = np.array(X).take(ind, axis = 0)
    with open(test_file, "w") as fd:
        write = csv.writer(fd, delimiter = ' ')

        for c in range(len(y.T)):
            line = []
            line.append('%s,%s' % (c, int(y.T[c][0])))
            fea = testdata.T[c]
            for i in range(len(ind)):
                line.append("%s:%s" % (ind[i], fea[i]))
            write.writerow(line)
#    testdata = None
    testset1 = ml.SparseDataSet(test_file)
    testset2 = ml.SparseDataSet(test_file)
#    classifier.decisionSurface(sl, trainingset1, testset1)
#===============================================================================
# Linear Classifier
#===============================================================================

    sl.train(trainingset1)
    result1 = sl.cv(testset1, 5)
#    r1.plotROC('roc_linear%s.eps' % (id))
    print result1

#===============================================================================
# Non Linear Classifier
#===============================================================================

    snl.train(trainingset2)
    result2 = snl.cv(testset2, 5)
#    r2.plotROC('roc_nonlinear%s.eps' % (id))
    print result2

    classifier.scatter(trainingset1)
    plt.show()

    sl = None
    snl = None
    trainingset1 = None
    trainingset2 = None
    testset1 = None
    testset2 = None
    r1 = None
    r2 = None
    k1 = None
    k2 = None
