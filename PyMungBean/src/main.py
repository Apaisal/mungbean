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
#from PyML import * #@UnusedWildImport
#from PyML.demo import demo, demo2d
from PyML.classifiers import svm, multi
from PyML.evaluators import roc as roc1
from PyML.evaluators import roc as roc2
#from matplotlib.collections import PolyCollection #, LineCollection
#from matplotlib.colors import colorConverter
#from mpl_toolkits.mplot3d.axes3d import Axes3D
#import classifier
#import fnmatch
#import os
#import sys
#import re

Idn = ''
selected_file = 'selected%s.data' % (Idn)
test_file = 'test%s.data' % (Idn)
svm_file = 'svm%s.data' % (Idn)

#ext_fig1 = plt.figure(1)
#ext_fig2 = plt.figure(2)
#ext_fig3 = plt.figure(3)
#ext_fig4 = plt.figure(4)
#ext_fig5 = plt.figure(5)
#ext_fig6 = plt.figure(6)
#ext_fig7 = plt.figure(7)
#ext_fig8 = plt.figure(8)
#ext_fig9 = plt.figure(9)
#sel_fig = plt.figure(10)
#seled_fig = plt.figure(11)
#final_fig = plt.figure(6)

#fea1 = ext_fig1.add_subplot(111)
#fea1.grid()
#fea2 = ext_fig2.add_subplot(111)
#fea2.grid()
#
#fea3 = ext_fig3.add_subplot(111)
#fea3.grid()
#fea4 = ext_fig4.add_subplot(111)
#fea4.grid()
#fea5 = ext_fig5.add_subplot(111)
#fea5.grid()
#fea6 = ext_fig6.add_subplot(111)
#fea6.grid()
#fea7 = ext_fig7.add_subplot(111)
#fea7.grid()
#fea8 = ext_fig8.add_subplot(111)
#fea8.grid()
#fea9 = ext_fig9.add_subplot(111)
#fea9.grid()
#fea3 = Axes3D(ext_fig3)
#fea3.grid()
#fea4 = Axes3D(ext_fig4)
#fea4.grid()
#fea5 = Axes3D(ext_fig5)
#fea5.grid()
#fea6 = Axes3D(ext_fig6)
#fea6.grid()


#fx = sel_fig.add_subplot(111)
#fx.grid()
#sx = seled_fig.add_subplot(111)

def extraction(files):
    feature1 = feature.extraction.first_order_stat(files)
    feature2 = feature.extraction.moment_base(files)
#    feature3 = feature.extraction.fourier(files)
    feature1.update(feature2)
#    features = []
#    features.append(feature1.items() + feature2.items())#+ feature3.items())
    return feature1

if __name__ == '__main__':

#===========================================================================
# Image preparation 
#===========================================================================
    kfiles = glob.glob('../dataset/training_set%s/kamphangsean2/*.jpg' % (Idn))
    cfiles = glob.glob('../dataset/training_set%s/chainat72/*.jpg' % (Idn))
    afiles = glob.glob('../dataset/training_set%s/authong1/*.jpg' % (Idn))
    mfiles = glob.glob('../dataset/training_set%s/motoso1/*.jpg' % (Idn))

    filesk = glob.glob('../dataset/test_set%s/kamphangsean2/*.jpg' % (Idn))
    filesc = glob.glob('../dataset/test_set%s/chainat72/c*.jpg' % (Idn))
    filesa = glob.glob('../dataset/test_set%s/authong1/*.jpg' % (Idn))
    filesm = glob.glob('../dataset/test_set%s/motoso1/*.jpg' % (Idn))

#===============================================================================
# Feature Extraction class k
#===============================================================================
    #Training set
    c1 = extraction(cfiles)
    c2 = extraction(kfiles)
    c3 = extraction(afiles)
    c4 = extraction(mfiles)

    #Test set
    ct1 = extraction(filesc)
    ct2 = extraction(filesk)
    ct3 = extraction(filesa)
    ct4 = extraction(filesm)
#    kfiles = None
#    cfiles = None

# Plot mean and variance
#    fea1.set_xlabel('Var')
#    fea1.set_ylabel('Mean')
##    fea1.set_xlabel('STD')
##    fea1.set_ylabel('RMS')
#    fea1.set_title('Color')
#    fea1.plot(c1["var"], c1["mean"], 'rx')
#    fea1.plot(c2["var"], c2["mean"], 'bo')
#    fea1.plot(c3["var"], c3["mean"], 'g^')
#    fea1.plot(c4["var"], c4["mean"], 'y.')
#    fea1.legend(('CN72', 'KPS2', 'AUT1', 'MTS1'))
## Plot Area
#    fea2.set_title('Size')
#    fea2.plot(c1["area"], 'rx')
#    fea2.plot(c2["area"], 'bo')
#    fea2.plot(c3["area"], 'g^')
#    fea2.plot(c4["area"], 'y.')
#    fea2.legend(('CN72', 'KPS2', 'AUT1', 'MTS1'))
#
## Plot Hu
#    fea3.set_ylabel('Coefficient')
#    fea3.set_title('First Absolute Orthogonal Moment Invariant')
#    fea3.plot(c1["hu1"], 'rx')
#    fea3.plot(c2["hu1"], 'bo')
#    fea3.plot(c3["hu1"], 'g^')
#    fea3.plot(c4["hu1"], 'y.')
#    fea3.legend(('CN72', 'KPS2', 'AUT1', 'MTS1'))
#
#    fea4.set_ylabel('Coefficient')
#    fea4.set_title('Second Absolute Orthogonal Moment Invariant')
#    fea4.plot(c1["hu2"], 'rx')
#    fea4.plot(c2["hu2"], 'bo')
#    fea4.plot(c3["hu2"], 'g^')
#    fea4.plot(c4["hu2"], 'y.')
#    fea4.legend(('CN72', 'KPS2', 'AUT1', 'MTS1'))
#
#    fea5.set_ylabel('Coefficient')
#    fea5.set_title('Third Absolute Orthogonal Moment Invariant')
#    fea5.plot(c1["hu3"], 'rx')
#    fea5.plot(c2["hu3"], 'bo')
#    fea5.plot(c3["hu3"], 'g^')
#    fea5.plot(c4["hu3"], 'y.')
#    fea5.legend(('CN72', 'KPS2', 'AUT1', 'MTS1'))
#
#    fea6.set_ylabel('Coefficient')
#    fea6.set_title('Fourth Absolute Orthogonal Moment Invariant')
#    fea6.plot(c1["hu4"], 'rx')
#    fea6.plot(c2["hu4"], 'bo')
#    fea6.plot(c3["hu4"], 'g^')
#    fea6.plot(c4["hu4"], 'y.')
#    fea6.legend(('CN72', 'KPS2', 'AUT1', 'MTS1'))
#
#    fea7.set_ylabel('Coefficient')
#    fea7.set_title('Fifth Absolute Orthogonal Moment Invariant')
#    fea7.plot(c1["hu5"], 'rx')
#    fea7.plot(c2["hu5"], 'bo')
#    fea7.plot(c3["hu5"], 'g^')
#    fea7.plot(c4["hu5"], 'y.')
#    fea7.legend(('CN72', 'KPS2', 'AUT1', 'MTS1'))
#
#    fea8.set_ylabel('Coefficient')
#    fea8.set_title('Sixth Absolute Orthogonal Moment Invariant')
#    fea8.plot(c1["hu6"], 'rx')
#    fea8.plot(c2["hu6"], 'bo')
#    fea8.plot(c3["hu6"], 'g^')
#    fea8.plot(c4["hu6"], 'y.')
#    fea8.legend(('CN72', 'KPS2', 'AUT1', 'MTS1'))
#
#    fea9.set_ylabel('Coefficient')
#    fea9.set_title('Seventh Absolute Orthogonal Moment Invariant')
#    fea9.plot(c1["hu7"], 'rx')
#    fea9.plot(c2["hu7"], 'bo')
#    fea9.plot(c3["hu7"], 'g^')
#    fea9.plot(c4["hu7"], 'y.')
#    fea9.legend(('CN72', 'KPS2', 'AUT1', 'MTS1'))

#    plt.show()
#    xs = np.arange(1, 51)
#    zs = np.arange(1, 8)
#    verts1 = []
#    verts2 = []
#    verts3 = []
#    verts4 = []
#
#    cc = lambda arg: colorConverter.to_rgba(arg, alpha = 0.2)
#
#    fea3.set_title('Absolute Orthogonal Moment Invariants of CN72')
#    verts1.append(zip(xs, x1[6]))
#    verts1.append(zip(xs, x1[7]))
#    verts1.append(zip(xs, x1[9]))
#    verts1.append(zip(xs, x1[2]))
#    verts1.append(zip(xs, x1[3]))
#    verts1.append(zip(xs, x1[4]))
#    verts1.append(zip(xs, x1[5]))
#
#    poly1 = PolyCollection(verts1, facecolors = [cc('r'), cc('g'), cc('b'), \
#                                        cc('y'), cc('c'), cc('m'), cc('k') ])
#    poly1.set_alpha(0.5)
#    fea3.add_collection3d(poly1, zs = zs, zdir = 'y')
#    fea3.autoscale_view(True, True, True)
#    fea3.set_xlabel('Seed Number')
#    fea3.set_xlim3d(-10, 60)
#    fea3.set_ylabel('Order')
#    fea3.set_ylim3d(0, 8)
#    fea3.set_zlabel('Coefficient')
#    fea3.set_zlim3d(-1, 1)
#
#    fea4.set_title('Absolute Orthogonal Moment Invariants of KPS2')
#    verts2.append(zip(xs, x2[6]))
#    verts2.append(zip(xs, x2[7]))
#    verts2.append(zip(xs, x2[9]))
#    verts2.append(zip(xs, x2[2]))
#    verts2.append(zip(xs, x2[3]))
#    verts2.append(zip(xs, x2[4]))
#    verts2.append(zip(xs, x2[5]))
#    poly2 = PolyCollection(verts2, facecolors = [cc('r'), cc('g'), cc('b'), \
#                                        cc('y'), cc('c'), cc('m'), cc('k') ])
#    poly2.set_alpha(0.5)
#    fea4.add_collection3d(poly2, zs = zs, zdir = 'y')
#    fea4.autoscale_view(True, True, True)
#    fea4.set_xlabel('Seed Number')
#    fea4.set_xlim3d(-10, 60)
#    fea4.set_ylabel('Order')
#    fea4.set_ylim3d(0, 8)
#    fea4.set_zlabel('Coefficient')
#    fea4.set_zlim3d(-1, 1)
#
#    fea5.set_title('Absolute Orthogonal Moment Invariants of AUT1')
#    verts3.append(zip(xs, x3[6]))
#    verts3.append(zip(xs, x3[7]))
#    verts3.append(zip(xs, x3[9]))
#    verts3.append(zip(xs, x3[2]))
#    verts3.append(zip(xs, x3[3]))
#    verts3.append(zip(xs, x3[4]))
#    verts3.append(zip(xs, x3[5]))
#    poly3 = PolyCollection(verts3, facecolors = [cc('r'), cc('g'), cc('b'), \
#                                        cc('y'), cc('c'), cc('m'), cc('k') ])
#    poly3.set_alpha(0.5)
#    fea5.add_collection3d(poly3, zs = zs, zdir = 'y')
#    fea5.autoscale_view(True, True, True)
#    fea5.set_xlabel('Seed Number')
#    fea5.set_xlim3d(-10, 60)
#    fea5.set_ylabel('Order')
#    fea5.set_ylim3d(0, 8)
#    fea5.set_zlabel('Coefficient')
#    fea5.set_zlim3d(-1, 1)
#
#    fea6.set_title('Absolute Orthogonal Moment Invariants of MST1')
#    verts4.append(zip(xs, x4[6]))
#    verts4.append(zip(xs, x4[7]))
#    verts4.append(zip(xs, x4[9]))
#    verts4.append(zip(xs, x4[2]))
#    verts4.append(zip(xs, x4[3]))
#    verts4.append(zip(xs, x4[4]))
#    verts4.append(zip(xs, x4[5]))
#    poly4 = PolyCollection(verts4, facecolors = [cc('r'), cc('g'), cc('b'), \
#                                        cc('y'), cc('c'), cc('m'), cc('k') ])
#    poly4.set_alpha(0.5)
#    fea6.add_collection3d(poly4, zs = zs, zdir = 'y')
#    fea6.autoscale_view(True, True, True)
#    fea6.set_xlabel('Seed Number')
#    fea6.set_xlim3d(-10, 60)
#    fea6.set_ylabel('Order')
#    fea6.set_ylim3d(0, 8)
#    fea6.set_zlabel('Coefficient')
#    fea6.set_zlim3d(-1, 1)

#    x1 = []
#    x2 = []
#    x3 = []
#    x4 = []
#    xlabel = []
#    for f in range(len(c1[0])):
#        xlabel.append(c1[0][f][0])
#        x1.append(c1[0][f][1])
#    for f in range(len(c2[0])):
#        x2.append(c2[0][f][1])
#    for f in range(len(c3[0])):
#        x3.append(c3[0][f][1])
#    for f in range(len(c4[0])):
#        x4.append(c4[0][f][1])

    X1 = np.concatenate((c1.values(), c2.values(), c3.values(), c4.values()), axis = 1)

    y1 = ones((1, 50))
    y2 = ones((1, 50)) * 2
    y3 = ones((1, 50)) * 3
    y4 = ones((1, 50)) * 4

    Y1 = np.concatenate((y1, y2, y3, y4), axis = 1).A

#===============================================================================
# Feature Selection
#===============================================================================
    FDR = []
    for i in range(len(X1)):
        FDR.append(feature.selection.FDR_comp(X1, Y1, i))
#    fx.bar(range(len(X1)), FDR, 0.05, color = 'r')
#    fx.set_ylabel('Ratio')
#    fx.set_xlabel('Feature')
#    fx.set_title('Fisher\'s Discriminate Ratio')
#    fx.set_xticks(range(len(X1)))
#    fx.set_xticklabels((c1.keys()))
#    plt.show()
    # Choice ratio more than 50
    ind , selected = feature.selection.choice_strongfeature(X1, Y1, FDR, 50)

    with open(selected_file, "w") as fd:
        write = csv.writer(fd, delimiter = ' ')
#        write = csv.writer(fd, delimiter=',')

        for c in range(len(Y1.T)):
            line = []
#            line.append('%s' % (int(y.T[c][0])))
            line.append('%s,%s' % (c, int(Y1.T[c][0])))

            fea = selected.T[c]
            for i in range(len(ind)):
#                line.append("%s" % (fea[i]))
                line.append("%s:%s" % (ind[i], fea[i]))
            write.writerow(line)
#    FDR = None


#===============================================================================
# Classifies
#===============================================================================

    X2 = np.concatenate((ct1.values(), ct2.values(), ct3.values(), ct4.values()), axis = 1)

    g1 = ones((1, 50))
    g2 = ones((1, 50)) * 2
    g3 = ones((1, 50)) * 3
    g4 = ones((1, 50)) * 4

    Y2 = np.concatenate((g1, g2, g3, g4), axis = 1).A

    testdata = np.array(X2).take(ind, axis = 0)
    with open(test_file, "w") as fd:
        write = csv.writer(fd, delimiter = ' ')
#        write = csv.writer(fd, delimiter=',')

        for c in range(len(Y2.T)):
            line = []
#            line.append('%s' % (int(y.T[c][0])))
            line.append('%s,%s' % (c, int(Y2.T[c][0])))

            fea = testdata.T[c]
            for i in range(len(ind)):
#                line.append("%s" % (fea[i]))
                line.append("%s:%s" % (ind[i], fea[i]))
            write.writerow(line)


#===============================================================================
# Machine Learning
#===============================================================================
    trainingset1 = ml.SparseDataSet(selected_file)
    trainingset2 = ml.SparseDataSet(selected_file)
#    trainingset1 = ml.VectorDataSet(selected_file, labelsColumn=0)
#    trainingset2 = ml.VectorDataSet(selected_file, labelsColumn=0)
    testset1 = ml.SparseDataSet(test_file)
    testset2 = ml.SparseDataSet(test_file)
#    testset1 = ml.VectorDataSet(test_file, labelsColumn=0)
#    testset2 = ml.VectorDataSet(test_file, labelsColumn=0)
#    classifier.decisionSurface(sl, trainingset1, testset1)

    k2 = ml.ker.Polynomial(3)
    k1 = ml.ker.Linear()

    snl = multi.OneAgainstRest(svm.SVM(\
                                      k2 , \
                                      c = 10, \
#                                      optimizer = 'mysmo' \
                                      ))
#    snl = ml.SVM(k2)
#    snl.C = 10
    sl = multi.OneAgainstRest(svm.SVM(\
                                      k1 , \
                                      c = 10, \
#                                      optimizer = 'mysmo' \
                                      ))
#    sl = ml.SVM(k1)
#    sl.C = 10
#===============================================================================
# Linear Classifier
#===============================================================================
#    classifier.decisionSurface(sl, trainingset1, testset1)
    itert = 100
    rocN = 100
    normalize = True
    sl.train(trainingset1)
#    result1 = []
#    sl.save("linear_svm")
#    for i in range(100):
#        result1.append(sl.cv(testset1))
#        result1[i].plotROC("./a.pdf")

    result1 = sl.nCV(testset1, \
                     seed = 1, \
                      cvType = "stratifiedCV", \
#                      cvType = "cv", \
#                       intermediateFile = './result_linear' \
                     iterations = itert, \
                      numFolds = 5)

#    demo2d.setData(trainingset1)
#    demo2d.getData()
#    demo2d.decisionSurface(sl)
#    result1[19].plotROC('roc_linear%s.pdf' % (Idn))

    with open("result_linear_iter", "w") as fd:
        for res in result1:
            fd.write(str(res) + "\n")
        fd.write(str(result1) + "\n")

    result1.save("./linear_result")
#    for i in range(len(result1)):
#        labels1 = result1[i].getGivenClass()
#        dvals1 = result1[i].getDecisionFunction()
#        folds1 = [(dvals1[j], labels1[j]) for j in range(len(labels1))]
#        for k in range(4):
#            rocFP1, rocTP1, area1 = roc1.roc_VA(folds1, rocN, n_samps = 100, selectClass = k)
#            roc1.plotROC(rocFP1, rocTP1, 'roc_linear%d_%d.pdf' % (i, k), numPoints = 100)
#
#===============================================================================
# Non Linear Classifier
#===============================================================================

    snl.train(trainingset2)
    result2 = snl.nCV(testset2, \
                      seed = 1, \
#                      cvType = "cv", \
                      cvType = "stratifiedCV", \
#                      intermediateFile = './result_nonlinear', \
                      iterations = itert, \
                      numFolds = 5)

#    snl.preproject(testset2)
#    result2[19].plotROC('roc_nonlinear%s.pdf' % (Idn))
#    result2.save("./result_nonlinear", "short")
    with open("result_nonlinear_iter", "w") as fd:
        for res in result2:
            fd.write(str(res) + "\n")
        fd.write(str(result2) + "\n")

    result2.save("./nonlinear_result")
#    labels2 = result2[-1].getGivenClass()
#    dvals2 = result2[-1].getDecisionFunction()
#    folds2 = [(dvals2[i], labels2[i]) for i in range(len(labels2))]
#    for k in range(4):
#        rocFP2, rocTP2, area2 = roc2.roc_VA(folds2, rocN, n_samps = 100, selectClass = k)
#        roc2.plotROC(rocFP2, rocTP2, 'roc_nonlinear%d_%d.pdf' % (0, k), numPoints = 100)

#    roc.test()

#    plt.show()

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
