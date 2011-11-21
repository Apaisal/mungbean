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

from matplotlib.collections import PolyCollection, LineCollection
from matplotlib.colors import colorConverter
from mpl_toolkits.mplot3d.axes3d import Axes3D
#import classifier
#import fnmatch
#import os
#import sys
#import re

id = ''
selected_file = 'selected%s.data' % (id)
test_file = 'test%s.data' % (id)
svm_file = 'svm%s.data' % (id)

ext_fig1 = plt.figure(1)
ext_fig2 = plt.figure(2)
ext_fig3 = plt.figure(3)
ext_fig4 = plt.figure(4)
ext_fig5 = plt.figure(5)
ext_fig6 = plt.figure(6)
sel_fig = plt.figure(7)
seled_fig = plt.figure(8)
#final_fig = plt.figure(6)

fea1 = ext_fig1.add_subplot(111)
fea1.grid()
fea2 = ext_fig2.add_subplot(111)
fea2.grid()

fea3 = Axes3D(ext_fig3)
fea3.grid()
fea4 = Axes3D(ext_fig4)
fea4.grid()
fea5 = Axes3D(ext_fig5)
fea5.grid()
fea6 = Axes3D(ext_fig6)
fea6.grid()

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

#===========================================================================
# Image preparation 
#===========================================================================
    kfiles = glob.glob('../dataset/training_set%s/kamphangsean2/*.jpg' % (id))
    cfiles = glob.glob('../dataset/training_set%s/chainat72/*.jpg' % (id))
    afiles = glob.glob('../dataset/training_set%s/authong1/*.jpg' % (id))
    mfiles = glob.glob('../dataset/training_set%s/motoso1/*.jpg' % (id))

    filesk = glob.glob('../dataset/test_set%s/kamphangsean2/*.jpg' % (id))
    filesc = glob.glob('../dataset/test_set%s/chainat72/c*.jpg' % (id))
    filesa = glob.glob('../dataset/test_set%s/authong1/*.jpg' % (id))
    filesm = glob.glob('../dataset/test_set%s/motoso1/*.jpg' % (id))

#===============================================================================
# Feature Extraction class k
#===============================================================================
    c1 = extraction(cfiles)
    c2 = extraction(kfiles)
    c3 = extraction(afiles)
    c4 = extraction(mfiles)

#    kfiles = None
#    cfiles = None

    x1 = []
    x2 = []
    x3 = []
    x4 = []
    xlabel = []
    for f in range(len(c1[0])):
        xlabel.append(c1[0][f][0])
        x1.append(c1[0][f][1])
    for f in range(len(c2[0])):
        x2.append(c2[0][f][1])
    for f in range(len(c3[0])):
        x3.append(c3[0][f][1])
    for f in range(len(c4[0])):
        x4.append(c4[0][f][1])

#    c1 = None
#    c2 = None

    y1 = ones((1, len(x1[0])))
    y2 = ones((1, len(x2[0]))) * 2
    y3 = ones((1, len(x2[0]))) * 3
    y4 = ones((1, len(x2[0]))) * 4

    X = np.concatenate((x1, x2, x3, x4), axis = 1)
    y = np.concatenate((y1, y2, y3, y4), axis = 1).A

# Plot mean and variance
    fea1.set_xlabel('Var')
    fea1.set_ylabel('Mean')
#    fea1.set_xlabel('STD')
#    fea1.set_ylabel('RMS')
    fea1.set_title('Color')
    fea1.plot(x1[0], x1[1], 'rx')
    fea1.plot(x2[0], x2[1], 'bo')
    fea1.plot(x3[0], x3[1], 'g^')
    fea1.plot(x4[0], x4[1], 'y.')
    fea1.legend(('CN72', 'KPS2', 'AUT1', 'MTS1'))
# Plot Area
    fea2.set_title('Size')
    fea2.plot(x1[8], 'rx')
    fea2.plot(x2[8], 'bo')
    fea2.plot(x3[8], 'g^')
    fea2.plot(x4[8], 'y.')
    fea2.legend(('CN72', 'KPS2', 'AUT1', 'MTS1'))

# Plot Hu
    xs = np.arange(1, 51)
    zs = np.arange(1, 8)
    verts1 = []
    verts2 = []
    verts3 = []
    verts4 = []

    cc = lambda arg: colorConverter.to_rgba(arg, alpha = 0.2)

    fea3.set_title('Absolute Orthogonal Moment Invariants of CN72')
    verts1.append(zip(xs, x1[6]))
    verts1.append(zip(xs, x1[7]))
    verts1.append(zip(xs, x1[9]))
    verts1.append(zip(xs, x1[2]))
    verts1.append(zip(xs, x1[3]))
    verts1.append(zip(xs, x1[4]))
    verts1.append(zip(xs, x1[5]))

    poly1 = PolyCollection(verts1, facecolors = [cc('r'), cc('g'), cc('b'), \
                                        cc('y'), cc('c'), cc('m'), cc('k') ])
    poly1.set_alpha(0.5)
    fea3.add_collection3d(poly1, zs = zs, zdir = 'y')
    fea3.autoscale_view(True, True, True)
    fea3.set_xlabel('Seed Number')
    fea3.set_xlim3d(-10, 60)
    fea3.set_ylabel('Order')
    fea3.set_ylim3d(0, 8)
    fea3.set_zlabel('Coefficient')
    fea3.set_zlim3d(-1, 1)

    fea4.set_title('Absolute Orthogonal Moment Invariants of KPS2')
    verts2.append(zip(xs, x2[6]))
    verts2.append(zip(xs, x2[7]))
    verts2.append(zip(xs, x2[9]))
    verts2.append(zip(xs, x2[2]))
    verts2.append(zip(xs, x2[3]))
    verts2.append(zip(xs, x2[4]))
    verts2.append(zip(xs, x2[5]))
    poly2 = PolyCollection(verts2, facecolors = [cc('r'), cc('g'), cc('b'), \
                                        cc('y'), cc('c'), cc('m'), cc('k') ])
    poly2.set_alpha(0.5)
    fea4.add_collection3d(poly2, zs = zs, zdir = 'y')
    fea4.autoscale_view(True, True, True)
    fea4.set_xlabel('Seed Number')
    fea4.set_xlim3d(-10, 60)
    fea4.set_ylabel('Order')
    fea4.set_ylim3d(0, 8)
    fea4.set_zlabel('Coefficient')
    fea4.set_zlim3d(-1, 1)

    fea5.set_title('Absolute Orthogonal Moment Invariants of AUT1')
    verts3.append(zip(xs, x3[6]))
    verts3.append(zip(xs, x3[7]))
    verts3.append(zip(xs, x3[9]))
    verts3.append(zip(xs, x3[2]))
    verts3.append(zip(xs, x3[3]))
    verts3.append(zip(xs, x3[4]))
    verts3.append(zip(xs, x3[5]))
    poly3 = PolyCollection(verts3, facecolors = [cc('r'), cc('g'), cc('b'), \
                                        cc('y'), cc('c'), cc('m'), cc('k') ])
    poly3.set_alpha(0.5)
    fea5.add_collection3d(poly3, zs = zs, zdir = 'y')
    fea5.autoscale_view(True, True, True)
    fea5.set_xlabel('Seed Number')
    fea5.set_xlim3d(-10, 60)
    fea5.set_ylabel('Order')
    fea5.set_ylim3d(0, 8)
    fea5.set_zlabel('Coefficient')
    fea5.set_zlim3d(-1, 1)

    fea6.set_title('Absolute Orthogonal Moment Invariants of MST1')
    verts4.append(zip(xs, x4[6]))
    verts4.append(zip(xs, x4[7]))
    verts4.append(zip(xs, x4[9]))
    verts4.append(zip(xs, x4[2]))
    verts4.append(zip(xs, x4[3]))
    verts4.append(zip(xs, x4[4]))
    verts4.append(zip(xs, x4[5]))
    poly4 = PolyCollection(verts4, facecolors = [cc('r'), cc('g'), cc('b'), \
                                        cc('y'), cc('c'), cc('m'), cc('k') ])
    poly4.set_alpha(0.5)
    fea6.add_collection3d(poly4, zs = zs, zdir = 'y')
    fea6.autoscale_view(True, True, True)
    fea6.set_xlabel('Seed Number')
    fea6.set_xlim3d(-10, 60)
    fea6.set_ylabel('Order')
    fea6.set_ylim3d(0, 8)
    fea6.set_zlabel('Coefficient')
    fea6.set_zlim3d(-1, 1)

#    plt.show()

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
#    plt.draw()
    # Choice ratio more than 50
    ind , selected = feature.selection.choice_strongfeature(X, y, FDR, 10)

    with open(selected_file, "w") as fd:
        write = csv.writer(fd, delimiter = ' ')
#        write = csv.writer(fd, delimiter=',')

        for c in range(len(y.T)):
            line = []
#            line.append('%s' % (int(y.T[c][0])))
            line.append('%s,%s' % (c, int(y.T[c][0])))

            fea = selected.T[c]
            for i in range(len(ind)):
#                line.append("%s" % (fea[i]))
                line.append("%s:%s" % (ind[i], fea[i]))
            write.writerow(line)
#    FDR = None
#===============================================================================
# Machine Learning
#===============================================================================
    trainingset1 = ml.SparseDataSet(selected_file)
    trainingset2 = ml.SparseDataSet(selected_file)
#    trainingset1 = ml.VectorDataSet(selected_file, labelsColumn=0)
#    trainingset2 = ml.VectorDataSet(selected_file, labelsColumn=0)

#===============================================================================
# Classifies
#===============================================================================


    c1 = extraction(filesc)
    c2 = extraction(filesk)
    c3 = extraction(filesa)
    c4 = extraction(filesm)

    f1 = []
    f2 = []
    f3 = []
    f4 = []
    for f in range(len(c1[0])):
        f1.append(c1[0][f][1])
    for f in range(len(c2[0])):
        f2.append(c2[0][f][1])
    for f in range(len(c3[0])):
        f3.append(c3[0][f][1])
    for f in range(len(c4[0])):
        f4.append(c4[0][f][1])

#    files1 = None
#    files2 = None
#    c1 = None
#    c2 = None

    g1 = ones((1, len(f1[0])))
    g2 = ones((1, len(f2[0]))) * 2
    g3 = ones((1, len(f2[0]))) * 3
    g4 = ones((1, len(f2[0]))) * 4

    X = np.concatenate((f1, f2, f3, f4), axis = 1)
    y = np.concatenate((g1, g2, g3, g4), axis = 1).A

    testdata = np.array(X).take(ind, axis = 0)
    with open(test_file, "w") as fd:
        write = csv.writer(fd, delimiter = ' ')
#        write = csv.writer(fd, delimiter=',')

        for c in range(len(y.T)):
            line = []
#            line.append('%s' % (int(y.T[c][0])))
            line.append('%s,%s' % (c, int(y.T[c][0])))

            fea = testdata.T[c]
            for i in range(len(ind)):
#                line.append("%s" % (fea[i]))
                line.append("%s:%s" % (ind[i], fea[i]))
            write.writerow(line)

    testset1 = ml.SparseDataSet(test_file)
    testset2 = ml.SparseDataSet(test_file)
#    testset1 = ml.VectorDataSet(test_file, labelsColumn=0)
#    testset2 = ml.VectorDataSet(test_file, labelsColumn=0)
#    classifier.decisionSurface(sl, trainingset1, testset1)

    k2 = ml.ker.Polynomial(3)
#    k2 = ker.Gaussian(gamma = 0.5)
#    k1 = ml.ker.Polynomial(5)
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

    sl.train(trainingset1)
#    sl.save("linear_svm")
    result1 = sl.cv(testset1)
#    demo2d.setData(trainingset1)
#    demo2d.getData()
#    demo2d.decisionSurface(sl)
    result1.plotROC('roc_linear%s.pdf' % (id))
    print result1

#===============================================================================
# Non Linear Classifier
#===============================================================================

    snl.train(trainingset2)
    result2 = snl.cv(testset2)
#    snl.preproject(testset2)
    result2.plotROC('roc_nonlinear%s.pdf' % (id))
    print result2


#    classifier.scatter(trainingset1)
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
