'''
Created on Feb 8, 2011

@author: anol
'''
import feature
import glob
import csv
import time
import matplotlib
import numpy as np
from numpy.matlib import zeros, ones
matplotlib.use('TkAgg') # do this before importing pylab
import matplotlib.pyplot as plt
import numpy.linalg as lin
from mpl_toolkits.mplot3d.axes3d import Axes3D

ext_fig1 = plt.figure(1)
ext_fig2 = plt.figure(2)
ext_fig3 = plt.figure(3)

sel_fig = plt.figure(4)
seled_fig = plt.figure(5)
cla_fig = plt.figure(6)

fea1 = ext_fig1.add_subplot(111)
fea1.grid()
fea2 = ext_fig2.add_subplot(111)
fea2.grid()
fea3 = ext_fig3.add_subplot(111)
fea3.grid()
#fea4 = ext_fig.add_subplot(3, 2, 5)
#fea4.grid()
#fea5 = ext_fig.add_subplot(3, 2, 6)
#fea5.grid()

fx = sel_fig.add_subplot(111)
fx.grid()

seledx = Axes3D(seled_fig)
seledx.grid()

classx = Axes3D(cla_fig)
classx.grid()

def extraction(files, c = 1):
    feature1 = feature.extraction.first_order_stat(files)
    feature2 = feature.extraction.moment_base(files)
#    feature3 = feature.extraction.fourier(files)
#    with open('class%s.csv' % (i), "w") as fd:
#        write = csv.writer(fd, delimiter = '\t', quotechar = '|', quoting = csv.QUOTE_MINIMAL)
#        header = feature1.keys() + feature2.keys()
#        fd.write('class\t')
#        write.writerow(header)
#        for i in range(len(files)):
#            values = []
#            fd.write("%s\t" % (i))
#            for h in header:
#                values.append(feature1[h][i])
#            write.writerow(values)
    features = []
    features.append(feature1.items() + feature2.items())#+ feature3.items())
    return features

if __name__ == '__main__':
    kfiles = glob.glob('../dataset/training_set/k*.jpg')
    cfiles = glob.glob('../dataset/training_set/c*.jpg')

#===============================================================================
# Feature Extraction class k
#===============================================================================
    c1 = extraction(kfiles, 1)
    c2 = extraction(cfiles, 2)

    x1 = []
    x2 = []
    xlabel = []
    for f in range(len(c1[0])):
        xlabel.append(c1[0][f][0])
        x1.append(c1[0][f][1])
        x2.append(c2[0][f][1])
#    win = fig.canvas.manager.window

    y1 = ones((1, len(x1[0])))
    y2 = ones((1, len(x2[0]))) * 2

    X = np.concatenate((x1, x2), axis = 1)
    y = np.concatenate((y1, y2), axis = 1)

# Plot mean and variance
    fea1.set_xlabel('STD')
    fea1.set_ylabel('RMS')
    fea1.set_title('Color Statistic')
    fea1.plot(x1[0], x1[1], 'rx')
    fea1.plot(x2[0], x2[1], 'bo')
# Plot Area
    fea2.set_title('Contour Area')
    fea2.plot(x1[8], 'b-')
    fea2.plot(x2[8], 'g.-')

## Plot Hu
    fea3.set_title('Moment of HU')
    fea3.plot(x1[2], 'b-')
    fea3.plot(x1[3], 'r-')
    fea3.plot(x1[4], 'g-')
    fea3.plot(x1[5], 'y-')
    fea3.plot(x1[6], 'c-')
    fea3.plot(x1[7], 'm-')
    fea3.plot(x1[9], 'k-')

    fea3.plot(x2[2], 'r--')
    fea3.plot(x2[3], 'g--')
    fea3.plot(x2[4], 'b--')
    fea3.plot(x2[5], 'y--')
    fea3.plot(x2[6], 'c--')
    fea3.plot(x2[7], 'm--')
    fea3.plot(x2[9], 'k--')

#===============================================================================
# Feature Selection
#===============================================================================
    FDR = []
    for i in range(len(X)):
        FDR.append(feature.selection.FDR_comp(X, y, i))
    sel = np.array(FDR)
    fx.bar(range(len(X)), FDR, 0.05, color = 'r')
    fx.set_ylabel('Ratio')
    fx.set_xlabel('Feature')
    fx.set_title('Fisher\'s Discriminate Ratio')
    fx.set_xticks(range(len(X)))
    fx.set_xticklabels((xlabel))
#===============================================================================
# Machine Learning
#===============================================================================
# Select hu5, hu6 and h7
    ind = (sel > 50).nonzero()[0]
    y1_temp = (y == 1).nonzero()[1]
    y2_temp = (y == 2).nonzero()[1]
    f_seled = X.take(ind, axis = 0)
    f1 = f_seled.take(y1_temp, axis = 1)
    f2 = f_seled.take(y2_temp, axis = 1)
    seledx.scatter(f1[0][0], f1[1][0], f1[2][0], c = 'r', marker = '^')
    seledx.scatter(f2[0][0], f2[1][0], f2[2][0], c = 'b', marker = 'o')
#===============================================================================
# Linear Classifier
#===============================================================================

#===============================================================================
# Non Linear Classifier
#===============================================================================

    plt.show()
