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


ax = plt.subplot(2, 2, 1)
ax.grid()
bx = plt.subplot(2, 2, 2)
bx.grid()
cx = plt.subplot(2, 2, 3)
cx.grid()
dx = plt.subplot(2, 2, 4)
dx.grid()
fx = plt.subplot(1, 1, 1)
fx.grid()

def extraction(files, c = 1):
    feature1 = feature.extraction.first_order_stat(files)
    feature2 = feature.extraction.moment_base(files)
    feature3 = feature.extraction.fourier(files)
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
    features.append(feature1.items() + feature2.items() + feature3.items())
    return features

if __name__ == '__main__':
    kfiles = glob.glob('../dataset/training_set/k*.jpg')
    cfiles = glob.glob('../dataset/training_set/c*.jpg')

#===============================================================================
# Feature Extraction class k
#===============================================================================
    c1 = extraction(kfiles, 1)
    c2 = extraction(cfiles, 2)
#    win = fig.canvas.manager.window
    x1 = np.array([c1[0][0][1], c1[0][1][1], c1[0][3][1], c1[0][2][1].T[0], c1[0][2][1].T[1], c1[0][2][1].T[2], c1[0][2][1].T[3], c1[0][2][1].T[4], c1[0][2][1].T[5], c1[0][2][1].T[6]])
    x2 = np.array([c2[0][0][1], c2[0][1][1], c2[0][3][1], c2[0][2][1].T[0], c2[0][2][1].T[1], c2[0][2][1].T[2], c2[0][2][1].T[3], c2[0][2][1].T[4], c2[0][2][1].T[5], c2[0][2][1].T[6]])
    y1 = ones((1, len(c1[0][0][1])))
    y2 = ones((1, len(c2[0][0][1]))) * 2

    X = np.concatenate((x1, x2), axis = 1)
    y = np.concatenate((y1, y2), axis = 1)
# Plot mean and variance
    ax.plot(c1[0][0][1], 'rx')
    ax.plot(c2[0][0][1], 'bo')

    bx.plot(c1[0][1][1], 'rx')
    bx.plot(c2[0][1][1], 'bo')

# Plot Hu
    cx.plot(c1[0][2][1], c1[0][2][1] * 0, 'rx')
    cx.plot(c2[0][2][1], c2[0][2][1] * 0, 'bo')
#    cx.plot(c1[0][2][1][0], c1[0][2][1][0] * 0, 'rx')
#    cx.plot(c2[0][2][1][0], c2[0][2][1][0] * 0, 'bo')


#    cx.plot(c1[0][2][1][1], c1[0][2][1][1] * 0, 'rx')
#    cx.plot(c2[0][2][1][1], c2[0][2][1][1] * 0, 'bo')

#    cx.plot(c1[0][2][1][2], c1[0][2][1][2] * 0, 'rx')
#    cx.plot(c2[0][2][1][2], c2[0][2][1][2] * 0, 'bo')

#    cx.plot(c1[0][2][1][3], c1[0][2][1][3] * 0, 'rx')
#    cx.plot(c2[0][2][1][3], c2[0][2][1][3] * 0, 'bo')
#
#    cx.plot(c1[0][2][1][4], c1[0][2][1][4] * 0, 'rx')
#    cx.plot(c2[0][2][1][4], c2[0][2][1][4] * 0, 'bo')
#
#    cx.plot(c1[0][2][1][5], c1[0][2][1][5] * 0, 'rx')
#    cx.plot(c2[0][2][1][5], c2[0][2][1][5] * 0, 'bo')
#
#    cx.plot(c1[0][2][1][6], c1[0][2][1][6] * 0, 'rx')
#    cx.plot(c2[0][2][1][6], c2[0][2][1][6] * 0, 'bo')
# Plot Area
    dx.plot(c1[0][3][1], 'b.')
    dx.plot(c2[0][3][1], 'g.')



#===============================================================================
# Feature Selection
#===============================================================================
    FDR = []
    for i in range(len(X)):
        FDR.append(feature.selection.FDR_comp(X, y, i))
    fx.plot(FDR)
#===============================================================================
# Machine Learning
#===============================================================================

#===============================================================================
# Linear Classifier
#===============================================================================

#===============================================================================
# Non Linear Classifier
#===============================================================================

    plt.show()
