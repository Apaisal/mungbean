from numpy import matlib as np
import numpy
from numpy import arange
import matplotlib
from matplotlib import pylab
pylab.rcParams['contour.negative_linestyle'] = 'solid'
from PyML.containers.vectorDatasets import VectorDataSet
from PyML.containers.labels import Labels

X = []
Y = []
#plotStr = ['or', 'ob']
plotStr = ['or', '+b']
xmin = 0
xmax = 1
ymin = 0
ymax = 1
def decisionSurface(classifier, trainingdata, testdata, fileName = None, **args) :

    data = trainingdata
    classifier.train(data)

    numContours = 3
    if 'numContours' in args :
        numContours = args['numContours']
    title = None
    if 'title' in args :
        title = args['title']
    markersize = 5
    fontsize = 'medium'
    if 'markersize' in args :
        markersize = args['markersize']
    if 'fontsize' in args :
        fontsize = args['fontsize']
    contourFontsize = 10
    if 'contourFontsize' in args :
        contourFontsize = args['contourFontsize']
    showColorbar = False
    if 'showColorbar' in args :
        showColorbar = args['showColorbar']
    show = True
    if fileName is not None :
        show = False
    if 'show' in args :
        show = args['show']

    # setting up the grid
    delta = 0.01
    if 'delta' in args :
        delta = args['delta']

#    x = arange(xmin, xmax, float(1) / testdata.labels.classSize[0])
#    y = arange(ymin, ymax, float(1) / testdata.labels.classSize[1])
    x = arange(xmin, xmax, delta)
    y = arange(ymin, ymax, delta)
#
    Z = numpy.zeros((len(x), len(y)), numpy.float_)
    gridX = numpy.zeros((len(x) * len(y), 2), numpy.float_)
    n = 0
    for i in range(len(x)) :
        for j in range(len(y)) :
            gridX[n][0] = x[i]
            gridX[n][1] = y[j]
            n += 1
    gridData = VectorDataSet(gridX)
    gridData.attachKernel(data.kernel)
    results = classifier.cv(gridData)

    n = 0
    for i in range(len(x)) :
        for j in range(len(y)) :
            Z[i][j] = results.decisionFunc[n]
            n += 1

    pylab.figure()
    im = pylab.imshow(numpy.transpose(Z),
                      interpolation = 'bilinear', origin = 'lower',
                      cmap = pylab.cm.gray, extent = (xmin, xmax, ymin, ymax))

    if numContours == 1 :
        C = pylab.contour(numpy.transpose(Z),
                          [0],
                          origin = 'lower',
                          linewidths = (3),
                          colors = 'black',
                          extent = (xmin, xmax, ymin, ymax))
    elif numContours == 3 :
        C = pylab.contour(numpy.transpose(Z),
                          [-1, 0, 1],
                          origin = 'lower',
                          linewidths = (1, 3, 1),
                          colors = 'black',
                          extent = (xmin, xmax, ymin, ymax))
    else :
        C = pylab.contour(numpy.transpose(Z),
                          numContours,
                          origin = 'lower',
                          linewidths = 2,
                          extent = (xmin, xmax, ymin, ymax))

    pylab.clabel(C,
                 inline = 1,
                 fmt = '%1.1f',
                 fontsize = contourFontsize)

    # plot the data
    scatter(data, markersize = markersize)
    xticklabels = pylab.getp(pylab.gca(), 'xticklabels')
    yticklabels = pylab.getp(pylab.gca(), 'yticklabels')
    pylab.setp(xticklabels, fontsize = fontsize)
    pylab.setp(yticklabels, fontsize = fontsize)

    if title is not None :
        pylab.title(title, fontsize = fontsize)
    if showColorbar :
        pylab.colorbar(im)

    # colormap:
    pylab.hot()
    if fileName is not None :
        pylab.savefig(fileName)
    if show :
        pylab.show()

def scatter(data, **args) :

    markersize = 5
    if 'markersize' in args :
        markersize = args['markersize']
    features = [0, 1]
    if 'features' in args :
        features = args['features']
    for c in range(data.labels.numClasses) :
        x1 = []
        x2 = []
        for p in data.labels.classes[c] :
            x = data.getPattern(p)
            x1.append(x[features[0]])
            x2.append(x[features[1]])
        pylab.plot(x1, x2, plotStr[c], markersize = markersize)


