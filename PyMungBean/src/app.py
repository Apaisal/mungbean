'''
Created on Aug 29, 2011

@author: anol
'''
import os
import fnmatch
import cv
import csv
import numpy as np
from plot import plot3d

from PyML import ker
from PyML.containers import vectorDatasets
from PyML.feature_selection import featsel
from PyML.classifiers import svm, multi, ridgeRegression, knn

from PyML.classifiers.svm import loadSVM
#from PyML.demo import demo2d


def GetDataSet(path):
    sImgExt = '*.jpg'
    listKinds = os.listdir(path)
    fileStruct = {}
    for kind in listKinds:
        if kind[0] != '.':
            fileStruct[kind] = []
            for file in os.listdir(os.path.join(path, kind)):
                if fnmatch.fnmatch(file, sImgExt):
                    d = {}
                    img = cv.LoadImage(os.path.join(path, kind, file))
                    d[file] = img
                    fileStruct[kind].append(d)
            fileStruct[kind].sort()
    return fileStruct

def ImagePrepare(dataSet, type):
    for kindname, files  in dataSet[type].items():
        print kindname
        for dFile in files:
            for filename, img in dFile.items():
                hsv = cv.CreateImage(cv.GetSize(img), img.depth, img.channels)
                cv.CvtColor(img, hsv, cv.CV_RGB2HSV)
                sat = cv.CreateImage(cv.GetSize(hsv), hsv.depth, 1)
                cv.Split(hsv, None, sat, None, None)
                cv.Smooth(sat, sat, cv.CV_MEDIAN, 3, 3)
                cv.Threshold(sat, sat, 100, 255, cv.CV_THRESH_BINARY)
                dFile[filename] = sat

def FeatureExtract(dataSet, type):
    with open(type, "w") as fd:
        write = csv.writer(fd, delimiter=',', lineterminator='\n')
        for kindname, files  in dataSet[type].items():
            print kindname
            for dFile in files:
                for filename, img in dFile.items():
                    moments = cv.Moments(img, True)
                    hu = cv.GetHuMoments(moments)
                    dFile['image'] = img
                    dFile['filename'] = filename
                    dFile['feature'] = {'hu':list(hu)}
                    dFile['kind'] = kindname
                    del dFile[filename]
#                    write.writerow(list(hu) + [kindname])
                    write.writerow(list(hu) + [kindname])
def NormalizeByMaxMin(dataSet, type, maximum=None, minimum=None, show=False):
    hu = {}
    a = plot3d
    with open(type, "w") as fd:
        write = csv.writer(fd, delimiter=',', lineterminator='\n')
        for key, value in dataSet[type].items():
            
            h1 = []
            h2 = []
            h3 = []
            h4 = []
            h5 = []
            h6 = []
            h7 = []
            for element in value:
    #            np.append(h1, element['feature']['hu'][0],1)
                h1.append(element['feature']['hu'][0])
                h2.append(element['feature']['hu'][1])
                h3.append(element['feature']['hu'][2])
                h4.append(element['feature']['hu'][3])
                h5.append(element['feature']['hu'][4])
                h6.append(element['feature']['hu'][5])
                h7.append(element['feature']['hu'][6])
            h1 = np.array(h1)
            h2 = np.array(h2)
            h3 = np.array(h3)
            h4 = np.array(h4)
            h5 = np.array(h5)
            h6 = np.array(h6)
            h7 = np.array(h7)
            if maximum is None:
                maximum = []
                maximum.append(h1.max())
                maximum.append(h2.max())
                maximum.append(h3.max())
                maximum.append(h4.max())
                maximum.append(h5.max())
                maximum.append(h6.max())
                maximum.append(h7.max())
            if minimum is None:
                minimum = []
                minimum.append(h1.min())
                minimum.append(h2.min())
                minimum.append(h3.min())
                minimum.append(h4.min())
                minimum.append(h5.min())
                minimum.append(h6.min())
                minimum.append(h7.min())    
           
            h1 = (((h1 - minimum[0]) / (maximum[0] - minimum[0])) * 2.0) - 1
            h2 = (((h2 - minimum[1]) / (maximum[1] - minimum[1])) * 2.0) - 1
            h3 = (((h3 - minimum[2]) / (maximum[2] - minimum[2])) * 2.0) - 1
            h4 = (((h4 - minimum[3]) / (maximum[3] - minimum[3])) * 2.0) - 1
            h5 = (((h5 - minimum[4]) / (maximum[4] - minimum[4])) * 2.0) - 1
            h6 = (((h6 - minimum[5]) / (maximum[5] - minimum[5])) * 2.0) - 1
            h7 = (((h7 - minimum[6]) / (maximum[6] - minimum[6])) * 2.0) - 1
#            h1 = (h1 - h1.mean()) / h1.std()
#            h2 = (h2 - h2.mean()) / h2.std()
#            h3 = (h3 - h3.mean()) / h3.std()
#            h4 = (h4 - h4.mean()) / h4.std()
#            h5 = (h5 - h5.mean()) / h5.std()
#            h6 = (h6 - h6.mean()) / h6.std()
#            h7 = (h7 - h7.mean()) / h7.std()
#            h1 /= h1.std()
#            h2 /= h2.std()
#            h3 /= h3.std()
#            h4 /= h4.std()
#            h5 /= h5.std()
#            h6 /= h6.std()
#            h7 /= h7.std()
            hu[key] = [h1, h2, h3, h4, h5, h6, h7]

            for element in range(len(value)):
                value[element]['feature']['hu'] = [h1[element], h2[element], h3[element], h4[element], h5[element], h6[element], h7[element]]
                write.writerow(value[element]['feature']['hu'] + [key])
    if show:
        a.plotHu(hu)
    return (maximum, minimum)


def TrainingFeature(dataSet, type):
    data = vectorDatasets.VectorDataSet(type, labelsColumn= -1)
    s = multi.OneAgainstRest(svm.SVM())
    s.train(data, saveSpace=False)
    s.save("svm.data")

def TestFeature(dataSet, type):
    data = vectorDatasets.VectorDataSet(type, labelsColumn= -1)
    s = loadSVM("svm.data", data)
    ret = s.test(data, saveSpace=False)
    rec = s.cv(data)
    pass

if __name__ == '__main__':
    firstStep = False
    secondStep = True
    Train = True
    Test = True
    #===========================================================================
    # Initialization 
    #===========================================================================
    rootTrainingSet = '../dataset/training_set'
    rootTestSet = '../dataset/test_set'
    dataSet = {}

    if firstStep:
        print 'Loading Trianing set'
        dataSet['training'] = GetDataSet(rootTrainingSet)
        print 'Loading Test set'
        dataSet['test'] = GetDataSet(rootTestSet)

        #===========================================================================
        # Image Preparation
        #===========================================================================
        print 'Prepare image of a Trianing set'
        ImagePrepare(dataSet, 'training')
        print 'Prepare image of a Test set'
        ImagePrepare(dataSet, 'test')

        #===========================================================================
        # Feature Extraction
        #===========================================================================
        print 'Extract feature in a image of a Trianing set'
        FeatureExtract(dataSet, 'training')
        print 'Extract feature in a image of a Test set'
        FeatureExtract(dataSet, 'test')

    #===========================================================================
    # Normalization 
    #===========================================================================
        print 'Normalization training set'
        max, min = NormalizeByMaxMin(dataSet, 'training', show=False)
        print 'Normalization test set'
        NormalizeByMaxMin(dataSet, 'test',maximum=max, minimum=min, show=False)

    #===========================================================================
    # Feature selection
    #===========================================================================
    
    traindata = vectorDatasets.VectorDataSet('training', labelsColumn= -1)
    testdata = vectorDatasets.VectorDataSet('test', labelsColumn= -1)
    
#    chooseclass = ['authong1', 'chainat72']
#    chooseclass = ['authong1','kamphangsean2']
#    chooseclass = ['authong1','motoso1']
#    chooseclass = ['chainat72','kamphangsean2']
#    chooseclass = ['chainat72','motoso1']
#    chooseclass = ['kamphangsean2','motoso1']
    chooseclass = [ \
#                   'authong1', \
                   'chainat72', \
                   'kamphangsean2', \
#                   'motoso1'
                   ]
    
    trainclasspair = traindata.__class__(traindata, classes=chooseclass)
    #traindata.normalize()
#    traindata.scale(1.0)

    
    testclasspair = traindata.__class__(testdata, classes=chooseclass)
    #testdata.normalize()
#    testdata.scale(1.0)
#    rfe = featsel.RFE()

    #===========================================================================
    # Machine Learning & Classification
    #===========================================================================
    s = None
    #        k= knn.KNN(1)
#        k = ridgeRegression.RidgeRegression(1)
#        k = svm.SVR()
    k = svm.SVM(\
#                                      ker.Gaussian(gamma=0.1) , \
#                                      ker.Polynomial(2), \
                                      c=10, \
#                                      optimizer='mysmo' \
                                      )
    if trainclasspair.labels.numClasses > 2:
        print "MultiClass Classifier"
        s = multi.OneAgainstRest(k)
    else:
        print "Two Class Classifier"
        s = k
        
    print "==========================================================================="
    print "\nCross validation Training set"
    print s.cv(trainclasspair)
    print "==========================================================================="
    print "\nCross validation Test set"
    print s.cv(testclasspair)

    print "==========================================================================="
    print "\nTraining DataSet"
    s.train(trainclasspair)
#    s.train(testdata)

    print "==========================================================================="
    print "\nTesting DataSet"
    print s.test(testclasspair)
    print "==========================================================================="
