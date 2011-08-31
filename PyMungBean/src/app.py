'''
Created on Aug 29, 2011

@author: anol
'''
import os
import fnmatch
import cv
import csv

from PyML.containers import vectorDatasets
from PyML.feature_selection import featsel
from PyML.classifiers import svm, multi

from PyML.classifiers.svm import loadSVM, SVM
from PyML.demo import demo2d
from PyML.feature_selection.featsel import OneAgainstRestSelect, FeatureSelector
from PyML.utils import misc

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
        write = csv.writer(fd, delimiter=',')
        for kindname, files  in dataSet[type].items():
            print kindname
            for dFile in files:
                for filename, img in dFile.items():
                    moments = cv.Moments(img, True)
                    hu = cv.GetHuMoments(moments)
                    dFile['image'] = img
                    dFile['filename'] = filename
                    dFile['feature'] = {'hu':hu}
                    del dFile[filename]
                    write.writerow(list(hu) + [kindname])

def Selection(dataSet, type):
    data = vectorDatasets.VectorDataSet(type, labelsColumn= -1)
    print featsel.featureCount(data)
    pass

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
    firstStep = True
    secondStep = True
    Train = False
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
    # Feature selection
    #===========================================================================
    traindata = vectorDatasets.VectorDataSet('training', labelsColumn= -1)
#    numFeatures = len(featsel.featureCount(traindata))
    mc = multi.OneAgainstRest(SVM(c=1000)) 
    ret = mc.cv(traindata)
    print "Time of Training : %f s" % (mc.getTrainingTime())
    
    testdata = vectorDatasets.VectorDataSet('test', labelsColumn= -1)
    ret = mc.test(testdata)
    print ret
    pass
