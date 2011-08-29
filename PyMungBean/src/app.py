'''
Created on Aug 29, 2011

@author: anol
'''
import os
import fnmatch
import cv
import csv

from PyML.containers import sequenceData, vectorDatasets
from PyML.feature_selection import featsel
from PyML.classifiers import svm
from PyML.demo import demo2d

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
        write = csv.writer(fd, delimiter = ',')
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

def FeatureSelection(dataSet, type):
    data = vectorDatasets.VectorDataSet(type)
    pass

if __name__ == '__main__':

    #===========================================================================
    # Initialization 
    #===========================================================================

    rootTrainingSet = '../dataset/training_set'
    rootTestSet = '../dataset/test_set'
    dataSet = {}
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
    print 'Feature selection'
    FeatureSelection(dataSet, 'training')
