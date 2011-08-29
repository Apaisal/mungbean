'''
Created on Aug 29, 2011

@author: anol
'''
import os
import fnmatch
import cv

def GetDataSet(path):
#===========================================================================
# File Crawler
#===========================================================================   
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
                    d[file]=img
                    fileStruct[kind].append(d)        
            fileStruct[kind].sort()                
    return fileStruct


if __name__ == '__main__':
    rootTrainingSet = '../dataset/training_set'
    rootTestSet = '../dataset/test_set'
    dataSet = {}
    dataSet['training'] = GetDataSet(rootTrainingSet)
    dataSet['test'] = GetDataSet(rootTestSet)
    
