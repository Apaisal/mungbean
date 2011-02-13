'''
Created on Feb 8, 2011

@author: anol
'''
import Image, ImageStat
import cv
from numpy.ma.core import arctan

def fourier(name_images):
    '''
    '''
    features = dict()
    for name in name_images:
        A = cv.LoadImageM(name, cv.CV_LOAD_IMAGE_GRAYSCALE)
        real = cv.CreateMat(A.cols, A.rows, cv.CV_64FC1)
        imagine = cv.CreateMat(A.cols, A.rows, cv.CV_64FC1)
        complex = cv.CreateMat(A.cols, A.rows, cv.CV_64FC2)
    return features

def moment_base(name_images):
    features = dict()
    for name in name_images:
        A = cv.LoadImageM(name, cv.CV_LOAD_IMAGE_GRAYSCALE)
#        cv.Threshold(A, A, 110, 255, cv.CV_THRESH_BINARY_INV)
        cv.ShowImage("m", findcontours(A))
        cv.WaitKey()
        moment = cv.Moments(A)
        hu = cv.GetHuMoments(moment)
        features[name] = {  "hu":hu
                          , "centroid": (moment.m10 / moment.m00, moment.m01 / moment.m00)
                          , "orientation" : orientation(moment)

                          }
    return features

def findcontours(img):
    storage = cv.CreateMemStorage(0)
    contours = cv.FindContours(img, storage, cv.CV_RETR_LIST, cv.CV_CHAIN_APPROX_SIMPLE, (0, 0))
    img_contour = cv.CreateImage(cv.GetSize(img), 8, 1)
    cv.SetZero(img_contour)
    _contours = contours
#    while _contours:
    print len(_contours)
    cv.DrawContours(img_contour, _contours , cv.Scalar(0, 255, 0), cv.Scalar(0, 0, 255), 40, 2, 8, (0, 0))

#        _contours = contours.h_next()
    print img_contour

def getPerimeter(img):
    '''
    '''

def first_order_stat(name_images):

    features = dict()
    for name in name_images:
        A = Image.open(name)
        A = A.convert('L')
#        A = A.point(lambda i: i < 100 and i)
#        A.save('temp.jpg','JPEG')
        statis = ImageStat.Stat(A)
        features[name] = {  'mean':statis._getmean()[0]
                            , 'stddev':statis._getstddev()[0]
                            , 'var':statis._getvar()[0]
                            , 'count':statis._getcount()[0]
                            , 'sum':statis._getsum()[0]
                            , 'sum2':statis._getsum2()[0]
                            , 'rms':statis._getrms()[0]
                            , 'median':statis._getmedian()[0]
                            , 'extrema':statis._getextrema()[0]
                            }

    return  features

def orientation(moments):
    '''
    '''
    u00 = cv.GetCentralMoment(moments, 0, 0)
    u11 = cv.GetCentralMoment(moments, 1, 1)
    u20 = cv.GetCentralMoment(moments, 2, 0)
    u02 = cv.GetCentralMoment(moments, 0, 2)

    du20 = u20 / u00
    du02 = u02 / u00
    du11 = u11 / u00

    return arctan(2 * du11 / (du20 - du02)) / 2
