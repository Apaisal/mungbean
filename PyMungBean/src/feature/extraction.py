'''
Created on Feb 8, 2011

@author: anol
'''
import Image, ImageStat, numpy
import cv
from numpy.ma.core import arctan, max

def cvShiftDFT(src_arr, dst_arr):

    size = cv.GetSize(src_arr)
    dst_size = cv.GetSize(dst_arr)

#    if dst_size != size:
#        cv.Error(cv.CV_StsUnmatchedSizes, "cv.ShiftDFT", "Source and Destination arrays must have equal sizes", __FILE__, __LINE__)

    if(src_arr is dst_arr):
        tmp = cv.CreateMat(size[1] / 2, size[0] / 2, cv.GetElemType(src_arr))

    cx = size[0] / 2
    cy = size[1] / 2 # image center

    q1 = cv.GetSubRect(src_arr, (0, 0, cx, cy))
    q2 = cv.GetSubRect(src_arr, (cx, 0, cx, cy))
    q3 = cv.GetSubRect(src_arr, (cx, cy, cx, cy))
    q4 = cv.GetSubRect(src_arr, (0, cy, cx, cy))
    d1 = cv.GetSubRect(src_arr, (0, 0, cx, cy))
    d2 = cv.GetSubRect(src_arr, (cx, 0, cx, cy))
    d3 = cv.GetSubRect(src_arr, (cx, cy, cx, cy))
    d4 = cv.GetSubRect(src_arr, (0, cy, cx, cy))

    if(src_arr is not dst_arr):
#        if(not cv.CV_ARE_TYPES_EQ(q1, d1)):
#            cv.Error(cv.CV_StsUnmatchedFormats, "cv.ShiftDFT", "Source and Destination arrays must have the same format", __FILE__, __LINE__)

        cv.Copy(q3, d1)
        cv.Copy(q4, d2)
        cv.Copy(q1, d3)
        cv.Copy(q2, d4)

    else:
        cv.Copy(q3, tmp)
        cv.Copy(q1, q3)
        cv.Copy(tmp, q1)
        cv.Copy(q4, tmp)
        cv.Copy(q2, q4)
        cv.Copy(tmp, q2)

def fourier(name_images):
    '''
    '''
    features = dict()
    for name in name_images:
        A = cv.LoadImageM(name, cv.CV_LOAD_IMAGE_GRAYSCALE)
        cv.Threshold(A, A, 100, 255, cv.CV_THRESH_TOZERO_INV)
        cv.ShowImage('fo', A)
        real = cv.CreateMat(A.cols, A.rows, cv.CV_64FC1)
        imagine = cv.CreateMat(A.cols, A.rows, cv.CV_64FC1)
        complex = cv.CreateMat(A.cols, A.rows, cv.CV_64FC2)
        cv.Threshold(A, A, 90, 255, cv.CV_THRESH_TOZERO_INV)
        cv.Scale(A, real, 1.0, 0.0)
        cv.SetZero(imagine)
        cv.Merge(real, imagine, None, None, complex)
#        cv.DCT(real, real, cv.CV_DXT_FORWARD)

        cv.DFT(complex, complex, cv.CV_DXT_SCALE, 0)
        cv.SetZero(real)
        cv.SetZero(imagine)
        cv.Split(complex, real, imagine, None, None)
        cv.Pow(real, real, 2.0)
        cv.Pow(imagine, imagine, 2.0)
        cv.Add(real, imagine, real, None)
        cv.Pow(real, real, 0.5)
#
#        # Compute log(1 + Mag)
#        cv.AddS(real, cv.ScalarAll(1.0), real, None) # 1 + Mag
#        cv.Log(real, real) # log(1 + Mag)
        cvShiftDFT(real, real)
        cvShiftDFT(imagine, imagine)

        cv.ShowImage('real', real)
#        cv.ShowImage('imagine', imagine)
        cv.WaitKey()
    return features

def moment_base(name_images):
    features = dict()
    for name in name_images:
        A = cv.LoadImageM(name, cv.CV_LOAD_IMAGE_GRAYSCALE)
        cv.Threshold(A, A, 90, 255, cv.CV_THRESH_BINARY_INV)
        chaincode(A)
        area = findcontoursarea(A)

        moment = cv.Moments(A)
        hu = cv.GetHuMoments(moment)
        features[name] = {  "hu":hu
#                          , "centroid": (moment.m10 / moment.m00, moment.m01 / moment.m00)
#                          , "orientation" : orientation(moment)
                          , "area" : area
                          }
    normalize(features)
    return features

def findcontoursarea(img):
    storage = cv.CreateMemStorage(0)
    contours = cv.FindContours(img, storage, cv.CV_RETR_LIST, cv.CV_CHAIN_APPROX_SIMPLE, (0, 0))
#    img_contour = cv.CreateImage(cv.GetSize(img), 8, 1)
    cv.SetZero(img)
    _contours = contours
    area = 0
    while _contours:
        cv.DrawContours(img, _contours , cv.Scalar(255, 255, 255, 0), cv.Scalar(0, 0, 255, 0), 0, cv.CV_FILLED, 8, (0, 0))
        area += cv.ContourArea(_contours)
        _contours = _contours.h_next()
#    cv.ShowImage("c", img_contour)
#    cv.WaitKey()
#    if cv.CheckContourConvexity(contours) == 1:
#        cv.convexHull()

    return area

def perimeter(name_images):
    features = dict()
    for name in name_images:
        A = cv.LoadImageM(name, cv.CV_LOAD_IMAGE_GRAYSCALE)
        cv.Threshold(A, A, 90, 255, cv.CV_THRESH_BINARY_INV)

    return features

def chaincode(img):
    storage = cv.CreateMemStorage(0)
    contours = cv.FindContours(img, storage, cv.CV_RETR_LIST, cv.CV_CHAIN_CODE, (0, 0))

#    img_contour = cv.CreateImage(cv.GetSize(img), 8, 3)
#    print contours
#    print len(contours)
#       cv.ShowImage("c", img_contour)
#    cv.WaitKey()

def first_order_stat(name_images):

    features = dict()
    for name in name_images:
        im = Image.open(name, 'r').convert('RGB')
        img = im.split()
        # Get green component
        img = img[1].point(lambda i: i < 90 and i)
#        img.save('temp.jpg', 'JPEG')
#        img.show()
        statis = ImageStat.Stat(img)
        features[name] = {  'mean':statis._getmean()[0]
                            , 'stddev':statis._getstddev()[0]
                            , 'var':statis._getvar()[0]
                            , 'count':statis._getcount()[0]
                            , 'sum':statis._getsum()[0]
                            , 'sum2':statis._getsum2()[0]
                            , 'rms':statis._getrms()[0]
#                            , 'median':statis._getmedian()[0]
#                            , 'extrema':statis._getextrema()[0]
                            }

    normalize(features)
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

def normalize(features):
    '''
    '''
    lfeature = []
    count = len(features.values()[0])
    for i in range(count):
        for file in features:
            lfeature.append(features[file].values()[i])
        lfeature.sort()
        max = lfeature[-1]
        min = lfeature[0]
        del lfeature[:]
        for file in features:
            keys = features[file].keys()
            if keys[i] == 'hu':
                break
            value = features[file][keys[i]]
            value -= min
            if value != 0:
                value /= (max - min)
            features[file][keys[i]] = value
