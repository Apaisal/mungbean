'''
Created on Feb 8, 2011

@author: anol
'''
import Image, ImageStat, numpy as np
import cv
from numpy.ma.core import arctan, max, cos, sin, sum, sqrt
from math import pi

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

#        cv.ShowImage('real', real)
#        cv.ShowImage('imagine', imagine)
#        cv.WaitKey()
    return features

def moment_base(name_images):

    area = []
    hu1 = []
    hu2 = []
    hu3 = []
    hu4 = []
    hu5 = []
    hu6 = []
    hu7 = []

    features = {'area':area, 'hu1':hu1, 'hu2':hu2, 'hu3':hu3, 'hu4':hu4, 'hu5':hu5, 'hu6':hu6, 'hu7':hu7}
    for name in name_images:
        A = cv.LoadImageM(name, cv.CV_LOAD_IMAGE_COLOR)

#        chaincode(A)
#        cv.ShowImage('moment', A)
#        cv.WaitKey()
        hsv = cv.CreateImage(cv.GetSize(A), 8, 3)

        s_plane = cv.CreateImage(cv.GetSize(A), 8, 1)

        cv.CvtColor(A, hsv, cv.CV_RGB2HSV)

        cv.Split(hsv, None, s_plane, None, None)
        gs = cv.CloneImage(s_plane)
        cv.Zero(gs)

        cv.Smooth(s_plane, s_plane, cv.CV_MEDIAN, 3, 3)

        cv.Add(s_plane, s_plane, gs)

        cv.Threshold(s_plane, s_plane, 100, 255, cv.CV_THRESH_BINARY)
#        cv.ShowImage("median", s_plane)
#        cv.WaitKey()

        area.append(findcontoursarea(s_plane))

        moment = cv.Moments(s_plane, True)
#        print moment.m00
#        area.append(cv.GetSpatialMoment(moment, 0, 0))
        hu = cv.GetHuMoments(moment)
        hu1.append(hu[0])
        hu2.append(hu[1])
        hu3.append(hu[2])
        hu4.append(hu[3])
        hu5.append(hu[4])
        hu6.append(hu[5])
        hu7.append(hu[6])
#        features[name] = {  "hu":hu
##                          , "centroid": (moment.m10 / moment.m00, moment.m01 / moment.m00)
##                          , "orientation" : orientation(moment)
#                          , "area" : area
#                          }
    normalize(features)
    A = None
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

    mean = []
    var = []
    std = []
    rms = []
    features = {
                'mean':mean
                , 'var':var
#                 'std':std
#                , 'rms':rms
                }
    for name in name_images:
        with open(name, "r") as fd:
#        im = cv.LoadImage(name, cv.CV_LOAD_IMAGE_COLOR)
#        conv = cv.CreateImage(cv.GetSize(im), 8, 3)
#        img = cv.CreateImage(cv.GetSize(im), 8, 1)
#        cv.CvtColor(im, conv, cv.CV_RGB2HLS)
#        cv.Split(conv, None, None, img, None)
#        cv.Threshold(img, img, 48, 255, cv.CV_THRESH_TOZERO)
#        pi = Image.fromstring("L", cv.GetSize(img), img.tostring())
            im = Image.open(fd, 'r').convert('RGB')
            pi = im.split()

            # Get green component
            pi = pi[1].point(lambda i: i < 90 and i)

            statis = ImageStat.Stat(pi)
            mean.append(statis._getmean()[0])
            var.append(statis._getvar()[0])
    #        std.append(statis._getstddev()[0])
    #        rms.append(statis._getrms()[0])

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
    for feature in features:
        arr = np.array(features[feature])
        arr = (((arr - arr.min()) / (arr.max() - arr.min())) * 2.0) - 1
        features[feature] = arr

def sign(number):return cmp(number, 0)

def elliptic_coeff(seq , n, K, t):
    T = sum(t)
    f1 = T / (2 * pow(n, 2) * pow(pi, 2))
    f2 = 0
    f3 = 0
    tp = 0
    dp = 0
    tp_1 = 0
    dp_1 = 0
    x = (2 * n * pi) / T
    for p in range(1, K + 1):
        tp = sum(t[0:p])
        dp = sum(seq[0:p])
        delta_tp = tp - tp_1
        delta_dp = dp - dp_1
        if delta_dp != 0:
            f2 += (delta_dp / delta_tp) * (cos(x * tp) - cos(x * tp_1))
            f3 += (delta_dp / delta_tp) * (sin(x * tp) - sin(x * tp_1))
        tp_1 = tp
        dp_1 = dp


    return f1 * f2, f1 * f3

def ellipticFS(seq, harmonic, time):
    ''''''
    x = []
    y = []
    t = []
    seta = 0
    A0 = 0
    C0 = 0
    An = 0
    Cn = 0
#    fourier_power = 0
    K = len(seq)

    for p in seq:
        delta_xi = sign(6 - p) * sign(2 - p)
        delta_yi = sign(4 - p) * sign(p)
        delta_ti = 1 + ((sqrt(2) - 1) / 2) * (1 - pow(-1, p))
        x.append(delta_xi)
        y.append(delta_yi)
        t.append(delta_ti)

    T = sum(t)
    z = 2 * pi * time

#    for p in range(1, K):
#        for j in range(p):

    for n in range(1, harmonic + 1):
        seta = (z * n) / T
        an, bn = elliptic_coeff(x, n, K, t)
        cn, dn = elliptic_coeff(y, n, K, t)

        An += an * cos(seta)
        An += bn * sin(seta)
        Cn += cn * cos(seta)
        Cn += cn * sin(seta)

        """Fourier Power"""
#        fourier_power += pow(an, 2)
#        fourier_power += pow(bn, 2)
#        fourier_power += pow(cn, 2)
#        fourier_power += pow(dn, 2)

        if p == 1:
            A0 = An
            C0 = Cn

    x = A0 + An
    y = C0 + Cn

#    fourier_power /= 2

    return x, y#, fourier_power


#    lfeature = []
#    count = len(features.values()[0])
#    for i in range(count):
#        for file in features:
#            lfeature.append(features[file].values()[i])
#        lfeature.sort()
#        max = lfeature[-1]
#        min = lfeature[0]
#        del lfeature[:]
#        for file in features:
#            keys = features[file].keys()
#            if keys[i] == 'hu':
#                break
#            value = features[file][keys[i]]
#            value -= min
#            if value != 0:
#                value /= (max - min)
#            features[file][keys[i]] = value
