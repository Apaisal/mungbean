'''
Created on May 4, 2011

@author: anol
'''
import cv

WIDTH = 101
HEIGTH = 101

WHITE = cv.Scalar(0xff, 0xff, 0xff, 0)
BLACK = cv.Scalar(0, 0, 0, 0)

def xfrange(start, stop, step):
    while start < stop:
        yield start
        start += step

def Rectangular(img, ptr1 = (25,25), ptr2 = (76,76)):
    rect = cv.CloneImage(img)
    cv.Zero(rect)
    cv.Zero(img)
    if (ptr1[0] > ptr2[0]) or (ptr1[1] > ptr2[1]):
        print "Please check size."
        return
    pt1 = ptr1
    pt2 = ptr2
    cv.Rectangle(rect, pt1, pt2, WHITE, 1, 8)
    storage = cv.CreateMemStorage()
    seq = cv.FindContours(rect, storage, cv.CV_RETR_LIST, cv.CV_CHAIN_APPROX_SIMPLE)
    cv.DrawContours(img, seq, WHITE, BLACK, -1, cv.CV_FILLED, 8)
    return img

if __name__ == '__main__':
    size = (WIDTH, HEIGTH)

    img = cv.CreateImage(size, cv.IPL_DEPTH_8U, 1)
    cv.Zero(img)
    pt1 = (25, 25)
    pt2 = (76, 76)
    cv.Rectangle(img, pt1, pt2, WHITE, 1, 8)
    rect = cv.CloneImage(img)
    cv.Zero(rect)
    storage = cv.CreateMemStorage()
    seq = cv.FindContours(img, storage, cv.CV_RETR_LIST, cv.CV_CHAIN_APPROX_SIMPLE)
    cv.DrawContours(rect, seq, WHITE, BLACK, -1, cv.CV_FILLED, 8)

    for deg in xfrange(0.0, 360.0, 0.5):
        dst = cv.CloneImage(rect)
        cv.Zero(dst)
        mapMatrix = cv.CreateMat(2, 3, cv.CV_32FC1)
        cv.Zero(mapMatrix)
        cv.GetRotationMatrix2D((cv.Round(size[0] * 0.5) + 1, cv.Round(size[0] * 0.5) + 1), deg, 1.0, mapMatrix)
        cv.WarpAffine(rect, dst, mapMatrix)
        moment = cv.Moments(dst, 1)
        hu = cv.GetHuMoments(moment)

#        print deg, hu
        print "%3.2f\t%+8.3e\t%+8.3e\t%+8.3e\t%+8.3e\t%+8.3e\t%+8.3e\t%+8.3e\n" % (deg, hu[0], hu[1], hu[2], hu[3], hu[4], hu[5], hu[6])
#        cv.ShowImage('Rectangular', rect)
        cv.ShowImage('Box', dst)
        if cv.WaitKey(10) == 27:
            break
    cv.DestroyWindow('Box')
