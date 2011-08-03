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

def Ellipse(img, size = (80,50)):
    ellip = cv.CloneImage(img)

    ellip_center = (cv.Round(WIDTH * 0.5) + 1, cv.Round(HEIGTH * 0.5) + 1)
    ellip_size = size
    cv.Zero(ellip)
    box = (ellip_center, ellip_size, 0.0)
    cv.EllipseBox(ellip, box, WHITE, 1, 8, 0)
    storage = cv.CreateMemStorage()
    seq = cv.FindContours(ellip, storage, cv.CV_RETR_LIST, cv.CV_CHAIN_APPROX_SIMPLE)
    cv.DrawContours(img, seq, WHITE, BLACK, -1, cv.CV_FILLED, 8)
    return img

if __name__ == '__main__':
    size = (WIDTH, HEIGTH)
    ellip = cv.CreateImage(size, cv.IPL_DEPTH_8U, 1)

    ellip_center = (cv.Round(WIDTH * 0.5) + 1, cv.Round(HEIGTH * 0.5) + 1)
    ellip_size = (80, 50)

    for deg in xfrange(0.0, 360.0, 0.5):
        cv.Zero(ellip)
        box = (ellip_center, ellip_size, deg)
        cv.EllipseBox(ellip, box, WHITE, 1, 8, 0)
        img = cv.CloneImage(ellip)
        cv.Zero(img)

        storage = cv.CreateMemStorage()
        seq = cv.FindContours(ellip, storage, cv.CV_RETR_LIST, cv.CV_CHAIN_APPROX_SIMPLE)
        cv.DrawContours(img, seq, WHITE, BLACK, -1, cv.CV_FILLED, 8)

        moment = cv.Moments(img, 1)
        hu = cv.GetHuMoments(moment)

        print "%3.2f\t%+8.3e\t%+8.3e\t%+8.3e\t%+8.3e\t%+8.3e\t%+8.3e\t%+8.3e\n" % (deg, hu[0], hu[1], hu[2], hu[3], hu[4], hu[5], hu[6])
#        cv.ShowImage('Rectangular', rect)
        cv.ShowImage('Box', img)
        if cv.WaitKey(10) == 27:
            break
    cv.DestroyWindow('Box')

