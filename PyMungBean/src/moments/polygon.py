#!/usr/bin/env python
# -*- coding: UTF-8 -*-

'''
Created on May 4, 2011

@author: anol
'''
import cv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d as p3
import matplotlib.cm as cm
from matplotlib import pyplot
#from matplotlib.cm import 
import os, sys
import string


WIDTH = 101
HEIGTH = 101

WHITE = cv.Scalar(0xff, 0xff, 0xff, 0)
BLACK = cv.Scalar(0, 0, 0, 0)

def xfrange(start, stop, step):
    while start < stop:
        yield start
        start += step

WINDOW_NAME = 'Polygon'

def onMouse(event, x, y, flags, param):
    ''''''

    if event == cv.CV_EVENT_LBUTTONDBLCLK:
        print "LB DB CLK"
        param[0].append((x, y))
        img = param[1]
        cv.Circle(img, (x, y), 0, WHITE)
    if event == cv.CV_EVENT_LBUTTONDOWN:
        print "Down"
        param[0].append((x, y))
        img = param[1]
        cv.Circle(img, (x, y), 0, WHITE)
    if event == cv.CV_EVENT_LBUTTONUP:
        print "Up"
    if flags == cv.CV_EVENT_FLAG_CTRLKEY:
        print "CTRL"
        
def help():
    print "d = remove image"
    print "m = rotate"
    print "e = create ellipse"
    print "c = create circle"
    print "r = create rectangle"
    print "p = create polygon"

if __name__ == '__main__':
    help()
    
    size = (WIDTH, HEIGTH)
    points = []
    img = cv.CreateImage(size, cv.IPL_DEPTH_8U, 1)
    cv.Zero(img)
    cv.NamedWindow(WINDOW_NAME)
    cv.SetMouseCallback(WINDOW_NAME, onMouse, [points, img])

    result = ''
    data = []
    while True:
        key = -1
        cv.ShowImage(WINDOW_NAME, img)
        key = cv.WaitKey(10)
        if key == "" or key == -1:
            continue
        elif key != -1:
            key &= 255
       
        if key == 27:
            break
        elif key == ord('d'):
            cv.Zero(img)
            del points[:]
            
        elif key == 10:
            cv.FillConvexPoly(img, points, WHITE)
            
        elif key == ord('m'):
            result = ''
            for deg in xfrange(0.0, 360.0, 1.0):
                dst = cv.CloneImage(img)
                cv.Zero(dst)
                mapMatrix = cv.CreateMat(2, 3, cv.CV_32FC1)
                cv.Zero(mapMatrix)
                cv.GetRotationMatrix2D((cv.Round(size[0] * 0.5) + 1, cv.Round(size[0] * 0.5) + 1), deg, 1.0, mapMatrix)
                cv.WarpAffine(img, dst, mapMatrix)
                moment = cv.Moments(dst, 1)
                hu = cv.GetHuMoments(moment)
                data.append(hu)
                log = "%3.2f\t%+8.3e\t%+8.3e\t%+8.3e\t%+8.3e\t%+8.3e\t%+8.3e\t%+8.3e\n" % (deg, hu[0], hu[1], hu[2], hu[3], hu[4], hu[5], hu[6])
                result += log
                print log
                cv.ShowImage('Box', dst)
                if cv.WaitKey(10) == 27:
                    break
            cv.DestroyWindow('Box')
            
        elif key == ord('e'):
            import ellipse
            print "Put ellipce size"
            x = raw_input('X :')
            y = raw_input('Y :')
            if x == "" or y == "":
                x = 80
                y = 50
            else:
                x = string.atoi(x)
                y = string.atoi(y)
                
            ellipse.Ellipse(img, (x, y))
            
        elif key == ord('r'):
            import rectangular
            print "Put rectangular size"
            x1 = raw_input('point1->X :')
            y1 = raw_input('point1->Y :')
            x2 = raw_input('point2->X :')
            y2 = raw_input('point2->Y :')
            
            if x1 == "" or y1 == "" or x2 == "" or y2 == "":
                x1 = 25
                y1 = 25
                x2 = 50
                y2 = 50
            else:
                x1 = string.atoi(x1)
                y1 = string.atoi(y1)
                x2 = string.atoi(x2)
                y2 = string.atoi(y2)
            rectangular.Rectangular(img, (x1, y1), (x2, y2))
            
        elif key == ord('c'):
            import circle
            print "Put radius of circle"
            r = raw_input('R :')
            if r == "" :
                r = 40
            else:
                r = string.atoi(r)
            circle.Circle(img, r)
            
        elif key == ord('s'):
            dir = "./result/"
            if not os.path.exists(dir):
                os.mkdir(dir)
            filename = raw_input('Name of File :')
            if filename == "":
                print "No filename"
                continue
            if os.path.exists(dir + filename):
                print "File exist"
                continue
            fd = open(dir + filename, 'w')
            fd.write(result)
            fd.close()
            cv.SaveImage(dir + filename + ".png", img)
            print "Save file completed"
            
#        elif key == ord('x'):
#            dst = cv.CloneImage(img)
#            cv.Zero(dst)
#            moments = cv.Moments(img)
#            m00 = cv.GetSpatialMoment(moments, 0, 0)
#            m01 = cv.GetSpatialMoment(moments, 0, 1)
#            m10 = cv.GetSpatialMoment(moments, 1, 0)
#
#            src_pt = (cv.Round(m01 / m00), cv.Round(m10 / m00));
#            cen_pt = (cv.Round(WIDTH * 0.5) + 1, cv.Round(HEIGTH * 0.5) + 1)
#            diff = [src_pt[0] - cen_pt[0] , src_pt[1] - cen_pt[1]]
#
#            rect = (diff[0], diff[1], WIDTH, HEIGTH)
#            cv.SetImageROI(img, rect)
#            roi_size = cv.GetSize(img)
#
#            if diff[0] > 0 and diff[1] > 0:
#                diff[0] = diff[1] = 0
#            elif diff[0] < 0 and diff[1] < 0:
#                diff[0] = abs(diff[0])
#                diff[1] = abs(diff[1])
# #FIXME:
#            elif diff[0] < 0 and diff[1] > 0:
#                diff[0] = 0
#                diff[1] = abs(diff[1])
#            elif diff[0] > 0 and diff[1] < 0:
#                diff[0] = abs(diff[0])
#                diff[1] = 0
#            dst_rect = (diff[0], diff[1], roi_size[0], roi_size[1])
#            cv.SetImageROI(dst, dst_rect)
#            cv.Copy(img, dst)
#            cv.ResetImageROI(img)
#            cv.ResetImageROI(dst)
#            cv.ShowImage('move', dst)
##            cv.WaitKey()
#            print 'plot'
        elif key == ord('p'):
            fig = plt.figure()

            select = raw_input('Selection order(1-7) or all(a) to show :')
            if select in str(range(1, 8)):

                ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], polar=True)
                N = 20
                theta = np.arange(0.0, 2 * np.pi, 2 * np.pi / N)
                radii = 10 * np.random.rand(N)
                width = 1 #np.pi / 4 * np.random.rand(N)
                bars = ax.bar(theta, radii, width=width, bottom=0.0)
                for r, bar in zip(radii, bars):
                    bar.set_facecolor(plt.jet(r / 10.))
                    bar.set_alpha(0.5)
            elif select == 'a':
                ax = p3.Axes3D(fig)
                for c, z in zip(['r', 'g', 'b', 'y'], [30, 20, 10, 0]):
                    xs = np.arange(20)
                    ys = np.random.rand(20)

                    # You can provide either a single color or an array. To demonstrate this,
                    # the first bar of each set will be colored cyan.
                    cs = [c] * len(xs)
                    cs[0] = 'c'
                    ax.bar(xs, ys, zs=z, zdir='y', color=cs, alpha=0.8)

                ax.set_xlabel('Degree')
                ax.set_ylabel('Hu')
                ax.set_zlabel('Value')
#            

            plt.show()
        else:
            continue    
        
#            print 'plot'
