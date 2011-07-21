'''
Created on May 30, 2011

@author: anol
'''
import cv
from numpy import fft
import numpy as np
import matplotlib.pyplot as plt
from feature.extraction import ellipticFS

if __name__ == '__main__':
    
    imgfile = '/home/anol/workspace/PyMungBean/dataset/multiple/Picture23.jpg'
    org_img = cv.LoadImage(imgfile, cv.CV_LOAD_IMAGE_COLOR)
    img = cv.CreateImage((640, 380), 8, 3)
    cv.Resize(org_img, img, cv.CV_INTER_LINEAR)
    cv.ShowImage("org image", img)
    output_img = cv.CloneImage(img)
    grayimg = cv.CreateImage(cv.GetSize(img), 8, 1)
    cv.Zero(grayimg)
    cv.CvtColor(img, grayimg, cv.CV_RGB2GRAY)
    
    dst = cv.CreateImage(cv.GetSize(img), 8, 3)
    
    hsv = cv.CreateImage(cv.GetSize(img), 8, 3)
       
    s_plane = cv.CreateImage(cv.GetSize(img), 8, 1)
 
    cv.CvtColor(img, hsv, cv.CV_RGB2HSV)

    cv.Split(hsv, None, s_plane, None, None)
    
#    cv.ShowImage("s plane", s_plane)

    gs = cv.CloneImage(s_plane)
    cv.Zero(gs)

    cv.Smooth(s_plane, s_plane, cv.CV_MEDIAN, 9, 9)
    cv.ShowImage("median", s_plane)
    cv.WaitKey()
    cv.Add(s_plane, s_plane, gs)
    
    cv.Threshold(gs, gs, 150, 255, cv.CV_THRESH_BINARY)

    seq = cv.FindContours(gs, cv.CreateMemStorage(), cv.CV_RETR_LIST, cv.CV_CHAIN_APPROX_SIMPLE, (0, 0))

    ext_color = cv.CV_RGB(255, 255, 255)
    hole_color = cv.CV_RGB(0, 0, 255)
    fore_img = cv.CloneImage(gs)
    cv.Zero(fore_img)
    while seq:
        cv.DrawContours(fore_img, seq, ext_color, hole_color, 0, thickness= -1, lineType=8, offset=(0, 0))
        seq = seq.h_next()
        
    seq = cv.FindContours(fore_img, cv.CreateMemStorage(), cv.CV_RETR_LIST, cv.CV_CHAIN_APPROX_SIMPLE, (0, 0))

    while seq:
        cv.Zero(dst)
        cv.DrawContours(dst, seq, ext_color, hole_color, 0, thickness= -1, lineType=8, offset=(0, 0))
        seed = cv.CloneImage(dst)
#        cv.And(seed, dst, seed)
        gray_seed = cv.CreateImage(cv.GetSize(seed), 8, 1)
#        cv.Zero(gray_seed)
        cv.CvtColor(seed, gray_seed, cv.CV_RGB2GRAY)
#
        cv.Threshold(gray_seed, gray_seed, 100, 255, cv.CV_THRESH_BINARY)
        moments = cv.Moments(gray_seed, True)
#        cv.ShowImage('binary', gray_seed)
#        cv.WaitKey()
        '''Get Centroid'''
        win_height = 120
        win_width = 120
        x_bar = cv.GetSpatialMoment(moments, 1, 0) / cv.GetSpatialMoment(moments, 0, 0)
        y_bar = cv.GetSpatialMoment(moments, 0, 1) / cv.GetSpatialMoment(moments, 0, 0)
        x_start = x_bar - win_width / 2
        y_start = y_bar - win_height / 2
        window = (cv.Round(x_start), cv.Round(y_start), win_width, win_height)
        cv.SetImageROI(gray_seed, window);
        num_pixel = cv.CountNonZero(gray_seed)
        print "A number of the kernel area : %s" % (num_pixel)
        if num_pixel > 3200:
            '''separate touching kernel'''
            print "Touching kernel"
            touching_grain = cv.CloneImage(gray_seed)
            cv.Zero(touching_grain)
            seq_b = cv.FindContours(gray_seed, cv.CreateMemStorage(), cv.CV_RETR_LIST, cv.CV_CHAIN_APPROX_NONE, (0, 0))

            real = []
            imag = []
            
            for x, y in seq_b:
                ''''''
                real.append(x)
                imag.append(y)

            comp = np.array([real, imag])
            
            #TODO '''Elliptic fourier series'''
            x, y, p = ellipticFS(seq_b, 2)
           
            plt.subplot(111)
            plt.plot(x, 'b-',y,'r-')
            plt.grid(True)
            plt.show()
        
            cv.ShowImage("bound", gray_seed)
            cv.WaitKey()    
        elif num_pixel < 1500:
            print 'flaw & imperfect kernel must reject'
            r = cv.CreateImage(cv.GetSize(output_img), 8, 1)
            g = cv.CreateImage(cv.GetSize(output_img), 8, 1)
            b = cv.CreateImage(cv.GetSize(output_img), 8, 1)          
            cv.Split(output_img, b, g, r, None)
            cv.SetImageROI(r, window);
            cv.Add(r, gray_seed, r)
            cv.ResetImageROI(r)
            cv.Merge(b, g, r, None, output_img)
            cv.ShowImage("output", output_img)
            cv.WaitKey()
        else:  
            print 'This is a single kernel'
#            Find HU moment
            moments = cv.Moments(gray_seed, True)
            h1, h2, h3, h4, h5, h6, h7 = cv.GetHuMoments(moments)
            
        cv.ResetImageROI(gray_seed);
        
        
        seq = seq.h_next()
        
    
