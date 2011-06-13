'''
Created on May 30, 2011

@author: anol
'''
import cv
from numpy import fft
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    
    imgfile = '/home/anol/Pictures/Picture23.jpg'
    org_img = cv.LoadImage(imgfile, cv.CV_LOAD_IMAGE_COLOR)
    img = cv.CreateImage((640, 380), 8, 3)
    cv.Resize(org_img, img, cv.CV_INTER_LINEAR)
    cv.ShowImage("org image", img)
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
#        cv.Threshold(gray_seed, gray_seed, 100, 255, cv.CV_THRESH_BINARY)
        moments = cv.Moments(gray_seed, 0)
        
        '''Get Centroid'''
        win_height = 120
        win_width = 120
        x_bar = cv.GetSpatialMoment(moments, 1, 0) / cv.GetSpatialMoment(moments, 0, 0)
        y_bar = cv.GetSpatialMoment(moments, 0, 1) / cv.GetSpatialMoment(moments, 0, 0)
        window = (x_bar - win_width / 2, y_bar - win_height / 2, win_width, win_height)
        cv.SetImageROI(gray_seed, window);
#        hu = cv.GetHuMoments(moment)
#        print hu
        num_pixel = cv.CountNonZero(gray_seed)
        if num_pixel > 3200:
            '''separate touching kernel'''
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
            a = fft.fft2(comp)
            plt.subplot(111)
            plt.plot(a[0], 'b-',a[1],'r-')
            plt.grid(True)
            plt.show()

#            cv.DFT(src, dst, cv.CV_DXT_SCALE, 0)
#            cv.Rectangle(touching_grain, (x, y), (x, y), ext_color)            
            cv.ShowImage("bound", touching_grain)
            cv.WaitKey()    
        elif num_pixel < 1500:
            '''flaw & imperfect kernel 
                reject'''
        else:  
            '''  
            cv.ShowImage("dst", gray_seed)
            cv.WaitKey()'''
        cv.ResetImageROI(gray_seed);
        
        
        seq = seq.h_next()
