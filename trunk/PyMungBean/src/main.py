'''
Created on Feb 8, 2011

@author: anol
'''
import feature 
import glob
import cv
import optparse
import io
import csv

if __name__ == '__main__':
    files = glob.glob('../image/10022011/*.jpg')
    
    feature1 = feature.extraction.first_order_stat(files)
    feature2 = feature.extraction.moment_base(files)
#    feature3 = feature.extraction.fourier(files)
    
    with open('firstorderstat.csv', "wb") as fd:      
        write = csv.writer(fd, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        header = feature1.values()[0].keys() + feature2.values()[0].keys() 
        fd.write('file_name\t')
        write.writerow(header)
        
        for f in feature1:
            fd.write(f + '\t')
            values = feature1[f].values() + feature2[f].values()
            write.writerow(values)
            
#            
    
#    index = 0        
    
#    while True:
#            img = cv.LoadImage(files[index], cv.CV_LOAD_IMAGE_COLOR)
#            
#            hsv = cv.CreateImage(cv.GetSize(img), img.depth, img.nChannels)
#            s = cv.CreateImage(cv.GetSize(img), img.depth, 1)
#            gray = cv.CreateImage(cv.GetSize(img), img.depth, 1)
#            edge = cv.CreateImage(cv.GetSize(img), img.depth, 1)
#                        
#            cv.CvtColor(img, hsv, cv.CV_RGB2HSV)
#            cv.CvtColor(img, gray, cv.CV_RGB2GRAY)
#            cv.Split(hsv, None, s, None, None)
#            cv.ShowImage('i', img)
#            cv.ShowImage('s', s)
#            cv.ShowImage('g', gray)
#            cv.Threshold(gray, gray, 110, 255, cv.CV_THRESH_TOZERO_INV)
#            cv.ShowImage('gt', gray)
#            
#            cv.Threshold(s, s, 110, 255, cv.CV_THRESH_TOZERO)
#            cv.ShowImage('t', s)
##            cv.Smooth(s, s, cv.CV_GAUSSIAN, 7, 7)
#            cv.Canny(s, edge, 0, 0, 3)
#            cv.ShowImage('e', edge)
#
#            while True:
#                key = cv.WaitKey()
#                if key == 65361:
#                    if (index != 0):
#                        index -= 1
#                elif key == 65363:
#                    if (index < files.__len__() - 1):
#                        index += 1                         
#                else:
#                    exit(0)
#                break
