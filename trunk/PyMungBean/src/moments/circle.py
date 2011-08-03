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

def Circle(img):
    circ = cv.CloneImage(img)
    center = (cv.Round(WIDTH * 0.5) + 1, cv.Round(HEIGTH * 0.5) + 1)
    radius = 40
    cv.Zero(circ)
    cv.Circle(circ, center, radius, WHITE)
    storage = cv.CreateMemStorage()
    seq = cv.FindContours(circ, storage, cv.CV_RETR_TREE, cv.CV_CHAIN_APPROX_SIMPLE)
    cv.DrawContours(img, seq, WHITE, BLACK, -1, cv.CV_FILLED, 8)
    return img

if __name__ == '__main__':
    pass
