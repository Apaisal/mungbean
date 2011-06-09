'''
Created on May 30, 2011

@author: anol
'''
import cv

if __name__ == '__main__':
    imgfile = '../dataset/multiple/Picture23.jpg'
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

    gs = cv.CloneImage(s_plane)
    cv.Zero(gs)

    cv.Smooth(s_plane, s_plane, cv.CV_GAUSSIAN, 9, 9)

    cv.ShowImage("median", s_plane)

    cv.Add(s_plane, s_plane, gs)

    cv.Threshold(s_plane, gs, 150, 255, cv.CV_THRESH_BINARY)

    seq = cv.FindContours(gs, cv.CreateMemStorage(), cv.CV_RETR_LIST, cv.CV_CHAIN_APPROX_NONE, (0, 0))

    ext_color = cv.CV_RGB(255, 255, 255)
    hole_color = cv.CV_RGB(0, 0, 0)
    fore_img = cv.CloneImage(gs)
    cv.Zero(fore_img)
    while seq:
        cv.DrawContours(fore_img, seq, ext_color, hole_color, 0, thickness = -1, lineType = 8, offset = (0, 0))
        seq = seq.h_next()

    cv.ShowImage("foreground", fore_img)
    cv.WaitKey()


    seq = cv.FindContours(fore_img, cv.CreateMemStorage(), cv.CV_RETR_LIST, cv.CV_CHAIN_APPROX_NONE, (0, 0))
    while seq:
        cv.Zero(dst)
        cv.DrawContours(dst, seq, ext_color, hole_color, 0, thickness = -1, lineType = 8, offset = (0, 0))
        seed = cv.CloneImage(img)
        cv.And(seed, dst, seed)
        cv.ShowImage("dst", seed)
        cv.WaitKey()
        seq = seq.h_next()
