/*
 ============================================================================
 Name        : Moment.c
 Author      : Anol
 Version     :
 Copyright   : Your copyright notice
 Description : Hello World in C, Ansi-style
 ============================================================================
 */

#include <stdio.h>
#include <stdlib.h>

#include <opencv/cxcore.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv/cxtypes.h>

#define WIDTH 101
#define HEIGTH 101

#define WHITE cvScalar(0xff, 0xff, 0xff, 0)
#define BLACK cvScalar(0, 0, 0, 0)

int main(void) {

	//	/*	Rectangular	*/
	CvSize size = cvSize(WIDTH, HEIGTH);
	//	IplImage * rect = cvCreateImage(size, IPL_DEPTH_8U, 1);
	//	CvRect rect_size = cvRect(50, 50, 400, 400);
	//	CvScalar rect_clr = cvScalar(0xff, 0xff, 0, 0);
	//		cvRectangleR(rect, rect_size, rect_clr, 1, 8, 0);

	/*	Ellipse	*/
	IplImage * ellip = cvCreateImage(size, IPL_DEPTH_8U, 1);
	CvPoint2D32f ellip_center = cvPoint2D32f((WIDTH * 0.5) + 1,
			(HEIGTH * 0.5) + 1);
	CvSize2D32f ellip_size = cvSize2D32f(80, 50);

	/*	Triangle	*/

	//	IplImage *img = cvCloneImage(ellip);
	//	cvZero(img);

	for (float deg = 0; deg <= 180; deg += 0.1) {
		cvZero(ellip);
		CvBox2D ellip_box = { ellip_center, ellip_size, (float) deg };
		cvEllipseBox(ellip, ellip_box, WHITE, 1, 8, 0);
		IplImage *img = cvCloneImage(ellip);
		cvZero(img);

		CvMemStorage* storage = cvCreateMemStorage(0);
		CvSeq* contour = NULL;

		cvFindContours(ellip, storage, &contour, sizeof(CvContour),
				CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE, cvPoint(0, 0));

		//		for (CvSeq* c = contour; c != NULL; c = c->h_next) {

		cvDrawContours(img, contour, WHITE, BLACK, -1, CV_FILLED, 8,
				cvPoint(0, 0));
		//		}

		CvMoments moment;
		CvHuMoments hu;
		cvMoments(img, &moment, 1);
		cvGetHuMoments(&moment, &hu);

		//		cvShowImage("Rectangular", rect);
		printf(
				"%3.2f : %+8.3E %+8.3E %+8.3E %+8.3E %+8.3E %+8.3E %+8.3E\n",
				deg, hu.hu1, hu.hu2, hu.hu3, hu.hu4, hu.hu5, hu.hu6, hu.hu7);

		cvShowImage("Ellipse", ellip);
		cvShowImage("Ellipse Contour", img);

		int ret = cvWaitKey(10);
		if (ret == 27)
			break;
		//		if (cvWaitKey(0) == 27)
		//			;
		//		break;
		//		cvWaitKey(0);
	}

	cvDestroyAllWindows();
	puts("!!!Application Closed!!!"); /* prints !!!Hello World!!! */
	return EXIT_SUCCESS;
}
