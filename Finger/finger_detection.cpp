#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/video/background_segm.hpp>
#include <iostream>
#include <fstream>
#include "image.h"
#include "preprocessing.h"

using namespace std;
using namespace cv;


bool touch(Vec3f circle, Point p, float epsilon)
{
	float distance = sqrt((circle[0] - p.x) * (circle[0] - p.x) + (circle[1] - p.y) * (circle[1] - p.y));
	return abs(distance - circle[2]) < epsilon;
}


void convexHulls(Mat& frame, vector<vector<Point>>& contours)
{
	int m = frame.rows, n = frame.cols;
	vector<vector<Point>> hull(contours.size());


	for (int i = 0; i < contours.size(); ++i)
	{
		Moments mu = moments(contours[i], false);

		// get the centroid of figures.
		Point2f mc = Point2f(mu.m10 / mu.m00, mu.m01 / mu.m00);
		convexHull(Mat(contours[i]), hull[i], false);

		drawContours(frame, hull, i, Scalar(0, 0, 128), 1, 8, vector<Vec4i>(), 0, Point());
		//for (int k = 0; k < hull[i].size(); ++k)
		//	circle(frame, hull[i][k], 4, Scalar(0, 0, 0), -1, 8, 0);
		//circle(frame, mc, 4, Scalar(0, 128, 0), -1, 8, 0);
		drawContours(frame, contours, 0, Scalar(128, 0, 0), 2);
	}
	

	/*
	Mat single_contour(m, n, CV_8U, Scalar(0, 0, 0));

	for (int i = 0; i < contours[0].size(); ++i)
		single_contour.at<uchar>(contours[0][i].y, contours[0][i].x) = 255;


	cv::dilate(single_contour, single_contour, cv::Mat(), cv::Point(-1, -1));

	vector<Vec3f> circles, circles_fil;
	HoughCircles(single_contour, circles, HOUGH_GRADIENT,
		2.7, 30, 150, 50, 0, 30);

	//Filter by circles who touch convexHull
	for (size_t i = 0; i < circles.size(); ++i)
		for (size_t j = 0; j < hull[0].size(); ++j)
			if (touch(circles[i], hull[0][j], 5))
				circles_fil.push_back(circles[i]);


	for (size_t i = 0; i < circles_fil.size(); ++i)
	{
		Point center(cvRound(circles_fil[i][0]), cvRound(circles_fil[i][1]));
		int radius = cvRound(circles_fil[i][2]);
		// draw the circle center
		circle(circlesFrame, center, 3, Scalar(255, 255, 255), -1, 8, 0);
		// draw the circle outline
		circle(circlesFrame, center, radius, Scalar(255, 255, 255), 3, 8, 0);
	}

	frame = circlesFrame;
	*/
}


int main() {
	VideoCapture capture;

	if (!capture.open(0))
		return 0;

	int frameJump = 2;
	PreProcessing preProcessing(frameJump);
	Image<Vec3b> frame;
	Mat fgMask, copy;

	while(true)
	{
		capture >> frame;
		if (frame.empty())
			break;

		preProcessing.setCurrentFrame(frame);
		preProcessing.frameDifferencing(20, false);
		preProcessing.fillGaps(preProcessing.getDifference(), 10);
		preProcessing.applyCanny(preProcessing.getDifference(), 50, 100);
		preProcessing.addContours();
		convexHulls(frame, preProcessing.getFilteredContours());

		imshow("filled", preProcessing.getDifference());
		imshow("Contours", preProcessing.getIntersectionFrame());
		imshow("Frame", frame);

		if (waitKey(10) == 27) break; // stop capturing by pressing ESC 
	}
	
	return 0;
}
