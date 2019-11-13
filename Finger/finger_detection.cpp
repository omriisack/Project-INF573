#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
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


void detectFingers(Mat& frame, vector<vector<Point>>& handContour)
{
	int m = frame.rows, n = frame.cols;

	Moments mu = moments(handContour[0], false);

	// get the centroid of figures.
	Point2f mc = Point2f(mu.m10 / mu.m00, mu.m01 / mu.m00);
	vector<vector<Point>> hull(1);
	convexHull(Mat(handContour[0]), hull[0], false);

	Mat circlesFrame = frame.clone();
	drawContours(circlesFrame, hull, 0, Scalar(0, 0, 128), 1, 8, vector<Vec4i>(), 0, Point());
	for (int k = 0; k < hull[0].size(); ++k)
		circle(circlesFrame, hull[0][k], 4, Scalar(0, 0, 0), -1, 8, 0);
	circle(circlesFrame, mc, 4, Scalar(0, 128, 0), -1, 8, 0);
	drawContours(circlesFrame, handContour, 0, Scalar(128, 0, 0), 2);


	Mat single_contour(m, n, CV_8U, Scalar(0, 0, 0));

	for (int i = 0; i < handContour[0].size(); ++i)
		single_contour.at<uchar>(handContour[0][i].y, handContour[0][i].x) = 255;


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
}


int main() {
	VideoCapture cap;

	if (!cap.open(0))
		return 0;

	int frameJump = 2;
	PreProcessing preProcessing(frameJump);
	Image<Vec3b> frame;
	while(true)
	{
		cap >> frame;
		if (frame.empty()) break; // end of video stream
		
		preProcessing.setCurrentFrame(frame);
		preProcessing.frameDifferencing(50, true);
		preProcessing.applyGaussianBlur(preProcessing.getCurrentFrame().greyImage(), Size(9, 9), 0.1, 0.1);
		preProcessing.applyCanny(preProcessing.getCurrentFrame().greyImage(), 50, 100);
		preProcessing.applyDilate(preProcessing.getCanny(), 1);
		imshow("canny", preProcessing.getCanny());
		
		
		preProcessing.findLargestIntersectionContour();
		
		vector<vector<Point>>& filtered = preProcessing.getFilteredContours();

		if (!filtered.size() || !filtered[0].size())
			continue;

		detectFingers(frame, filtered);

		imshow("Finger Detection", frame);
		
		if (waitKey(10) == 27) break; // stop capturing by pressing ESC 
	}
	
	return 0;
}
