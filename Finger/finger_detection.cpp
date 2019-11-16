#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/video/background_segm.hpp>
#include <iostream>
#include <fstream>
#include "image.h"
#include "preprocessing.h"
#include "finger_detection.h"

using namespace std;
using namespace cv;


bool touch(Vec3f circle, Point p, float epsilon)
{
	float distance = sqrt((circle[0] - p.x) * (circle[0] - p.x) + (circle[1] - p.y) * (circle[1] - p.y));
	return abs(distance - circle[2]) < epsilon;
}


void maxAreaConvexHull(Mat& frame, vector<vector<Point>>& contours, vector<Point>& handConvexHull, bool show)
{
	int m = frame.rows, n = frame.cols;
	vector<vector<Point>> hull(contours.size());
	int max_area = -1, max_index = -1;

	for (int i = 0; i < contours.size(); ++i)
		convexHull(Mat(contours[i]), hull[i], false);

	for (int i = 0; i < hull.size(); i++) {
		double area = contourArea(hull[i], false);
		if (area > max_area) {
			max_area = area;
			max_index = i;
		}
	}

	if (max_index < 0)
		return;

	handConvexHull = hull[max_index];
	
	if (show)
	{
		Moments mu = moments(contours[max_index], false);

		// get the centroid of figures.
		Point2f mc = Point2f(mu.m10 / mu.m00, mu.m01 / mu.m00);


		drawContours(frame, hull, max_index, Scalar(0, 0, 128), 1, 8, vector<Vec4i>(), 0, Point());
		for (int k = 0; k < hull[max_index].size(); ++k)
			circle(frame, hull[max_index][k], 4, Scalar(0, 0, 0), -1, 8, 0);
		circle(frame, mc, 4, Scalar(0, 128, 0), -1, 8, 0);

	}
}
struct Dist
{
	bool operator () (const Point p, const Point q)
	{
		return (norm(p - q) <= 100);
	}
};

bool closeToVector(Point& p, vector<Point>& vec, int dist)
{
	for (Point q : vec)
	{
		if (norm(p - q) < dist)
			return true;
	}
	return false;
}

void detectFingers(Image<Vec3b>& frame, vector<Point>& handConvexHull)
{

	Moments mu = moments(handConvexHull, false);

	// get the centroid of figures.
	Point2f mc = Point2f(mu.m10 / mu.m00, mu.m01 / mu.m00);
	
	vector<Point> filtered;
	vector<int> labels;

	for (Point p : handConvexHull)
	{
		if (p.y < mc.y && !closeToVector(p, filtered, 30))
		{
			filtered.push_back(p);
		}
	}


	for (int k = 0; k < filtered.size(); ++k)
		circle(frame, filtered[k], 4, Scalar(0, 255, 0), -1, 8, 0);

	//partition(handConvexHull, labels, Dist());



	/*
	vector<Vec3f> circles, circles_fil;
	HoughCircles(handConvexHull, circles, HOUGH_GRADIENT,
		2.7, 30, 150, 50, 0, 30);


	//Filter by circles who touch convexHull
	for (size_t i = 0; i < circles.size(); ++i)
		for (size_t j = 0; j < handConvexHull.size(); ++j)
			if (touch(circles[i],handConvexHull[j], 5))
				circles_fil.push_back(circles[i]);


	for (size_t i = 0; i < circles_fil.size(); ++i)
	{
		Point center(cvRound(circles_fil[i][0]), cvRound(circles_fil[i][1]));
		int radius = cvRound(circles_fil[i][2]);
		// draw the circle center
		circle(frame, center, 3, Scalar(255, 255, 255), -1, 8, 0);
		// draw the circle outline
		circle(frame, center, radius, Scalar(255, 255, 255), 3, 8, 0);
	}
	*/
	
}


int main() {
	VideoCapture capture;

	if (!capture.open(0))
		return 0;

	int frameJump = 2;
	PreProcessing preProcessing(frameJump);
	Image<Vec3b> frame;


	while(true)
	{
		capture >> frame;
		if (frame.empty())
			break;
		auto conFrame = frame.clone();

		vector<Point> handConvexHull;
		preProcessing.setCurrentFrame(frame);
		preProcessing.frameDifferencing(20, false);
		preProcessing.fillHorizontalGaps(preProcessing.getDifference(), 10);
		preProcessing.fillVerticalGaps(preProcessing.getDifference(), 10);
		preProcessing.applyCanny(preProcessing.getDifference(), 50, 100);
		preProcessing.addContours();

		maxAreaConvexHull(conFrame, preProcessing.getFilteredContours(), handConvexHull, true);
		detectFingers(frame ,handConvexHull);

		imshow("filled", preProcessing.getDifference());
		imshow("Contours", preProcessing.getIntersectionFrame());
		imshow("Convex Hull", conFrame);
		imshow("Frame", frame);

		handConvexHull.clear();
		if (waitKey(10) == 27) break; // stop capturing by pressing ESC 
	}
	
	return 0;
}
