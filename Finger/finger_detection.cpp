#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/video/background_segm.hpp>
#include <iostream>
#include <fstream>
#include <algorithm>
#include "preprocessing.h"

using namespace std;
using namespace cv;


bool touch(Vec3f circle, Point p, float epsilon)
{
	float distance = sqrt((circle[0] - p.x) * (circle[0] - p.x) + (circle[1] - p.y) * (circle[1] - p.y));
	return abs(distance - circle[2]) < epsilon;
}



void maxAreaConvexHull(Mat& frame, vector<vector<Point>>& contours, vector<Point>& handContour, vector<Point>& handConvexHull, bool show)
{
	if(contours.empty())
		return;

	int m = frame.rows, n = frame.cols;
	vector<vector<Point>> hull(contours.size());
	int max_area = -1, max_index = -1;

	for (int i = 0; i < contours.size(); ++i)
		cv::convexHull(Mat(contours[i]), hull[i], false);

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
	handContour = contours[max_index];
	
	if (show)
	{
		Moments mu = moments(contours[max_index], false);

		// get the centroid of figures.
		Point2f mc = Point2f(mu.m10 / mu.m00, mu.m01 / mu.m00);


		drawContours(frame, hull, max_index, Scalar(0, 0, 128), 1, 8, vector<Vec4i>(), 0, Point());
		drawContours(frame, contours, max_index, Scalar(128, 0, 128), 1, 8, vector<Vec4i>(), 0, Point());

		for (int k = 0; k < hull[max_index].size(); ++k)
			circle(frame, hull[max_index][k], 4, Scalar(0, 0, 0), -1, 8, 0);
		circle(frame, mc, 4, Scalar(0, 128, 0), -1, 8, 0);

	}
}

struct Dist
{
	int distance;
	Dist(int input) : distance(input) {};
	bool operator () (const Point p, const Point q)
	{
		return (norm(p - q) <= distance);
	}
};



void detectFingers(Image<Vec3b>& frame, vector<Point>& handContour, vector<Point>& handConvexHull)
{
	Moments mu = moments(handConvexHull, false);

	// get the centroid of figures.
	Point2f mc = Point2f(mu.m10 / mu.m00, mu.m01 / mu.m00);
	
	vector<Vec4i> defects;
	vector<int> labels, intHull;
	vector<Point> defectPoints;

	partition(handConvexHull, labels, Dist(15));

	
	auto max = max_element(begin(labels), end(labels));
	if (!labels.empty() && !handConvexHull.empty() && !handContour.empty())
	{
		
		vector<Point> avgPoints(*max + 1, Point(0,0)), filteredConvextPoints;
		vector<int> sizeLabels(*max + 1, 0);
		
		for (int i = 0; i < labels.size(); i++) {
			avgPoints[labels[i]] += handConvexHull[i];
			sizeLabels[labels[i]] += 1;
		}
		
		for (int i = 0; i < avgPoints.size(); i++) {
			avgPoints[i] /= sizeLabels[i];
		}

		for (int i = 0; i < avgPoints.size(); i++) {
			if (avgPoints[i].y < mc.y)
				filteredConvextPoints.push_back(avgPoints[i]);
		}


		for (int k = 0; k < avgPoints.size(); ++k)
		{
			circle(frame, filteredConvextPoints[k], 4, Scalar(0, 255, 0), -1, 8, 0);
		}
			
		cv::convexHull(handContour, intHull, false, false);

		cv::convexityDefects(handContour, intHull, defects);

		for (int i = 0; i < defects.size(); i++)
		{
			//Get only distant defects (area between fingers) and above centroid
			if((defects[i].val[3] > 250) && (handContour[defects[i].val[2]].y < mc.y))
				defectPoints.push_back(handContour[defects[i].val[2]]);
		}
		
		for (int k = 0; k < defectPoints.size(); ++k)
			circle(frame, defectPoints[k], 4, Scalar(0, 0, 255), -1, 8, 0);
	}
	
}


int main() {
	VideoCapture capture;

	if (!capture.open(0))
		return 0;

	PreProcessing preProcessing;
	Image<Vec3b> frame;

	while(true)
	{
		capture >> frame;
		if (frame.empty())
			break;
		auto conFrame = frame.clone();
		Image<Vec3b> pointed = (Image<Vec3b>)frame.clone();
		vector<Point> handConvexHull;
		vector<Point> handContour;

		preProcessing.setCurrentFrame(frame);
		
		// frame differencing
		preProcessing.frameDifferencingAvgRun(170, 40, true);
		// preProcessing.frameDifferencingBgSb(10, true);
		
		Image<uchar> colorFiltered;
		preProcessing.filterSkinColor(preProcessing.getFilteredByDifference(), colorFiltered);

		preProcessing.applyCanny(colorFiltered, 50, 100);
		preProcessing.addContours();
		
		maxAreaConvexHull(conFrame, preProcessing.getFilteredContours(), handContour, handConvexHull, true);
		
		//detectFingers(pointed, handContour, handConvexHull);

		imshow("Convex Hull", conFrame);
		//imshow("Pointed", pointed);

		handConvexHull.clear();
		if (waitKey(10) == 27) break; // stop capturing by pressing ESC 
	}
	
	return 0;
}
