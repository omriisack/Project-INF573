#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/video/background_segm.hpp>
#include <iostream>
#include <fstream>
#include <algorithm>
#include "preprocessing.h"

using namespace std;
using namespace cv;

struct Dist
{
	int distance;
	Dist(int input) : distance(input) {};
	bool operator () (const Point p, const Point q)
	{
		return (norm(p - q) <= distance);
	}
};


Point& getClosestFromSide(Point& refPoint, vector<Point>& candidates, bool fromLeft)
{
	if (candidates.empty())
		return Point(-1, -1); //Dummy value

	int bestIndex = -1, minDistance = 10000, distance;
	for (int i = 0; i < candidates.size(); i++)
	{
		//If looking for closest from left, then  refPoint.x > candidates.x
		distance = fromLeft ? refPoint.x - candidates[i].x : candidates[i].x - refPoint.x;
		if (distance <= minDistance && distance >= 0)
		{
			minDistance = distance;
			bestIndex = i;
		}
	}

	if (bestIndex == -1) //No point found from any candidates
		return Point(-1, -1);

	return candidates[bestIndex];
}


void maxAreaConvexHull(Mat& frame, vector<vector<Point>>& contours, vector<Point>& handContour, vector<Point>& handConvexHull, bool show)
{
	if(contours.empty())
		return;

	int m = frame.rows, n = frame.cols;
	vector<vector<Point>> hull(contours.size());
	int max_area = -1, max_index = -1;

	for (int i = 0; i < contours.size(); i++)
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


void detectFingers(Image<Vec3b>& frame, vector<Point>& handContour, vector<Point>& handConvexHull)
{
	if (handContour.empty() || handConvexHull.empty())
		return;

	Moments mu = moments(handConvexHull, false);

	// get the centroid of figures.
	Point2f mc = Point2f(mu.m10 / mu.m00, mu.m01 / mu.m00);
	
	vector<Vec4i> defects;
	vector<int> labels, intHull;
	vector<Point> finalDefects, finalConvexPoints;
	Point tempL, tempR;

	partition(handConvexHull, labels, Dist(15));
	
	// Calculate avg points of each label
	auto max = max_element(begin(labels), end(labels));
	vector<Point> avgPoints(*max + 1, Point(0,0));
	vector<int> sizeLabels(*max + 1, 0);
	for (int i = 0; i < labels.size(); i++) {
		avgPoints[labels[i]] += handConvexHull[i];
		sizeLabels[labels[i]] += 1;
	}
	for (int i = 0; i < avgPoints.size(); i++) {
		avgPoints[i] /= sizeLabels[i];
	}
	// Filter points under the centroid

	for (int i = 0; i < avgPoints.size(); i++) {
		if (avgPoints[i].y < mc.y)
			finalConvexPoints.push_back(avgPoints[i]);
	}
	
	// Find defect points
	cv::convexHull(handContour, intHull, false, false);
	cv::convexityDefects(handContour, intHull, defects);
	for (int i = 0; i < defects.size(); i++) {
		//Get only distant defects (area between fingers) and above centroid
		if((defects[i].val[3] > 250) && (handContour[defects[i].val[2]].y < mc.y))
			finalDefects.push_back(handContour[defects[i].val[2]]);
	}

	//Draw lines from every defects to its closest convex points from both sides
	for (int i = 0; i < finalConvexPoints.size(); i++)
	{
		tempR = getClosestFromSide(finalConvexPoints[i], finalDefects, false);
		tempL = getClosestFromSide(finalConvexPoints[i], finalDefects, true);
		if (tempR.x > -1)
			line(frame, finalConvexPoints[i], tempR, Scalar(255, 255, 0), 1);
		if (tempL.x > -1)
			line(frame, finalConvexPoints[i], tempL, Scalar(255, 255, 0), 1);
	}


	// Draw circle for the points found
	for (int i = 0; i < finalConvexPoints.size(); i++)
		circle(frame, finalConvexPoints[i], 4, Scalar(0, 255, 0), -1, 8, 0);

	for (int i = 0; i < finalDefects.size(); i++)
		circle(frame, finalDefects[i], 4, Scalar(0, 0, 255), -1, 8, 0);
	
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
		preProcessing.frameDifferencingAvgRun(15, 5, false, true);
		// mask
		preProcessing.filterByMask(preProcessing.getDifference(), true);
		// filter the skin color
		preProcessing.filterSkinColor(preProcessing.getFilteredByMask(), true);
		// canny
		preProcessing.applyCanny(preProcessing.getFilteredByMask(), 50, 110);
		// convex hull
		maxAreaConvexHull(conFrame, preProcessing.getContours(), handContour, handConvexHull, true);
		detectFingers(pointed, handContour, handConvexHull);

		imshow("Convex Hull", conFrame);
		imshow("Pointed", pointed);

		if (waitKey(10) == 27) break; // stop capturing by pressing ESC 
	}
	
	return 0;
}
