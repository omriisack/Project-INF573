#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/video/background_segm.hpp>
#include <iostream>
#include <fstream>
#include <algorithm>
#include "preprocessing.h"
#include <tuple>
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

bool isCloseToOthers(Point& p, vector<Point>& others, int distance)
{
	for (Point q : others)
		if (norm(p - q) < distance)
			return true;

	return false;
}

bool compareByX(Point& a, Point& b)
{
	return a.x < b.x; // value will be true if a is left of b
}

double getAngle(tuple<Point, Point>& l1, tuple<Point, Point>& l2)
{
	double dy1 = get<0>(l1).y - get<1>(l1).y;
	double dy2 = get<0>(l2).y - get<1>(l2).y;
	if (!dy1 || !dy2)
		return 0;

	double dx1 = get<0>(l1).x - get<1>(l1).x;
	double dx2 = get<0>(l2).x - get<1>(l2).x;

	double slope1 = dx1 / dy1, slope2 = dx2 / dy2;

	return abs(atan( (slope1 - slope2) / (1 + slope1*slope2) ));
}

void getLinesBetweenPoints(Mat& frame, vector<Point>& v1, vector<Point>& v2, vector<tuple<Point, Point>>& lines)
{
	if (v1.empty() || v2.empty())
		return;

	sort(v1.begin(), v1.end(),compareByX);
	sort(v2.begin(), v2.end(), compareByX);

	vector<Point> first = v1.size() > v2.size() ? v1 : v2;
	vector<Point> second = v1.size() > v2.size() ? v2 : v1;

	int count = min(first.size(), second.size()), i = 0;

	while (i < count)
	{
		lines.push_back(make_tuple(first[i], second[i]));
		if (i < first.size() - 1)
			lines.push_back(make_tuple(second[i], first[i+1]));
		++i;
	}
}


bool contourComparator(vector<Point>& c1, vector<Point>& c2)
{
	return contourArea(c1, false) >= contourArea(c2, false);
}


void createFilteredPoints(vector<Point>& handContour, vector<Point>& handConvexHull, Point2f& mc, vector<Point>& finalConvexPoints, vector<Point>& finalDefects)
{
	vector<Vec4i> defects;
	vector<int> labels, intHull;

	// get the centroid of figures.
	partition(handConvexHull, labels, Dist(15));

	// Calculate avg points of each label
	auto max = max_element(begin(labels), end(labels));
	vector<Point> avgPoints(*max + 1, Point(0, 0));
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
		Point& candidate = handContour[defects[i].val[2]];
		//Get only distant defects (area between fingers) and above centroid
		if ((defects[i].val[3] > 250) && (handContour[defects[i].val[2]].y - 10 < mc.y) && !isCloseToOthers(candidate, finalConvexPoints, 10))
			finalDefects.push_back(candidate);
	}


}

double avgY(vector<Point>& vec)
{
	if (!vec.size())
		return 0;

	double avg = 0;
	for (Point p : vec)
		avg += p.y;

	return avg / vec.size();
}

bool detectFingers(Image<Vec3b>& frame, vector<Point>& handContour, vector<Point>& handConvexHull, Point2f& mc, bool show)
{
	if (handContour.empty() || handConvexHull.empty() || contourArea(handConvexHull) < 8000)
		return false;
	vector<Point> finalDefects, finalConvexPoints;
	vector<tuple<Point, Point>> lines;
	double defectsYAvg, convexYAvg;

	createFilteredPoints(handContour, handConvexHull, mc, finalConvexPoints, finalDefects);
	getLinesBetweenPoints(frame, finalConvexPoints, finalDefects, lines);
	
	if (lines.empty() || finalConvexPoints.size() < 2 || finalConvexPoints.size() > 5 || finalDefects.size() > 6 || finalDefects.size() >= 2 * finalConvexPoints.size())
		return false;


	defectsYAvg = avgY(finalDefects);
	convexYAvg = avgY(finalConvexPoints);
	double defectsConvexYRelation = (convexYAvg - mc.y) / (defectsYAvg - mc.y);
	if (defectsConvexYRelation < 1 || defectsConvexYRelation > 2.5)
		return false;


	int badAngles = 0;
	for (int i = 0; i < lines.size() - 1; i++)
		if (getAngle(lines[i], lines[i+1]) > 1.22)
			badAngles++;


	if (badAngles > 1 || badAngles == 1 && lines.size() == 2 ) // Allow one large angle for the thumb, as long as its not the only one
		return false;


	// Draw circle for the points found
	for (int i = 0; i < finalConvexPoints.size(); i++)
		circle(frame, finalConvexPoints[i], 4, Scalar(0, 255, 0), -1, 8, 0);


	if (show)
	{
		for (int i = 0; i < finalDefects.size(); i++)
			circle(frame, finalDefects[i], 4, Scalar(0, 0, 255), -1, 8, 0);

		/*for (int i = 0; i < lines.size(); i++)
			line(frame, get<0>(lines[i]), get<1>(lines[i]), Scalar(255, 255, 0), 1);*/
	}
	

	return true;
}


bool iterateContours(Image<Vec3b>& frame, vector<vector<Point>>& contours, bool show)
{
	if (contours.empty())
		return false;
	vector<Point> convex;
	bool detected = false;

	for (int i = 0; i < contours.size(); i++)
	{
		convexHull(contours[i], convex);
		Moments mu = moments(convex, false);
		Point2f mc = Point2f(mu.m10 / mu.m00, mu.m01 / mu.m00);

		if (detectFingers(frame, contours[i], convex, mc, show))
		{
			detected = true;
			if (show)
				drawContours(frame, contours, i, Scalar(255, 255, 255), 1);
		}
			
	}
	return detected;
}


int main() {
	VideoCapture capture;

	if (!capture.open(0))
		return 0;

	PreProcessing preProcessing;
	Image<Vec3b> frame;
	bool detected = false;

	while(true)
	{
		capture >> frame;
		if (frame.empty())
			break;

		Image<Vec3b> result = (Image<Vec3b>)frame.clone();
		vector<Point> handConvexHull;
		vector<Point> handContour;

		preProcessing.setCurrentFrame(frame);

		// frame differencing
		preProcessing.frameDifferencingAvgRun(35, 15, detected, false, true);
		// mask
		preProcessing.filterByMask(preProcessing.getDifference(), false);

		// filter the skin color
		preProcessing.filterSkinColor(preProcessing.getFilteredByMask(), false);

		// canny
		preProcessing.applyCanny(preProcessing.getDifference(), 50, 20);
		
		detected = iterateContours(result, preProcessing.getContours(), true);
		imshow("Hand Detector", result);

		if (waitKey(10) == 27) break; // stop capturing by pressing ESC 
	}
	
	return 0;
}

