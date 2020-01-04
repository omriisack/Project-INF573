#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/video/background_segm.hpp>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <tuple>

#include "image.h"

double getAngle(tuple<Point, Point>& l1, tuple<Point, Point>& l2);

void getLinesBetweenPoints(Mat& frame, vector<Point>& v1, vector<Point>& v2, vector<tuple<Point, Point>>& lines);

void createFilteredPoints(vector<Point>& handContour, vector<Point>& handConvexHull, Point2f& mc, vector<Point>& finalConvexPoints, vector<Point>& finalDefects);

bool detectHand(Image<Vec3b>& frame, vector<Point>& handContour, vector<Point>& handConvexHull, Point2f& mc, bool show)

bool findHandContour(Image<Vec3b>& frame, vector<vector<Point>>& contours, bool show);
