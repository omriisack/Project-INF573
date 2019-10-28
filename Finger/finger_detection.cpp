#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <iostream>
#include <fstream>
#include "image.h"


using namespace std;
using namespace cv;


int main() {
	VideoCapture cap;

	if (!cap.open(0))
		return 0;

	while(true)
	{
		Image<Vec3b> Frame;
		cap >> Frame;
		int m = Frame.rows, n = Frame.cols;
		Mat grey, can;
		vector<vector<Point> > contours;
		vector<vector<Point>> filtered(1);
		vector<Vec4i> hierarchy;

		if (Frame.empty()) break; // end of video stream
		cvtColor(Frame, grey, COLOR_BGR2GRAY);
		Canny(grey, can, 20, 150);
		cv::dilate(can, can, cv::Mat(), cv::Point(-1, -1));
		findContours(can, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_NONE);


		int max_index = -1;
		double biggest_area = 0.0;

		for (int i = 0; i < contours.size(); i++) {
			double area = contourArea(contours[i], false);
			if (area > biggest_area) {
				biggest_area = area;
				max_index = i;
			}
		}

		if (!contours.size())
			continue;
		
		for (int j = 0; j < contours[max_index].size(); ++j)
			if (contours[max_index][j].y < m * 0.8)
				filtered[0].push_back(contours[max_index][j]);
		
		if (!filtered.size() || !filtered[0].size())
			continue;


		Moments mu = moments(filtered[0], false);

		// get the centroid of figures.
		Point2f mc = Point2f(mu.m10 / mu.m00, mu.m01 / mu.m00);
		vector<vector<Point>> hull(1);
		convexHull(Mat(filtered[0]), hull[0], false);
		drawContours(Frame, hull, 0, Scalar(0, 0, 128), 1, 8, vector<Vec4i>(), 0, Point());
		for (int k = 0; k < hull[0].size(); ++k)
		{
			circle(Frame, hull[0][k], 4, Scalar(0, 0, 0), -1, 8, 0);
		}
		circle(Frame, mc, 4, Scalar(0, 128, 0), -1, 8, 0);
		drawContours(Frame, filtered, 0, Scalar(128, 0, 0), 2);
	
		
		imshow("Finger Detection", Frame);
		if (waitKey(10) == 27) break; // stop capturing by pressing ESC 
	}
	
	return 0;
}
