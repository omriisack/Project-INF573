#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <iostream>
#include <fstream>
#include "image.h"


using namespace std;
using namespace cv;


bool touch(Vec3f circle, Point p, float epsilon)
{
	float distance = sqrt((circle[0] - p.x) * (circle[0] - p.x) + (circle[1] - p.y) * (circle[1] - p.y));
	return abs(distance - circle[2]) < epsilon;
}

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
		GaussianBlur(grey, grey, Size(9, 9), 0.1, 0.1);

		Canny(grey, can, 20, 100);
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
			circle(Frame, hull[0][k], 4, Scalar(0, 0, 0), -1, 8, 0);
		circle(Frame, mc, 4, Scalar(0, 128, 0), -1, 8, 0);
		drawContours(Frame, filtered, 0, Scalar(128, 0, 0), 2);
	

		Mat single_contour(m, n, CV_8U, Scalar(0, 0, 0));
		
		for (int i = 0; i < filtered[0].size(); ++i)
			single_contour.at<uchar>(filtered[0][i].y, filtered[0][i].x) = 255;
	

		cv::dilate(single_contour, single_contour, cv::Mat(), cv::Point(-1, -1));

		vector<Vec3f> circles, circles_fil;
		HoughCircles(single_contour, circles, CV_HOUGH_GRADIENT,
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
			circle(Frame, center, 3, Scalar(255, 255, 255), -1, 8, 0);
			// draw the circle outline
			circle(Frame, center, radius, Scalar(255, 255, 255), 3, 8, 0);
		}

		//imshow("blank", single_contour);
		imshow("Finger Detection", Frame);
		
		if (waitKey(10) == 27) break; // stop capturing by pressing ESC 
	}
	
	return 0;
}
