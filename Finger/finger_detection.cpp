#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <iostream>
#include <fstream>
#include "image.h"
#include "preprocessing.h"

using namespace std;
using namespace cv;


void fillCountours(Mat& frame, uchar threshold) {
	bool fill = false;

	// from left to right
	Mat rightToLeft = frame.clone();
	for(int i = 0; i < frame.rows; i++) {
		for(int j = 0; j < frame.cols; j++) {
			if (!fill && rightToLeft.at<uchar>(i, j) > threshold)
				fill = true;
			else if (fill && rightToLeft.at<uchar>(i, j) > threshold)
				fill = false;
			else if (fill)
				rightToLeft.at<uchar>(i, j) = rightToLeft.at<uchar>(i, j-1);
		}

		fill = false;
	}

	// from right to left
	fill = false;
	Mat leftToRight = frame.clone();
	for(int i = 0; i < frame.rows; i++) {
		for(int j = frame.cols-1; j >= 0; j--) {
			if (!fill && leftToRight.at<uchar>(i, j) > threshold)
				fill = true;
			else if (fill && leftToRight.at<uchar>(i, j) > threshold)
				fill = false;
			else if (fill)
				leftToRight.at<uchar>(i, j) = leftToRight.at<uchar>(i, j+1);
		}
		fill = false;
	}

	// intersection ?
	for(int i = 0; i < frame.rows; i++) {
		for(int j = 0; j < frame.cols; j++) {
			if (rightToLeft.at<uchar>(i, j) > threshold && leftToRight.at<uchar>(i, j) > threshold)
				frame.at<uchar>(i, j) = 255;
		}
	}
}

// Computer the difference between the current frame and previous frame
// Suppose that the frames are in CV_8U (COLOR_BGR2GRAY)
Mat frameDifferencing(Mat currFrame, Mat prevFrame, uchar threshold, bool fill, bool show) {
	Mat diff;
	absdiff(prevFrame, currFrame, diff);
	Mat foregroundMask = Mat::zeros(diff.rows, diff.cols, CV_8U);

	for(int i = 0; i < diff.rows; i++) {
		for(int j = 0; j < diff.cols; j++) {
			uchar pix = diff.at<uchar>(i, j);

			if(pix>threshold)
				foregroundMask.at<uchar>(i, j) = 255;
		}
	}

	// Blur ?
	// GaussianBlur(diff, diff, Size(9, 9), 30, 30);
	// GaussianBlur(foregroundMask, foregroundMask, Size(9, 9), 30, 30);

	// Dilate ?
	// cv::dilate(diff, diff, cv::Mat(), cv::Point(-1, -1));
	cv::dilate(foregroundMask, foregroundMask, cv::Mat(), cv::Point(-1, -1), 2);


	if (fill) {
		fillCountours(diff, threshold);
		fillCountours(foregroundMask, threshold);
	}
	
	if (show) {
		imshow("diff", diff);
		imshow("diff with threshold", foregroundMask);
	}

	return foregroundMask;
}

// Count the number of pixels that are not black
int evaluateMovement(Mat frame) {
	int count = 0;
	for(int i = 0; i < frame.rows; i++) {
		for(int j = 0; j < frame.cols; j++) {
			if (frame.at<uchar>(i, j) > 0)
				count++;
		}
	}

	return count;
}

bool findLargestIntersectionContour(const Mat& diff, const vector<vector<Point>>& contours, vector<Point>& largestIntersection) {
	int max_index = -1, max_area = 0;
	vector<vector<Point>> intersections;

	for (const vector<Point>& contour : contours) {
		for (const Point& p : contour) {
			if (diff.at<uchar>(p.y, p.x) > 0) {
				intersections.push_back(contour);
				break;
			}
		}
	}

	for (int i = 0; i < intersections.size(); i++) {
		double area = contourArea(intersections[i], false);
		if (area > max_area) {
			max_area = area;
			max_index = i;
		}
	}

	if (max_index >= 0) {
		largestIntersection = intersections[max_index];
		return true;
	}

	return false;
}

bool touch(Vec3f circle, Point p, float epsilon)
{
	float distance = sqrt((circle[0] - p.x) * (circle[0] - p.x) + (circle[1] - p.y) * (circle[1] - p.y));
	return abs(distance - circle[2]) < epsilon;
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
		int m = frame.rows, n = frame.cols;
		
		preProcessing.setCurrentFrame(frame);
		preProcessing.frameDifferencing(50, true);
		preProcessing.applyGaussianBlur(preProcessing.getCurrentFrame().greyImage(), Size(9, 9), 0.1, 0.1);
		preProcessing.applyCanny(preProcessing.getCurrentFrame().greyImage(), 20, 100);
		preProcessing.applyDilate(preProcessing.getCanny(), 1);
		imshow("canny", preProcessing.getCanny());
		if (preProcessing.findLargestIntersectionContour()) {
			imshow("intersection frame", preProcessing.getIntersectionFrame());
		}

		// vector<vector<Point>>& filtered = preProcessing.getFilteredContours();

		// if (!filtered.size() || !filtered[0].size())
		// 	continue;


		// Moments mu = moments(filtered[0], false);

		// // // get the centroid of figures.
		// Point2f mc = Point2f(mu.m10 / mu.m00, mu.m01 / mu.m00);
		// vector<vector<Point>> hull(1);
		// convexHull(Mat(filtered[0]), hull[0], false);

		// Mat circlesFrame = frame.clone();
		// drawContours(circlesFrame, hull, 0, Scalar(0, 0, 128), 1, 8, vector<Vec4i>(), 0, Point());
		// for (int k = 0; k < hull[0].size(); ++k)
		// 	circle(circlesFrame, hull[0][k], 4, Scalar(0, 0, 0), -1, 8, 0);
		// circle(circlesFrame, mc, 4, Scalar(0, 128, 0), -1, 8, 0);
		// drawContours(circlesFrame, filtered, 0, Scalar(128, 0, 0), 2);
	

		// Mat single_contour(m, n, CV_8U, Scalar(0, 0, 0));
		
		// for (int i = 0; i < filtered[0].size(); ++i)
		// 	single_contour.at<uchar>(filtered[0][i].y, filtered[0][i].x) = 255;
	

		// cv::dilate(single_contour, single_contour, cv::Mat(), cv::Point(-1, -1));

		// vector<Vec3f> circles, circles_fil;
		// HoughCircles(single_contour, circles, HOUGH_GRADIENT,
		// 	2.7, 30, 150, 50, 0, 30);	

		// //Filter by circles who touch convexHull
		// for (size_t i = 0; i < circles.size(); ++i)
		// 	for (size_t j = 0; j < hull[0].size(); ++j)
		// 		if (touch(circles[i], hull[0][j], 5))
		// 			circles_fil.push_back(circles[i]);
				

		// for (size_t i = 0; i < circles_fil.size(); ++i)
		// {
		// 	Point center(cvRound(circles_fil[i][0]), cvRound(circles_fil[i][1]));
		// 	int radius = cvRound(circles_fil[i][2]);
		// 	// draw the circle center
		// 	circle(circlesFrame, center, 3, Scalar(255, 255, 255), -1, 8, 0);
		// 	// draw the circle outline
		// 	circle(circlesFrame, center, radius, Scalar(255, 255, 255), 3, 8, 0);
		// }

		// imshow("blank", single_contour);
		// imshow("Finger Detection", circlesFrame);
		
		if (waitKey(10) == 27) break; // stop capturing by pressing ESC 
	}
	
	return 0;
}
