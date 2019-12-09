#include <queue>
#include "preprocessing.h"
#include <iostream>

using namespace std;
using namespace cv;

void PreProcessing::setCurrentFrame(Image<Vec3b>& frame) {
	currentFrame = frame;
}

void PreProcessing::frameDifferencingBgSb(uchar threshold, bool show) {
	Mat copy = currentFrame.clone();
	GaussianBlur(copy, copy, Size(9, 9), 30, 30);

	//update the background model
	bgs->apply(copy, difference);

	erode(difference, difference, cv::Mat(), cv::Point(-1, -1), 2);
	dilate(difference, difference, cv::Mat(), cv::Point(-1, -1), 2);
    frameThreshold(difference, threshold);
    
    if (show) {
        imshow("diff with threshold", difference);
    }
}

void PreProcessing::frameDifferencingAvgRun(uchar hight, uchar lowt, bool detected, bool labColor, bool show) {
	Mat copy = Image<Vec3b>(currentFrame.clone()), diff;

	// Calculate an "avg" frame, compute the differencing on it
	if (accumulatedFrame.empty()) {
		copy.convertTo(accumulatedFrame, CV_32F);
	}
	Image<Vec3b> resultAccumulatedFrame = Image<Vec3b>(Mat::zeros(currentFrame.rows, currentFrame.cols, CV_32FC3));
	convertScaleAbs(accumulatedFrame, resultAccumulatedFrame);

	if (labColor) {
		// Use Lab colors
		Mat LabResultAccumulatedFrame;
		Mat LabCopy;
		cvtColor(resultAccumulatedFrame, LabResultAccumulatedFrame, COLOR_BGR2Lab);
		cvtColor(copy, LabCopy, COLOR_BGR2Lab);
		absdiff(LabResultAccumulatedFrame, LabCopy, diff);
		difference = Image<uchar>(matNorm(diff));
	} else {
		absdiff(resultAccumulatedFrame, copy, diff);
		cvtColor(diff, difference, COLOR_BGR2GRAY);
	}

	cv::Scalar mean, stddev;
	cv::meanStdDev(difference, mean, stddev);
	// float a = evaluateMovement(resultAccumulatedFrame, copy);
	// accumulateWeighted(copy, accumulatedFrame, 0.005 * sqrt(a));

	cv::erode(difference, difference, cv::Mat(), cv::Point(-1, -1), 2);
	cv::dilate(difference, difference, cv::Mat(), cv::Point(-1, -1), 2);

	

	applyMeanDenoise(difference);

	if (show)
		imshow("difference", difference);

	// Threshold the difference
	Image<uchar> thresholded = Image<uchar>(Mat::zeros(difference.rows, difference.cols, CV_8U));
	frameThresholdSeeds(difference, thresholded, hight, lowt);
	difference = thresholded;


	filterByMask(difference, false);
	filterSkinColor(filteredByMask, false);
	Mat noSkinMovement;
	absdiff(difference, filteredByColor, noSkinMovement);
	
	double movement = evaluateMovementByColor();
	movement /= 255;
	if (movement*100 > 10 && detected)
		accumulateWeighted(copy, accumulatedFrame, (movement));
	else
		accumulateWeighted(copy, accumulatedFrame, 0.0005);

    if (show) {
		imshow("resultAccumulatedFrame", resultAccumulatedFrame);
        imshow("diff with threshold", thresholded);
    }
}

void PreProcessing::filterByMask(const Image<uchar>& mask, bool show) {
	filteredByMask = Image<Vec3b>(Mat::zeros(mask.rows, mask.cols, CV_8UC3));

	for (int i = 0; i < currentFrame.rows; ++i) 
		for (int j = 0; j < currentFrame.cols; ++j) 
			if (mask.at<uchar>(i, j)) 
				filteredByMask.at<Vec3b>(i, j) = currentFrame.at<Vec3b>(i, j);
	
	if (show)
		imshow("filter by mask original frame", filteredByMask);
}

void PreProcessing::filterSkinColor(const Image<Vec3b>& input, bool show) {
	Image<Vec3b> inputHLS;
	cvtColor(input, inputHLS, COLOR_BGR2HLS);
	Vec3b upper = Vec3b(40,  251,  251), lower = Vec3b(3, 0.10 * 255, 0.10 * 255);
	inRange(inputHLS, lower, upper, filteredByColor);
	/*erode(filteredByColor, filteredByColor, cv::Mat(), cv::Point(-1, -1), 1);
	dilate(filteredByColor, filteredByColor, cv::Mat(), cv::Point(-1, -1), 1);*/
	if (show)
		imshow("filter by color", filteredByColor);
}

void PreProcessing::frameThreshold(Mat& frame, uchar threshold) {
	for(int i = 0; i < frame.rows; i++) {
        for(int j = 0; j < frame.cols; j++) {
            auto pix = frame.at<uchar>(i, j);
            if(pix > threshold)
				frame.at<uchar>(i, j) = 255;
			else
				frame.at<uchar>(i, j) = 0;
        }
    }
}

void PreProcessing::frameThresholdSeeds(const Image<uchar>& frame, Image<uchar>& res, int hight, int lowt) {
	queue<Point> Q;

	// Find seeds
	for (int i = 0; i < frame.rows; i++) {
		for (int j = 0; j < frame.cols; j++) {
			if (frame.at<uchar>(i, j) >= hight) {
				Q.push(Point(j, i));
			}
		}
	}

	// Propagate seeds
	res = Image<uchar>(Mat::zeros(frame.rows, frame.cols, CV_8U)) ;
	while (!Q.empty()) {
		int i = Q.front().y, j = Q.front().x;
		Q.pop();

		// Check if already visited this pixel
		if (res.at<uchar>(i, j) == 255)
			continue;

		res.at<uchar>(i, j) = 255;
		
		// Visit pixels around
		for (int k = -1; k < 2; k++) {
			for (int l = -1; l < 2; l++){
				if ((k == 0 && l == 0) || i+k < 0 || j+l < 0 || i+k >= res.rows || j+l >= res.cols)
					continue;
				else if (res.at<uchar>(i+k, j+l) != 255 && frame.at<uchar>(i+k, j+l) >= lowt)
					Q.push(Point(j+l, i+k));
			}
		}
	}
}

Mat PreProcessing::matNorm(Mat& mat) {
	Mat norm = Mat::zeros(mat.rows, mat.cols, CV_8U);
	for(int i = 0; i < mat.rows; ++i) {
        for(int j = 0; j < mat.cols; ++j) {
            Vec3b pix = mat.at<Vec3b>(i, j);
			norm.at<uchar>(i, j) = cv::norm(pix);
        }
    }
	return norm;
}

double PreProcessing::evaluateMovementByColor() {
	int res;
	Mat noSkinMovement;
	subtract(difference, filteredByColor, noSkinMovement);

	imshow("noSkinMovement", noSkinMovement);
	return mean(noSkinMovement)[0];
}

int PreProcessing::evaluateMovement(Mat& frame1, Mat& frame2) {
	//Finding the standard deviations of current and previous frame.
		Scalar prevStdDev, currentStdDev;
		meanStdDev(frame1, Scalar(), prevStdDev);
		meanStdDev(frame2, Scalar(), currentStdDev);

		Scalar diff = currentStdDev - prevStdDev;
		int sum = 0;
		for (int i = 0; i < 3; i++)
			if(diff[i] < 0)
				sum += -diff[i];
			else
				sum += diff[i];
		// return sum;
		return cv::norm(diff);
}
