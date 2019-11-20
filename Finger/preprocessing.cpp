#include "preprocessing.h"

using namespace std;
using namespace cv;

// Set the current Frame
// Suppose that this function is called every frame!
void PreProcessing::setCurrentFrame(Image<Vec3b>& frame) {
	currentFrame = frame;
}

void PreProcessing::frameThreshold(Mat frame, uchar threshold) {
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

// Computer the difference between the current frame and previous frame
void PreProcessing::frameDifferencingBgSb(uchar threshold, bool show) {
	Mat copy = currentFrame.clone();
	GaussianBlur(copy, copy, Size(9, 9), 30, 30);

	//update the background model
	bgs->apply(copy, difference);

	cv::erode(difference, difference, cv::Mat(), cv::Point(-1, -1), 2);
	cv::dilate(difference, difference, cv::Mat(), cv::Point(-1, -1), 2);
    frameThreshold(difference, threshold);
    
    if (show) {
        imshow("diff with threshold", difference);
    }
}

Mat PreProcessing::matNorm(Mat& mat) {
	Mat norm = Mat::zeros(mat.rows, mat.cols, CV_8U);
	for(int i = 0; i < mat.rows; i++) {
        for(int j = 0; j < mat.cols; j++) {
            Vec3b pix = mat.at<Vec3b>(i, j);
			norm.at<uchar>(i, j) = sqrt((pix[0] * pix[0]) + (pix[1] * pix[1]) + (pix[2] * pix[2]));
        }
    }
	return norm;
}

int PreProcessing::evaluateMovement(Mat frame1, Mat frame2) {
	//Finding the standard deviations of current and previous frame.
		Scalar prevStdDev, currentStdDev;
		meanStdDev(frame1, Scalar(), prevStdDev);
		meanStdDev(frame2, Scalar(), currentStdDev);

		Scalar diff = currentStdDev - prevStdDev;
		int sum = 0;
		for (int i = 0; i < 4; i++)
			if(diff[i] < 0)
				sum += -diff[i];
			else
				sum += diff[i];
		
		return sum;
}

// Computer the difference between the current frame and previous frame
void PreProcessing::frameDifferencingAvgRun(uchar threshold, bool show) {
	auto copy = Image<Vec3b>(currentFrame.clone());
	// GaussianBlur(copy, copy, Size(11, 11), 30, 30);

	// Calculate an "avg" frame, compute the differencing on it
	if (accumulatedFrame.empty()) {
		copy.convertTo(accumulatedFrame, CV_32F);
	}
	Image<Vec3b> resultAccumulatedFrame = Image<Vec3b>(Mat::zeros(currentFrame.rows, currentFrame.cols, CV_32FC3));
	convertScaleAbs(accumulatedFrame, resultAccumulatedFrame);

	Mat LabResultAccumulatedFrame;
	Mat LabCopy;
	cvtColor(resultAccumulatedFrame, LabResultAccumulatedFrame, COLOR_BGR2Lab);
	cvtColor(copy, LabCopy, COLOR_BGR2Lab);

	Mat diff;
	absdiff(LabResultAccumulatedFrame, LabCopy, diff);

	if (evaluateMovement(accumulatedFrame, copy) > 2)
		accumulateWeighted(copy, accumulatedFrame, 0.01);
	
	difference = Image<uchar>(matNorm(diff));
	cv::erode(difference, difference, cv::Mat(), cv::Point(-1, -1), 2);
	cv::dilate(difference, difference, cv::Mat(), cv::Point(-1, -1), 2);
    frameThreshold(difference, threshold);
    
    if (show) {
		imshow("resultAccumulatedFrame", resultAccumulatedFrame);
        imshow("diff with threshold", difference);
    }
}

void PreProcessing::addContours()
{
	int m = currentFrame.rows, n = currentFrame.cols;

	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	vector<Vec3f> circles;
	intersectionFrame = Image<Vec3b>(currentFrame.clone());

	findContours(canny, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_NONE);
	filteredContours = contours;
	drawContours(intersectionFrame, contours, -1, Scalar(128, 0, 0), 2);
}


void fillRow(Image<uchar>& frame, int row, int first, int last)
{
	if (first < last)
	{
		for (int i = first; i < last; ++i)
			frame.at<uchar>(row, i) = 255;
	}
}

void fillCol(Image<uchar>& frame, int col, int first, int last)
{
	if (first < last)
	{
		for (int i = first; i < last; ++i)
			frame.at<uchar>(i, col) = 255;
	}
}

void PreProcessing::fillHorizontalGaps(Image<uchar>& frame, int gap)
{
	int m = frame.rows, n = frame.cols, start, end;

	for (int i = 0; i < m; ++i) // Scan each line
		for (start = 0; start < n - 2; ++start) // Scan each line 
			if (frame.at<uchar>(i, start) == 255 && frame.at<uchar>(i, start + 1) == 0)
				//If you get to the edge of a white part, check where it ends
				for (end = start + 2; end < n; ++end)
					//If next white space is within "gap" pixels, fille everything in between
					if (frame.at<uchar>(i, end) == 255 && end - start < gap)
					{
						fillRow(frame, i, start, end);
						start = end - 1; //Continue scan from where stopped 
						//will be incremented by one since its the end of the loop
						break;
					}	
}


void PreProcessing::fillVerticalGaps(Image<uchar>& frame, int gap)
{
	int m = frame.rows, n = frame.cols, start, end;

	for (int i = 0; i < n; ++i) // Scan each line
		for (start = 0; start < m - 2; ++start) // Scan each line 
			if (frame.at<uchar>(start, i) == 255 && frame.at<uchar>(start + 1, i) == 0)
				//If you get to the edge of a white part, check where it ends
				for (end = start + 2; end < m; ++end)
					//If next white space is within "gap" pixels, fille everything in between
					if (frame.at<uchar>(end, i) == 255 && end - start < gap)
					{
						fillCol(frame, i, start, end);
						start = end - 1; //Continue scan from where stopped 
						//will be incremented by one since its the end of the loop
						break;
					}
}
