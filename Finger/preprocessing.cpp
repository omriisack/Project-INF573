#include "preprocessing.h"

using namespace std;
using namespace cv;

// Set the current Frame
// Suppose that this function is called every frame!
void PreProcessing::setCurrentFrame(Image<Vec3b>& frame) {
	currentFrame = frame;
}

// Computer the difference between the current frame and previous frame
void PreProcessing::frameDifferencing(uchar threshold, bool show) {
	Mat copy = currentFrame.clone();

	GaussianBlur(copy, copy, Size(5, 5), 30, 30);

	//update the background model
	bgs->apply(copy, difference);

	cv::erode(difference, difference, cv::Mat(), cv::Point(-1, -1), 1);
	cv::dilate(difference, difference, cv::Mat(), cv::Point(-1, -1), 4);

    for(int i = 0; i < difference.rows; i++) {
        for(int j = 0; j < difference.cols; j++) {
            auto pix = difference.at<uchar>(i, j);
            if(pix > threshold)
				difference.at<uchar>(i, j) = 255;
        }
    }

    
    if (show) {
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


void fillLine(Image<uchar>& frame, int line, int first, int last)
{
	if (first < last)
	{
		for (int i = first; i < last; ++i)
			frame.at<uchar>(line, i) = 255;
	}
}

void PreProcessing::fillGaps(Image<uchar>& frame, int gap)
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
						fillLine(frame, i, start, end);
						start = end - 1; //Continue scan from where stopped 
						//will be incremented by one since its the end of the loop
						break;
					}	
}


