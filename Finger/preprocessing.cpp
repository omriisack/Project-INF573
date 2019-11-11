#include "preprocessing.h"

using namespace std;
using namespace cv;


// Set the current Frame
// Suppose that this function is called every frame!
void PreProcessing::setCurrentFrame(Image<Vec3b>& frame) {
    if (currentFrameIdx%frameJump == 0) {
        previousFrame = currentFrame;
        currentFrameIdx = 0;
    }

    currentFrame = Image<Vec3b>(frame.clone());
}

// Computer the difference between the current frame and previous frame
void PreProcessing::frameDifferencing(uchar threshold, bool show) {
    Mat diff;
    absdiff(previousFrame.greyImage(), currentFrame.greyImage(), diff);
    difference = Image<uchar>(Mat::zeros(diff.rows, diff.cols, CV_8U));

    for(int i = 0; i < diff.rows; i++) {
        for(int j = 0; j < diff.cols; j++) {
            uchar pix = diff.at<uchar>(i, j);

            if(pix>threshold)
                difference.at<uchar>(i, j) = 255;
        }
    }

    // Blur ?
    // GaussianBlur(diff, diff, Size(9, 9), 30, 30);
    // GaussianBlur(foregroundMask, foregroundMask, Size(9, 9), 30, 30);
    
    if (show) {
        imshow("diff", diff);
        imshow("diff with threshold", difference);
    }
}

bool PreProcessing::findLargestIntersectionContour(vector<Point>& largestIntersection) {
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(canny, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_NONE);

    int max_index = -1, max_area = 0;
    vector<vector<Point>> intersections;

    for (const vector<Point>& contour : contours) {
        for (const Point& p : contour) {
            if (difference.at<uchar>(p.y, p.x) > 0) {
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
