#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include "image.h"

class PreProcessing {
    Image<Vec3b> currentFrame;
    Image<Vec3b> previousFrame;
    
    Image<uchar> difference;
    Image<uchar> canny;
    Image<Vec3b> intersectionFrame;

    vector<vector<Point>> filteredContours;

    int frameJump;
    int currentFrameIdx;

    public:
    PreProcessing(int in_frameJump = 1): frameJump(in_frameJump), currentFrameIdx(0) {}

    // Set the current Frame (suppose that this function is called every frame!)
    void setCurrentFrame(Image<Vec3b>& frame);

    // Get functions
    Image<Vec3b>& getCurrentFrame() { return currentFrame; }
    Image<Vec3b>& getPreviousFrame() { return previousFrame; }
    Image<uchar>& getDifference() { return difference; }
    Image<uchar>& getCanny() { return canny; }
    Image<Vec3b>& getIntersectionFrame() { return intersectionFrame; }
    vector<vector<Point>>& getFilteredContours() { return filteredContours; }

    // Computer the difference between the current frame and previous frame
    void frameDifferencing(uchar threshold, bool show);
    bool findLargestIntersectionContour();
    
    void applyCanny(Mat& frame, double threshold1, double threshold2) {
        cv::Canny(frame, canny, threshold1, threshold2);
    }
    static void applyGaussianBlur(Mat& frame, Size2i size, double sigmaX, double sigmaY) {
        cv::GaussianBlur(frame, frame, size, 30, sigmaX, sigmaY);
    }
    static void applyDilate(Mat& frame, int strength = 1) {
        cv::dilate(frame, frame, cv::Mat(), cv::Point(-1, -1), strength);
    }
};
