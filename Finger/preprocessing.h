#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include "image.h"

using namespace std;
using namespace cv;

class PreProcessing {
    Image<Vec3b> currentFrame;
    Image<Vec3b> previousFrame;
    
    Image<uchar> difference;
    Image<uchar> canny;
    
    int frameJump;
    int currentFrameIdx;

    public:
    PreProcessing(int in_frameJump = 1): frameJump(in_frameJump) {}

    // Set the current Frame (suppose that this function is called every frame!)
    void setCurrentFrame(Image<Vec3b>& frame);

    // Get functions
    Image<Vec3b>& getCurrentFrame() { return currentFrame; }
    Image<Vec3b>& getPreviousFrame() { return previousFrame; }
    Image<uchar>& getDifference() { return difference; }
    Image<uchar>& getCanny() { return canny; }

    // Computer the difference between the current frame and previous frame
    void frameDifferencing(uchar threshold, bool show);
    bool findLargestIntersectionContour(vector<Point>& largestIntersection);
    
    void applyCanny(Mat& frame, double threshold1, double threshold2) {
        Canny(frame, canny, threshold1, threshold2);
    }
    static void applyGaussianBlur(Mat& frame, Size2i& size, double sigmaX, double sigmaY) {
        GaussianBlur(frame, frame, size, 30, sigmaX, sigmaY);
    }
    static void applyDilate(Mat& frame, int strength) {
        cv::dilate(frame, frame, cv::Mat(), cv::Point(-1, -1), strength);
    }
};
