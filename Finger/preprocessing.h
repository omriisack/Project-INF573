#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/video/background_segm.hpp>

#include "image.h"

class PreProcessing {
    Image<Vec3b> currentFrame;
	Image<uchar> difference;
    Image<uchar> canny;
    Image<Vec3b> intersectionFrame;
    Mat accumulatedFrame;
    vector<vector<Point>> filteredContours;
	Ptr<BackgroundSubtractor> bgs;

    static Mat matNorm(Mat& mat);
    static int evaluateMovement(Mat frame1, Mat frame2);

    public:
    PreProcessing() { bgs = createBackgroundSubtractorKNN(); }

    // Set the current Frame (suppose that this function is called every frame!)
    void setCurrentFrame(Image<Vec3b>& frame);

    // Get functions
    Image<Vec3b>& getCurrentFrame() { return currentFrame; }
	Image<uchar>& getDifference() { return difference; }
    Image<uchar>& getCanny() { return canny; }
    Image<Vec3b>& getIntersectionFrame() { return intersectionFrame; }
    vector<vector<Point>>& getFilteredContours() { return filteredContours; }

	
    // Computer the difference between the current frame and previous frame
    void frameDifferencingBgSb(uchar threshold, bool show);
    void frameDifferencingAvgRun(uchar threshold, bool show);
	void frameThreshold(Mat frame, uchar threshold);
    void addContours();
	void fillHorizontalGaps(Image<uchar>& frame, int gap);
	void fillVerticalGaps(Image<uchar>& frame, int gap);


    void applyCanny(Mat& frame, double threshold1, double threshold2) {
        cv::Canny(frame, canny, threshold1, threshold2);
    }
    static void applyGaussianBlur(Mat& frame, Size2i size, double sigmaX, double sigmaY) {
        cv::GaussianBlur(frame, frame, size, 30, sigmaX, sigmaY);
    }
    static void applyDilate(Mat& frame, int strength = 1) {
        cv::dilate(frame, frame, cv::Mat(), cv::Point(-1, -1), strength);
    }

	static void applyErode(Mat& frame, int strength = 1) {
		cv::erode(frame, frame, cv::Mat(), cv::Point(-1, -1), strength);
	}
};
