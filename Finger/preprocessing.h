#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/video/background_segm.hpp>

#include "image.h"

class PreProcessing {
    Image<Vec3b> currentFrame;     // Original current frame
	Image<uchar> difference;       // After differencing operation
	Image<Vec3b> filteredByMask;   // Original frame filtered by a given mask
    Image<uchar> filteredByColor;  // Given frame filtered by color range
    
    Image<uchar> canny;
    vector<vector<Point>> contours;
    
    Mat accumulatedFrame;
	Ptr<BackgroundSubtractor> bgs;
	bool shouldAccumulate;

    // Utils
    static Mat matNorm(Mat& mat);
	double evaluateMovementByColor();
    static int evaluateMovement(Mat& frame1, Mat& frame2);
  

    public:
		PreProcessing() { bgs = createBackgroundSubtractorKNN(); }

    // Set the current Frame (suppose that this function is called every frame!)
    void setCurrentFrame(Image<Vec3b>& frame);

    // Get functions
    Image<Vec3b>& getCurrentFrame() { return currentFrame; }
	Image<uchar>& getDifference() { return difference; }
	Image<Vec3b>& getFilteredByMask() { return filteredByMask; }
	Image<uchar>& getFilteredByColor() { return filteredByColor; }
    Image<uchar>& getCanny() { return canny; }
    vector<vector<Point>>& getContours() { return contours; }

    // Frame Differencing
    void frameDifferencingBgSb(uchar threshold, bool show);
    void frameDifferencingAvgRun(uchar hight, uchar lowt, bool detected, bool labColor, bool show);
	
    // Filters
	void filterByMask(const Image<uchar>& mask, bool show);
	void filterSkinColor(const Image<Vec3b>& input, bool show);
    void frameThreshold(Mat& frame, uchar threshold);
    void frameThresholdSeeds(const Image<uchar>& frame, Image<uchar>& res, int t1, int t2);
    
    // Other operations
    void applyMeanReduction(Mat& frame) {
        cv::Scalar mean, stddev;
        cv::meanStdDev(frame, mean, stddev);
        for(int i = 0; i < frame.rows; i++) {
            for(int j = 0; j < frame.cols; j++) {
                auto pix = frame.at<uchar>(i, j);
                if(pix < 2*mean[0])
                    frame.at<uchar>(i, j) = pix/2;
                else
                    frame.at<uchar>(i, j) = pix;
            }
        }
    }
    void applyCanny(Mat& frame, double threshold1, double threshold2) {
        cv::Canny(frame, canny, threshold1, threshold2, 3, true);
        contours.clear();
        vector<Vec4i> hierarchy;
		vector<vector<Point>> allContours;
	    findContours(canny, allContours, hierarchy, RETR_CCOMP, CHAIN_APPROX_NONE);
		for (int i = 0; i < allContours.size(); i++) //Add only closed contours
			if (hierarchy[i][2] > 0)
				contours.push_back(allContours[i]);

    }
    static void applyGaussianBlur(Mat& frame, Size2i size, double sigmaX, double sigmaY) {
        cv::GaussianBlur(frame, frame, size, sigmaX, sigmaY);
    }
    static void applyDilate(Mat& frame, int strength = 1) {
        cv::dilate(frame, frame, cv::Mat(), cv::Point(-1, -1), strength);
    }
	static void applyErode(Mat& frame, int strength = 1) {
		cv::erode(frame, frame, cv::Mat(), cv::Point(-1, -1), strength);
	}
};
