README:
You will find here a quick description of all the files used in our project.

preprocessing.h:
    This file contains the declaration of the PreProcessing class.
    
    Mat matNorm(Mat& mat)
        Compute the norm of each Vec3b element in the input matrix.
    double evaluateMovementByColor()
        Evaluate how much movement there is in the member variable 'difference' according
        to the no-skin color pixels.
    int evaluateMovement(Mat& frame1, Mat& frame2)
        Evaluate the movement between two frames according to the difference of the standard
        deviation of the two frames.
    void setCurrentFrame(Image<Vec3b>& frame)
        Set the 'currentFrame' member variable.
    void frameDifferencingBgSb(uchar threshold, bool show)
        Compute the differencing using an algorithm (here BackgroundSubtractorKNN)
        implemented in OpenCV. With the option to display the result.
    void frameDifferencingAvgRun(uchar hight, uchar lowt, bool detected, bool labColor, bool show);
        Compute the differencing using a the accumulated frame. With the option to display
        the result.
    void filterByMask(const Image<uchar>& mask, bool show)
        Filter the 'currentFrame' member variable according to a given mask. With the
        option to display the result.
    void filterSkinColor(const Image<Vec3b>& input, bool show)
        Filter the skin color pixels of a given frame. With the option to display
        the result.
    void frameThreshold(Mat& frame, uchar threshold)
        Simple threshold of a given frame.
    void frameThresholdSeeds(const Image<uchar>& frame, Image<uchar>& res, int t1, int t2)
        Threshold with seeds of a given frame.
    void applyMeanReduction(Mat& frame)
        Reduce the value of pixels in a given frame that have a lower value than the mean
        pixel value.
    void applyCanny(Mat& frame, double threshold1, double threshold2)
        Apply the Canny algorithm to a given frame.
    void applyGaussianBlur(Mat& frame, Size2i size, double sigmaX, double sigmaY)
        Apply a gaussion blur to a given frame.
    void applyDilate(Mat& frame, int strength = 1)
        Apply dilate to a given frame.
    void applyErode(Mat& frame, int strength = 1)
        Apply erode to a given frame.

preprocessing.cpp:
    This file contains the implementation of the PreProcessing class.

hand_detection.h:
    This file contains the declaration of the functions used to analyze contours and
    determine if a hand was found or not.

    bool findHandContour(Image<Vec3b>& frame, vector<vector<Point>>& contours, bool show)
        Search for a hand contour in given contours. Draw a corresponding contour in
        the given frame.

hand_detection.cpp:
    This file contains the implementations of the functions declared in hand_detection.h.

main.cpp:
    This file contain a main loop calling the different modules ('preprocessing' and
    'hand_detection) to process the realtime frames of the webcam.
