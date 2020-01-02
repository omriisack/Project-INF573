#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/video/background_segm.hpp>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <tuple>
#include <opencv2/videoio.hpp>
#include "preprocessing.h"
#include "hand_detection.h"

using namespace std;
using namespace cv;

int main() {
	VideoCapture capture;

	if (!capture.open(0))
		return 0;

	PreProcessing preProcessing;
	Image<Vec3b> frame;
	bool detected = false;



	//VideoWriter video("finger_detection_demo.avi", CV_FOURCC('M', 'J', 'P', 'G'), 24, Size(640, 480));
	//video.open("finger_detection_demo.avi", CV_FOURCC('M', 'J', 'P', 'G'), 24, Size(640, 480));



	while(true)
	{
		capture >> frame;
		if (frame.empty())
			break;

		Image<Vec3b> result = (Image<Vec3b>)frame.clone();
		vector<Point> handConvexHull;
		vector<Point> handContour;

		preProcessing.setCurrentFrame(frame);

		// frame differencing
		preProcessing.frameDifferencingAvgRun(35, 15, detected, false, true);


		// canny
		preProcessing.applyCanny(preProcessing.getDifference(), 50, 20);

		// mask
		preProcessing.filterByMask(preProcessing.getDifference(), false);

		// filter the skin color
		preProcessing.filterSkinColor(preProcessing.getFilteredByMask(), false);

		
		detected = findHandContour(result, preProcessing.getContours(), true);
		imshow("Hand Detector", result);

		//video.write(result);

		if (waitKey(10) == 27) break; // stop capturing by pressing ESC 
	}
	//video.release();
	return 0;
}

