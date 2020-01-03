#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/video/background_segm.hpp>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <tuple>

#include "image.h"

bool findHandContour(Image<Vec3b>& frame, vector<vector<Point>>& contours, bool show);
