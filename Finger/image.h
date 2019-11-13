#pragma once

#include <vector>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;

// Traits
template <typename T>
struct pixel_type
{
	static const int value = -1;
};
template <>
struct pixel_type<uchar>
{
	static const int value = CV_8U;
};
template <>
struct pixel_type<Vec3b>
{
	static const int value = CV_8UC3;
};
template <>
struct pixel_type<float>
{
	static const int value = CV_32F;
};

template <typename T> class Image : public Mat {
	Mat grey;
	bool greyed = false;
public:
	// Constructors
	Image() {}
	explicit Image(const Mat& A):Mat(A) {}
	Image(int w,int h):Mat(h,w,pixel_type<T>::value) {}
	// Accessors
	inline T operator()(int x,int y) const { return at<T>(y,x); }
	inline T& operator()(int x,int y) { return at<T>(y,x); }
	inline T operator()(const Point& p) const { return at<T>(p.y,p.x); }
	inline T& operator()(const Point& p) { return at<T>(p.y,p.x); }
	//
	inline int width() const { return cols; }
	inline int height() const { return rows; }
	// To display a floating type image
	Mat& greyImage() {
		if (!greyed) {
			cvtColor(*this, grey, COLOR_BGR2GRAY);
			greyed = true;
		}

		return grey;
	}
	
	Mat gradient() const {
		Mat Igrey;
		cvtColor(*this, Igrey, COLOR_BGR2GRAY);

		int m = Igrey.rows, n = Igrey.cols;
		Mat G2 = Mat(m, n, CV_32F);

		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				if (i == 0 || i == m-1 || j == 0 || j == n-1) {
					G2.at<float>(i, j) = 0;
					continue;
				}

				float Ix, Iy; 
				Ix = 0.5 * (float(Igrey.at<uchar>(i, j+1)) - float(Igrey.at<uchar>(i, j-1)));
				Iy = 0.5 * (float(Igrey.at<uchar>(i+1, j)) - float(Igrey.at<uchar>(i-1, j)));

				G2.at<float>(i, j) = Ix*Ix + Iy*Iy;
			}
		}

		return G2;
	}
};

// Correlation
double NCC(const Image<float>& I1,Point m1,const Image<float>& I2,Point m2,int n);


