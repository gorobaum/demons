#ifndef DEMONS_INTERPOLATION_H_
#define DEMONS_INTERPOLATION_H_

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;

class Interpolation {
	public:
		static float NNInterpolation(Mat image, float x, float y);
		static float bilinearInterpolation(Mat image, float x, float y);
		static uchar getPixel(Mat image, int x, int y);
	private:
		static int getNearestInteger(float number);
};

#endif