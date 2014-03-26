#ifndef DEMONS_INTERPOLATION_H_
#define DEMONS_INTERPOLATION_H_

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;

class Interpolation {
	public:
		static int NNInterpolation(Mat image, float row, float col);
		static int bilinearInterpolation(Mat image, float row, float col);
		static int getPixel(Mat image, int row, int col);
	private:
		static int getNearestInteger(float number);
};

#endif