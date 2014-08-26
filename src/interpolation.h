#ifndef DEMONS_INTERPOLATION_H_
#define DEMONS_INTERPOLATION_H_

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

class Interpolation {
	public:
		static float fbilinearInterpolation(cv::Mat image, float row, float col, bool print);
		static uchar bilinearInterpolation(cv::Mat image, float row, float col, bool print);
		static uchar NNInterpolation(cv::Mat image, float row, float col);
		static cv::Scalar getPixel(cv::Mat image, int row, int col);
	private:
		static int getNearestInteger(float number);
};

#endif