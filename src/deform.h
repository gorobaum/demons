#ifndef DEMONS_DEFORM_H_
#define DEMONS_DEFORM_H_

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

class Deform {
	public:
		explicit Deform (cv::Mat originalImage):
			originalImage_(originalImage) {}
		cv::Mat applySinDeformation(double amp, double freq);
		cv::Mat rotate(double angle);
		cv::Mat translation(int width, int height);
	protected:
		cv::Mat originalImage_;
};

#endif