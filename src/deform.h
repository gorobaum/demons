#ifndef DEMONS_DEFORM_H_
#define DEMONS_DEFORM_H_

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

class Deform {
	public:
		explicit Deform (cv::Mat originalImage):
			originalImage_(originalImage) {}
		cv::Mat applySinDeformation();
	protected:
		cv::Mat originalImage_;
};

#endif