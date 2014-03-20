#ifndef DEMONS_DEFORM_H_
#define DEMONS_DEFORM_H_

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;

class Deform {
	public:
		explicit Deform (Mat originalImage):
			originalImage_(originalImage) {}
		Mat applySinDeformation();
	protected:
		Mat originalImage_;
};

#endif