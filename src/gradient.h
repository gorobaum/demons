#ifndef DEMONS_GRADIENT_H_
#define DEMONS_GRADIENT_H_

#include <vector>
#include <array>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "vectorfield.h"
#include "imagefunctions.h"

class Gradient {
public:
	Gradient(cv::Mat &image) : image_(image) {};
	VectorField getBasicGradient();
private:
	cv::Mat &image_;
};

#endif