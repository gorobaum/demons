#ifndef DEMONS_INTERPOLATION_H_
#define DEMONS_INTERPOLATION_H_

#include "CImg.h"

using namespace cimg_library;

class Interpolation {
	public:
		static float NNInterpolation(CImg<float> image, float x, float y);
		static float bilinearInterpolation(CImg<float> image, float x, float y);
	private:
		static int getNearestInteger(float number);
};

#endif