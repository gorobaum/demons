#ifndef DEMONS_DEFORM_H_
#define DEMONS_DEFORM_H_

#include "CImg.h"

using namespace cimg_library;

class Deform {
	public:
		explicit Deform (CImg<float> originalImage):
			originalImage_(originalImage) {}
		CImg<float> applySinDeformation ();
	protected:
		float NNInterpolation(CImg<float> image, float x, float y);
		float bilinearInterpolation(CImg<float> image, float x, float y);
		int getNearestInteger(float number);
		CImg<float> originalImage_;
};

#endif