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
		CImg<float> originalImage_;
};

#endif