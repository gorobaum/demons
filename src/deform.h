#ifndef DEMONS_AUX_H_
#define DEMONS_AUX_H_

#include "CImg.h"

using namespace cimg_library;

class Deform {
public:
	CImg<float> applySinDeformation(CImg<float> image);
};

#endif