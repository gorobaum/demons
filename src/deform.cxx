
#include <cstdio>
#include "deform.h"
	
CImg<float> Deform::applySinDeformation(CImg<float> image) {
	CImg<float> deformated(image);
	printf("%f\n", *image.data());
	return deformated;
}