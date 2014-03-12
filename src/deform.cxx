
#include <cstdio>
#include "deform.h"
	
CImg<float> Deform::applySinDeformation() {
	CImg<float> deformatedImage(originalImage_);
	printf("%f\n", *deformatedImage.data());
	return deformatedImage;
}