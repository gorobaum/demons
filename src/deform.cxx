
#include <cstdio>
#include <cmath>

#include "deform.h"
#include "interpolation.h"
	
Mat Deform::applySinDeformation() {
	Mat deformatedImage(originalImage_);
	cimg_forXY(deformatedImage,x,y) {
		deformatedImage(x,y) = Interpolation::NNInterpolation(originalImage_, x, y + 2*sin(x/16));
	}
	return deformatedImage;
}
