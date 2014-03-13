
#include <cstdio>
#include <cmath>

#include "deform.h"
#include "interpolation.h"
	
CImg<float> Deform::applySinDeformation() {
	CImg<float> deformatedImage(originalImage_);
	for (int y = 0; y < deformatedImage.height(); y++) {
		for (int x = 0; x < deformatedImage.width(); x++) {
			float retrieve = originalImage_.atXY(x, y, 0, 0, 0.0);
			float newX, newY;
			newX = x;
			newY = y + 2*sin(x/16);
			float newValue = Interpolation::NNInterpolation(originalImage_, newX, newY);
			deformatedImage.set_linear_atXY(newValue, x, y);
		}
	}
	return deformatedImage;
}
