
#include <cstdio>
#include <cmath>
#include "deform.h"
	
CImg<float> Deform::applySinDeformation() {
	CImg<float> deformatedImage(originalImage_);
	for (int y = 0; y < deformatedImage.height(); y++) {
		for (int x = 0; x < deformatedImage.width(); x++) {
			float retrieve = originalImage_.atXY(x, y, 0, 0, 0.0);
			float newX, newY;
			newX = x;
			newY = y + 2*sin(x/16);
			int newValue = bilinearInterpolation(originalImage_, newX, newY);
			deformatedImage.set_linear_atXY(newValue, x, y);
		}
	}
	return deformatedImage;
}

float Deform::bilinearInterpolation(CImg<float> image, float x, float y) {
	int u = trunc(x);
	int v = trunc(y);
	float pixelOne = originalImage_.atXY(u, v, 0, 0, 0.0);
	float pixelTwo = originalImage_.atXY(u+1, v, 0, 0, 0.0);
	float pixelThree = originalImage_.atXY(u, v+1, 0, 0, 0.0);
	float pixelFour = originalImage_.atXY(u+1, v+1, 0, 0, 0.0);

	float interpolation = (u+1-x)*(v+1-y)*pixelOne*1.0 + (x-u)*(v+1-y)*pixelTwo*1.0 + (u+1-x)*(y-v)*pixelThree*1.0 + (x-u)*(y-v)*pixelFour*1.0;
	return interpolation;
}

float Deform::NNInterpolation(CImg<float> image, float x, float y) {
	return 0.0;
}

int Deform::getNearestInteger(float number) {
	if ((number - floor(number)) <= 0.5) {
		return floor(number);
	} else {
		return floor(number) + 1.0;
	}
}
