
#include <cstdio>
#include <cmath>

#include "interpolation.h"

float Interpolation::bilinearInterpolation(CImg<float> image, float x, float y) {
	int u = trunc(x);
	int v = trunc(y);
	float pixelOne = image.atXY(u, v, 0, 0, 0.0);
	float pixelTwo = image.atXY(u+1, v, 0, 0, 0.0);
	float pixelThree = image.atXY(u, v+1, 0, 0, 0.0);
	float pixelFour = image.atXY(u+1, v+1, 0, 0, 0.0);

	float interpolation = (u+1-x)*(v+1-y)*pixelOne*1.0 
												+ (x-u)*(v+1-y)*pixelTwo*1.0 
												+ (u+1-x)*(y-v)*pixelThree*1.0
												+ (x-u)*(y-v)*pixelFour*1.0;
	return interpolation;
}

float Interpolation::NNInterpolation(CImg<float> image, float x, float y) {
	int realX = getNearestInteger(x);
	int realY = getNearestInteger(y);
	return image.atXY(realX, realY, 0, 0, 0.0);
}

int Interpolation::getNearestInteger(float number) {
	if ((number - floor(number)) <= 0.5) {
		return floor(number);
	} else {
		return floor(number) + 1.0;
	}
}
