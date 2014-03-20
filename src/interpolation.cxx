
#include <cstdio>
#include <cmath>

#include "interpolation.h"

float Interpolation::bilinearInterpolation(Mat image, float x, float y) {
	int u = trunc(x);
	int v = trunc(y);
	float pixelOne = getPixel(image, u, v);
	float pixelTwo = getPixel(image, u+1, v);
	float pixelThree = getPixel(image, u, v+1);
	float pixelFour = getPixel(image, u+1, v+1);

	float interpolation = (u+1-x)*(v+1-y)*pixelOne*1.0 
												+ (x-u)*(v+1-y)*pixelTwo*1.0 
												+ (u+1-x)*(y-v)*pixelThree*1.0
												+ (x-u)*(y-v)*pixelFour*1.0;
	return interpolation;
}

float Interpolation::NNInterpolation(Mat image, float x, float y) {
	int realX = getNearestInteger(x);
	int realY = getNearestInteger(y);
	return image.at<uchar>(realX, realY);
}

uchar Interpolation::getPixel(Mat image, int x, int y){

	if (x > image.cols) {
		return 0;
	} else if (y > image.rows) {
		return 0;
	} else {
		return image.at<uchar>(y,x);
	}
}

int Interpolation::getNearestInteger(float number) {
	if ((number - floor(number)) <= 0.5) {
		return floor(number);
	} else {
		return floor(number) + 1.0;
	}
}
