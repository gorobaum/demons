#ifndef DEMONS_INTERPOLATION_H_
#define DEMONS_INTERPOLATION_H_

#include <cmath>
#include <vector>
#include "image.h"

class Interpolation {
	public:
		Interpolation(Image<float>& image) : image_(image) {};
		template<typename T> 
		T trilinearInterpolation(float x, float y, float z);
		template<typename T> 
		T NNInterpolation(float x, float y, float z);
	private:
		Image<float>& image_;
		int getNearestInteger(float number);
};

template<typename T>
T Interpolation::trilinearInterpolation(float x, float y, float z) {
	int x0 = trunc(x);
	int y0 = trunc(y);
	int z0 = trunc(z);
	int x1 = x0+1;
	int y1 = y0+1;
	int z1 = z0+1;
	float xd = (x-x0)/(x1-x0);
	float yd = (y-y0)/(y1-y0);
	float zd = (z-z0)/(z1-z0);
	float c00 = image_.getPixelAt(x0,y0,z0)*(1-xd) + image_.getPixelAt(x1,y0,z0)*xd;
	float c10 = image_.getPixelAt(x0,y1,z0)*(1-xd) + image_.getPixelAt(x1,y1,z0)*xd;
	float c01 = image_.getPixelAt(x0,y0,z1)*(1-xd) + image_.getPixelAt(x1,y0,z1)*xd;
	float c11 = image_.getPixelAt(x0,y1,z1)*(1-xd) + image_.getPixelAt(x1,y1,z1)*xd;
	float c0 = c00*(1-yd)+c10*yd;
	float c1 = c01*(1-yd)+c11*yd;
	float c = c0*(1-zd)+c1*zd;
	return c;
}

template<typename T>
T Interpolation::NNInterpolation(float x, float y, float z) {
	int nearX = getNearestInteger(x);
	int nearY = getNearestInteger(y);
	int nearZ = getNearestInteger(z);
	return image_.getPixelAt(nearX,nearY,nearZ);
}

#endif