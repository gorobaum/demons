#ifndef DEMONS_INTERPOLATION_H_
#define DEMONS_INTERPOLATION_H_

#include <cmath>
#include <vector>
#include "image.h"

class Interpolation {
	public:
		Interpolation(Image<unsigned char>& image) : image_(image) {};
		template<typename T> 
		T trilinearInterpolation(float x, float y, float z);
		template<typename T> 
		T NNInterpolation(float x, float y, float z);
	private:
		Image<unsigned char>& image_;
		int getNearestInteger(float number);
};

template<typename T>
T Interpolation::trilinearInterpolation(float x, float y, float z) {
	int t = trunc(x);
	int u = trunc(y);
	int v = trunc(z);
	return image_.getPixelAt(t,u,v);
}

template<typename T>
T Interpolation::NNInterpolation(float x, float y, float z) {
	int nearX = getNearestInteger(x);
	int nearY = getNearestInteger(y);
	int nearZ = getNearestInteger(z);
	return image_.getPixelAt(nearX,nearY,nearZ);
}

#endif