#ifndef DEMONS_INTERPOLATION_H_
#define DEMONS_INTERPOLATION_H_

#include <cmath>
#include <vector>
#include "image.h"

class Interpolation {
	public:
		Interpolation(Image<unsigned char>& image) : image_(image) {};
		template<typename T> 
		T trilinearInterpolation(double x, double y, double z);
		template<typename T> 
		T NNInterpolation(double x, double y, double z);
	private:
		Image<unsigned char>& image_;
		int getNearestInteger(double number);
};

template<typename T>
T Interpolation::trilinearInterpolation(double x, double y, double z) {
	int x0 = trunc(x);
	int y0 = trunc(y);
	int z0 = trunc(z);
	int x1 = x0+1;
	int y1 = y0+1;
	int z1 = z0+1;
	double xd = (x-x0)/(x1-x0);
	double yd = (y-y0)/(y1-y0);
	double zd = (z-z0)/(z1-z0);
	double c00 = image_.getPixelAt(x0,y0,z0)*(1-xd) + image_.getPixelAt(x1,y0,z0)*xd;
	double c10 = image_.getPixelAt(x0,y1,z0)*(1-xd) + image_.getPixelAt(x1,y1,z0)*xd;
	double c01 = image_.getPixelAt(x0,y0,z1)*(1-xd) + image_.getPixelAt(x1,y0,z1)*xd;
	double c11 = image_.getPixelAt(x0,y1,z1)*(1-xd) + image_.getPixelAt(x1,y1,z1)*xd;
	double c0 = c00*(1-yd)+c10*yd;
	double c1 = c01*(1-yd)+c11*yd;
	double c = c0*(1-zd)+c1*zd;
	return c;
}

template<typename T>
T Interpolation::NNInterpolation(double x, double y, double z) {
	int nearX = getNearestInteger(x);
	int nearY = getNearestInteger(y);
	int nearZ = getNearestInteger(z);
	return image_.getPixelAt(nearX,nearY,nearZ);
}

#endif