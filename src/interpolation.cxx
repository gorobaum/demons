
#include <cstdio>
#include <cmath>
#include <iostream>

#include "interpolation.h"

int Interpolation::bilinearInterpolation(Mat image, float row, float col) {
	int u = trunc(row);
	int v = trunc(col);
	int pixelOne = getPixel(image, u, v);
	int pixelTwo = getPixel(image, u+1, v);
	int pixelThree = getPixel(image, u, v+1);
	int pixelFour = getPixel(image, u+1, v+1);

	int interpolation = (u+1-row)*(v+1-col)*pixelOne
												+ (row-u)*(v+1-col)*pixelTwo 
												+ (u+1-row)*(col-v)*pixelThree
												+ (row-u)*(col-v)*pixelFour;
	return interpolation;
}

int Interpolation::NNInterpolation(Mat image, float row, float col) {
	int nearRow = getNearestInteger(row);
	int nearCol = getNearestInteger(col);
	return getPixel(image, nearRow, nearCol);
}

int Interpolation::getPixel(Mat image, int row, int col){
	
	if (col > image.cols-1 || col < 0)
		return 0;
	else if (row > image.rows-1 || row < 0)
		return 0;
	else {
		uchar* iRow = image.ptr(row);
		return iRow[col];
	}
}

int Interpolation::getNearestInteger(float number) {
	if ((number - floor(number)) <= 0.5) return floor(number);
	return floor(number) + 1.0;
}
