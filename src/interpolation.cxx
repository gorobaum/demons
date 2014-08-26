
#include <cstdio>
#include <cmath>
#include <iostream>

#include "interpolation.h"

float Interpolation::fbilinearInterpolation(cv::Mat image, float row, float col, bool print) {
	int u = trunc(row);
	int v = trunc(col);
	uchar pixelOne = getPixel(image, u, v).val[0];
	uchar pixelTwo = getPixel(image, u+1, v).val[0];
	uchar pixelThree = getPixel(image, u, v+1).val[0];
	uchar pixelFour = getPixel(image, u+1, v+1).val[0];

	float interpolation = (u+1-row)*(v+1-col)*pixelOne
												+ (row-u)*(v+1-col)*pixelTwo 
												+ (u+1-row)*(col-v)*pixelThree
												+ (row-u)*(col-v)*pixelFour;
	if (print) {
		std::cout<< "Row = " << row << " Col = " << col << "\n";
		std::cout<< "u = " << u << " v = " << v << "\n";
		std::cout<< "(u+1-row) = " << (u+1-row) << " (v+1-col) = " << (v+1-col) << " pixelOne = " << (int)pixelOne << "\n";
		std::cout<< "(row-u) = " << (row-u) << " (v+1-col) = " << (v+1-col) << " pixelTwo = " << (int)pixelTwo << "\n";
		std::cout<< "(u+1-row) = " << (u+1-row) << " (col-v) = " << (col-v) << " pixelThree = " << (int)pixelThree << "\n";
		std::cout<< "(row-u) = " << (row-u) << " (col-v) = " << (col-v) << " pixelFour = " << (int)pixelFour << "\n";
		std::cout<< "Interpolation = " << interpolation << "\n";
	}
	return interpolation;
}

uchar Interpolation::bilinearInterpolation(cv::Mat image, float row, float col, bool print) {
	int u = trunc(row);
	int v = trunc(col);
	uchar pixelOne = getPixel(image, u, v).val[0];
	uchar pixelTwo = getPixel(image, u+1, v).val[0];
	uchar pixelThree = getPixel(image, u, v+1).val[0];
	uchar pixelFour = getPixel(image, u+1, v+1).val[0];

	uchar interpolation = (u+1-row)*(v+1-col)*pixelOne
												+ (row-u)*(v+1-col)*pixelTwo 
												+ (u+1-row)*(col-v)*pixelThree
												+ (row-u)*(col-v)*pixelFour;
	if (print) {
		std::cout<< "Row = " << row << " Col = " << col << "\n";
		std::cout<< "u = " << u << " v = " << v << "\n";
		std::cout<< "(u+1-row) = " << (u+1-row) << " (v+1-col) = " << (v+1-col) << " pixelOne = " << (int)pixelOne << "\n";
		std::cout<< "(row-u) = " << (row-u) << " (v+1-col) = " << (v+1-col) << " pixelTwo = " << (int)pixelTwo << "\n";
		std::cout<< "(u+1-row) = " << (u+1-row) << " (col-v) = " << (col-v) << " pixelThree = " << (int)pixelThree << "\n";
		std::cout<< "(row-u) = " << (row-u) << " (col-v) = " << (col-v) << " pixelFour = " << (int)pixelFour << "\n";
		std::cout<< "Interpolation = " << interpolation << "\n";
	}
	return interpolation;
}

uchar Interpolation::NNInterpolation(cv::Mat image, float row, float col) {
	int nearRow = getNearestInteger(row);
	int nearCol = getNearestInteger(col);
	cv::Scalar aux = getPixel(image, nearRow, nearCol);
	return aux.val[0];
}

cv::Scalar Interpolation::getPixel(cv::Mat image, int row, int col) {
	if (col > image.cols-1 || col < 0)
		return 0;
	else if (row > image.rows-1 || row < 0)
		return 0;
	else {
		return image.at<uchar>(row, col);
	}
}

int Interpolation::getNearestInteger(float number) {
	if ((number - floor(number)) <= 0.5) return floor(number);
	return floor(number) + 1.0;
}
