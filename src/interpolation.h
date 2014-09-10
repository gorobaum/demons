#ifndef DEMONS_INTERPOLATION_H_
#define DEMONS_INTERPOLATION_H_

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "imagefunctions.h"

class Interpolation {
	public:
		template<typename T> 
		static T bilinearInterpolation(cv::Mat image, float row, float col, bool print);
		template<typename T> 
		static T NNInterpolation(cv::Mat image, float row, float col);
	private:
		static int getNearestInteger(float number);
};

template<typename T>
T Interpolation::bilinearInterpolation(cv::Mat image, float row, float col, bool print) {
	int u = trunc(row);
	int v = trunc(col);
	uchar pixelOne = ImageFunctions::getPixel<uchar>(image, u, v);
	uchar pixelTwo = ImageFunctions::getPixel<uchar>(image, u+1, v);
	uchar pixelThree = ImageFunctions::getPixel<uchar>(image, u, v+1);
	uchar pixelFour = ImageFunctions::getPixel<uchar>(image, u+1, v+1);

	T interpolation = (u+1-row)*(v+1-col)*pixelOne
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

template<typename T>
T Interpolation::NNInterpolation(cv::Mat image, float row, float col) {
	int nearRow = getNearestInteger(row);
	int nearCol = getNearestInteger(col);
	T aux = ImageFunctions::getPixel<T>(image, nearRow, nearCol);
	return aux;
}

#endif