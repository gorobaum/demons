#ifndef DEMONS_INTERPOLATION_H_
#define DEMONS_INTERPOLATION_H_

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "imagefunctions.h"

class Interpolation {
	public:
		Interpolation(cv::Mat &image) : image_(image) {};
		template<typename T> 
		T bilinearInterpolation(float row, float col, bool print);
		template<typename T> 
		T NNInterpolation(float row, float col);
	private:
		cv::Mat &image_;
		int getNearestInteger(float number);
};

template<typename T>
T Interpolation::bilinearInterpolation(float row, float col, bool print) {
	int u = trunc(row);
	int v = trunc(col);
	uchar pixelOne = ImageFunctions::getValue<uchar>(image_, u, v);
	uchar pixelTwo = ImageFunctions::getValue<uchar>(image_, u+1, v);
	uchar pixelThree = ImageFunctions::getValue<uchar>(image_, u, v+1);
	uchar pixelFour = ImageFunctions::getValue<uchar>(image_, u+1, v+1);

	T interpolation = (u+1-row)*(v+1-col)*pixelOne
												+ (row-u)*(v+1-col)*pixelTwo 
												+ (u+1-row)*(col-v)*pixelThree
												+ (row-u)*(col-v)*pixelFour;
	if (print) {
		std::cout<< "Image Rows = " << image_.rows << "\n";
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
T Interpolation::NNInterpolation(float row, float col) {
	int nearRow = getNearestInteger(row);
	int nearCol = getNearestInteger(col);
	T aux = ImageFunctions::getValue<T>(image_, nearRow, nearCol);
	return aux;
}

#endif