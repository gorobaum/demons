#include "gradient.h"
#include <iostream>
#include <vector>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>

VectorField Gradient::getBasicGradient() {
	int rows = image_.rows, cols = image_.cols;
	std::vector<int> dimensions;
	dimensions.push_back(rows);
	dimensions.push_back(cols);
	VectorField gradient(dimensions, 0.0);
	for (int x = 0; x < dimensions[0]; x++)
		for (int y = 0; y < dimensions[1]; y++) {
			std::vector<double> gradVector(3, 0.0);
			gradVector[0] += ImageFunctions::getValue<uchar>(image_, x-1,y)*(-0.5);
			gradVector[0] += ImageFunctions::getValue<uchar>(image_, x+1,y)*(0.5);
			gradVector[1] += ImageFunctions::getValue<uchar>(image_, x,y-1)*(-0.5);
			gradVector[1] += ImageFunctions::getValue<uchar>(image_, x,y+1)*(0.5);
			gradient.updateVector(x, y, gradVector);
		}
	return gradient;
}