#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

#include "vectorfield.h"

VectorField::VectorField (cv::Mat &vectorX, cv::Mat &vectorY) {
  vectorX_ = vectorX.clone();
	vectorY_ = vectorY.clone();
}

VectorField::VectorField (int rows, int cols) {
  vectorX_ = cv::Mat::zeros(rows, cols, CV_64F);
	vectorY_ = cv::Mat::zeros(rows, cols, CV_64F);
}

std::vector<float> VectorField::getVectorAt(int row, int col) {
	std::vector<float> auxVec;
	auxVec.push_back(getValue(vectorX_, row, col));
	auxVec.push_back(getValue(vectorY_, row, col));
	return auxVec;
}

void VectorField::updateVector(int row, int col, float xValue, float yValue) {
	vectorX_.at<float>(row, col) = xValue;
	vectorY_.at<float>(row, col) = yValue;
}

void VectorField::applyGaussianFilter() {
	GaussianBlur(vectorX_, vectorX_, cv::Size(3, 3), 0, 0);
	GaussianBlur(vectorY_, vectorY_, cv::Size(3, 3), 0, 0);
}

float VectorField::getValue(cv::Mat image, int row, int col) {
	if (col > image.cols-1 || col < 0)
		return 0;
	else if (row > image.rows-1 || row < 0)
		return 0;
	else {
		return image.at<float>(row, col);
	}
}