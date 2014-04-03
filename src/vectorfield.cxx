#include <opencv2/imgproc/imgproc.hpp>

#include "vectorfield.h"

VectorField::VectorField (cv::Mat vectorX, cv::Mat vectorY) {
  vectorX_ = vectorX;
	vectorY_ = vectorY;
	startRow();
}

VectorField::VectorField (int rows, int cols) {
  vectorX_ = cv::Mat::zeros(rows, cols, CV_LOAD_IMAGE_GRAYSCALE);
	vectorY_ = cv::Mat::zeros(rows, cols, CV_LOAD_IMAGE_GRAYSCALE);
	startRow();
}

void VectorField::startRow() {
	currentRow = 0;
	ptrRowX = vectorX_.ptr(currentRow);
	ptrRowY = vectorY_.ptr(currentRow);
}

std::vector<uchar> VectorField::getVectorAt(int row, int col) {
	updateCurrentRow(row);
	std::vector<uchar> vector(ptrRowX[col], ptrRowY[col]);
	return vector;
}

void VectorField::updateVector(int row, int col, uchar xValue, uchar yValue) {
	updateCurrentRow(row);
	ptrRowX[col] = xValue;
	ptrRowY[col] = yValue;
}

void VectorField::applyGaussianFilter() {
	GaussianBlur( vectorX_, vectorX_, cv::Size( 3, 3 ), 0, 0 );
	GaussianBlur( vectorY_, vectorY_, cv::Size( 3, 3 ), 0, 0 );
}

void VectorField::updateCurrentRow(int newRow) {
	if (currentRow != newRow) {
		ptrRowX = vectorX_.ptr(newRow);
		ptrRowY = vectorY_.ptr(newRow);
		currentRow = newRow;
	}
}