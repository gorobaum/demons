
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <ctime>
#include <vector>
#include <opencv2/imgproc/imgproc.hpp>

#include "demons.h"
#include "interpolation.h"

void Demons::demons() {
	int rows = staticImage_.rows, cols = staticImage_.cols;
	// Create the deformed image
	deformedImage_ = cv::Mat::zeros(rows, cols, CV_LOAD_IMAGE_GRAYSCALE);
	VectorField gradients = findGrad();
	VectorField displField(rows, cols);
	int iteration = 1;
	std::vector<int> compression_params;
	compression_params.push_back(CV_IMWRITE_JPEG_QUALITY);
	compression_params.push_back(95);
	for(int i = 0; i < 100; i++) {
		time(&startTime);
		for(int row = 0; row < rows; row++) {
			uchar* rowDeformed = deformedImage_.ptr(row);
			for(int col = 0; col < cols; col++) {
				std::vector<float> displVector = displField.getVectorAt(row, col);
				float newRow = row - displVector[0];
				float newCol = col - displVector[1];
				rowDeformed[col] = Interpolation::bilinearInterpolation(movingImage_, newRow, newCol);
				std::vector<float> gradient = gradients.getVectorAt(row, col);
				updateDisplField(displField, gradient, row, col);
			}
		}
		displField.applyGaussianFilter();
		double iterTime = getIterationTime(startTime);
		std::cout << "Iteration " << iteration << " took " << iterTime << " seconds.\n";
    iteration++;
	}
	std::cout << "termino rapa\n";
}

void Demons::updateDisplField(VectorField displacement, std::vector<float> gradient, int row, int col) {
	uchar* staticRow = staticImage_.ptr(row);
	uchar* deformedRow = deformedImage_.ptr(row);
	float dif = (deformedRow[col] - staticRow[col]);
	float division = pow(dif,2) + pow(gradient[0],2) + pow(gradient[1],2);
	if (division > 0.0) {
		std::vector<float> displVector = displacement.getVectorAt(row, col);
		float xValue = displVector[0] + dif/division*gradient[0];
		float yValue = displVector[1] + dif/division*gradient[1];
		displacement.updateVector(row, col, xValue, yValue);
	} else {
		displacement.updateVector(row, col, 0.0, 0.0);
	}
}

VectorField Demons::findGrad() {
	cv::Mat sobelX;
	cv::Mat sobelY;
	cv::Sobel(staticImage_, sobelX, CV_64F, 1, 0);
	cv::Sobel(staticImage_, sobelY, CV_64F, 0, 1);
	sobelX = normalizeSobelImage(sobelX);
	sobelY = normalizeSobelImage(sobelY);
	VectorField grad(sobelX, sobelY);
	return grad;
}

cv::Mat Demons::normalizeSobelImage(cv::Mat sobelImage) {
	double minVal, maxVal;
  minMaxLoc(sobelImage, &minVal, &maxVal); //find minimum and maximum intensities
  cv::Mat normalized;
  sobelImage.convertTo(normalized, CV_8U, 255.0/(maxVal - minVal), -minVal * 255.0/(maxVal - minVal));
  return normalized;
}

double Demons::getIterationTime(time_t startTime) {
	double iterationTime = difftime(time(NULL), startTime);
	totalTime += iterationTime;
	return iterationTime;
}

cv::Mat Demons::getRegistration() {
	return deformedImage_;
}