
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <ctime>
#include <vector>
#include <opencv2/imgproc/imgproc.hpp>
#include <sstream>

#include "demons.h"
#include "interpolation.h"

void Demons::demons() {
	int rows = staticImage_.rows, cols = staticImage_.cols;
	// Create the deformed image
	deformedImage_ = movingImage_.clone();
	VectorField gradients = findGrad();
	gradients.printField("Gradients.dat");
	VectorField displField(rows, cols);
	int iteration = 1;
	compression_params.push_back(CV_IMWRITE_JPEG_QUALITY);
	compression_params.push_back(95);
	for(int i = 0; i < 300; i++) {
		time(&startTime);
		updateDisplField(displField, gradients);
		displField.applyGaussianFilter();
		updateDeformedImage(displField);
		double iterTime = getIterationTime(startTime);
		printVFN(displField, iteration);
		printVFI(displField, iteration);
		printDeformedImage(iteration);
		std::cout << "Iteration " << iteration << " took " << iterTime << " seconds.\n";
		iteration++;
	}
	std::cout << "termino rapa\n";
}

void Demons::updateDeformedImage(VectorField displField) {
	int rows = displField.getRows(), cols = displField.getCols();
	for(int row = 0; row < rows; row++) {
		uchar* deformedImageRow = deformedImage_.ptr(row);
		for(int col = 0; col < cols; col++) {
			std::vector<float> displVector = displField.getVectorAt(row, col);
			float newRow = row - displVector[0];
			float newCol = col - displVector[1];
			deformedImageRow[col] = Interpolation::bilinearInterpolation(movingImage_, newRow, newCol);
		}
	}
}

void Demons::printDeformedImage(int iteration) {
	std::string imageName("Iteration");
	std::ostringstream converter;
	converter << iteration;
	imageName += converter.str() + ".jpg";
	std::cout << imageName.c_str() << "\n";
    imwrite(imageName.c_str(), deformedImage_, compression_params);
}

void Demons::printVFN(VectorField vectorField, int iteration) {
	std::string filename("VFN-Iteration");
	std::ostringstream converter;
	converter << iteration;
	filename += converter.str() + ".dat";
	VectorField normalized = vectorField.getNormalized();
	normalized.printField(filename.c_str());
}

void Demons::printVFI(VectorField vectorField, int iteration) {
	vectorField.printFieldImage(iteration, compression_params);
}

void Demons::updateDisplField(VectorField displacement, VectorField gradients) {
	int rows = displacement.getRows(), cols = displacement.getCols();
	for(int row = 0; row < rows; row++) {
		uchar* staticRow = staticImage_.ptr(row);
		uchar* deformedRow = deformedImage_.ptr(row);
		for(int col = 0; col < cols; col++) {
			std::vector<float> gradient = gradients.getVectorAt(row, col);

			float diff = (deformedRow[col] - staticRow[col]);
			float denominator = diff*diff + gradient[0]*gradient[0] + gradient[1]*gradient[1];
			if (denominator > 0.0) {
				std::vector<float> displVector = displacement.getVectorAt(row, col);
				float xValue = displVector[0] + gradient[0]*diff/denominator;
				float yValue = displVector[1] + gradient[1]*diff/denominator;
				displacement.updateVector(row, col, xValue, yValue);
			}
		}
	}
}

VectorField Demons::findGrad() {
	int rows = staticImage_.rows, cols = staticImage_.cols;
	cv::Mat sobelX = cv::Mat::zeros(rows, cols, CV_32F);
	cv::Mat sobelY = cv::Mat::zeros(rows, cols, CV_32F);
	cv::Sobel(staticImage_, sobelX, CV_32F, 1, 0);
	cv::Sobel(staticImage_, sobelY, CV_32F, 0, 1);
	sobelX = normalizeSobelImage(sobelX);
	sobelY = normalizeSobelImage(sobelY);
	VectorField grad(sobelX, sobelY);
	return grad;
}

cv::Mat Demons::normalizeSobelImage(cv::Mat sobelImage) {
	double minVal, maxVal;
	minMaxLoc(sobelImage, &minVal, &maxVal); //find minimum and maximum intensities
	cv::Mat normalized;
	sobelImage.convertTo(normalized, CV_32F, 255.0/(maxVal - minVal), -minVal * 255.0/(maxVal - minVal));
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