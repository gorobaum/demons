
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
	deformedImage_ = cv::Mat::zeros(rows, cols, CV_LOAD_IMAGE_GRAYSCALE);
	VectorField gradients = findGrad();
	printDisplField(gradients, 0);
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
		printDisplField(displField, iteration);
		std::string imageName("Iteration");
		std::ostringstream converter;
		converter << iteration;
		imageName += converter.str() + ".jpg";
		std::cout << imageName.c_str() << "\n";
		std::cout << "Iteration " << converter.str() << " took " << iterTime << " seconds.\n";
    	iteration++;
        imwrite(imageName.c_str(), deformedImage_, compression_params);
	}
	std::cout << "termino rapa\n";
}

void Demons::printDisplField(VectorField vectorField, int iteration) {
	std::string filename("VectorField");
	std::ostringstream converter;
	converter << iteration;
	filename += converter.str() + ".dat";
	VectorField normalized = vectorField.getNormalized();
	normalized.printField(filename.c_str());
}

void Demons::updateDisplField(VectorField displacement, std::vector<float> gradient, int row, int col) {
	uchar* staticRow = staticImage_.ptr(row);
	uchar* deformedRow = deformedImage_.ptr(row);
	float diff = (staticRow[col] - deformedRow[col]);
	float denominator = diff*diff + gradient[0]*gradient[0] + gradient[1]*gradient[1];
	if (denominator > 0.0) {
		std::vector<float> displVector = displacement.getVectorAt(row, col);
		float xValue = displVector[0] + gradient[0]*diff/denominator;
		float yValue = displVector[1] + gradient[1]*diff/denominator;
		displacement.updateVector(row, col, xValue, yValue);
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