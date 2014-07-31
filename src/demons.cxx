
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
	std::vector<double> norm(10,0.0);
	VectorField gradients = findGrad();
	gradients.printField("Gradients.dat");
	VectorField displField(rows, cols);
	VectorField deltaField(rows, cols);
	int iteration = 1;
	compression_params.push_back(CV_IMWRITE_JPEG_QUALITY);
	compression_params.push_back(95);
	do {
		time(&startTime);
		deltaField = newDeltaField(gradients);
		deltaField.applyGaussianFilter();
		updateDisplField(displField, deltaField);
		// displField.applyGaussianFilter();
		updateDeformedImage(displField);
		double iterTime = getIterationTime(startTime);
		printVFN(displField, iteration);
		printVFI(displField, iteration);
		printDeformedImage(iteration);
		std::cout << "Iteration " << iteration << " took " << iterTime << " seconds.\n";
		iteration++;
	} while(stopCriteria(norm, displField, deltaField));
	std::cout << "termino rapa\n";
}

bool Demons::stopCriteria(std::vector<double> &norm, VectorField displField, VectorField deltaField) {
	double newNorm = deltaField.sumOfAbs()/displField.sumOfAbs();
	std::cout << (newNorm - norm[9]) << "\n";
	if (std::abs((newNorm - norm[9])) > 0.0001) {
		for (int i = 9; i > 0; i--) norm[i] = norm[i-1];
		norm[0] = newNorm;
		return true;
	}
	return false;
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

void Demons::updateDisplField(VectorField displField, VectorField deltaField) {
	displField.add(deltaField);
}

VectorField Demons::newDeltaField(VectorField gradients) {
	int rows = gradients.getRows(), cols = gradients.getCols();
	VectorField deltaField(rows, cols);
	for(int row = 0; row < rows; row++) {
		uchar* staticRow = staticImage_.ptr(row);
		uchar* deformedRow = deformedImage_.ptr(row);
		for(int col = 0; col < cols; col++) {
			std::vector<float> gradient = gradients.getVectorAt(row, col);

			float diff = (deformedRow[col] - staticRow[col]);
			float denominator = diff*diff + gradient[0]*gradient[0] + gradient[1]*gradient[1];
			if (denominator > 0.0) {
				float xValue = gradient[0]*diff/denominator;
				float yValue = gradient[1]*diff/denominator;
				deltaField.updateVector(row, col, xValue, yValue);
			}
		}
	}
	return deltaField;
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

double Demons::getIterationTime(time_t startTime) {
	double iterationTime = difftime(time(NULL), startTime);
	totalTime += iterationTime;
	return iterationTime;
}

cv::Mat Demons::getRegistration() {
	return deformedImage_;
}