
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <ctime>
#include <vector>
#include <opencv2/imgproc/imgproc.hpp>
#include <sstream>

#include "demons.h"
#include "interpolation.h"

#define RMSEcriteria 10
#define CORRCOEFcriteria 0.95
#define STOPcriteria 0.0001

void Demons::demons() {
	int rows = staticImage_.rows, cols = staticImage_.cols;
	// Create the deformed image
	deformedImage_ = movingImage_.clone();
	std::vector<float> norm(10,0.0);
	VectorField gradients = findGrad();
	gradients.printField("Gradients.dat");
	VectorField displField(rows, cols);
	VectorField deltaField(rows, cols);
	int iteration = 1;
	compression_params.push_back(CV_IMWRITE_JPEG_QUALITY);
	compression_params.push_back(95);
	while(1) {
		time(&startTime);
		deltaField = newDeltaField(gradients);
		if(iteration != 1 && stopCriteria(norm, displField, deltaField)) break;
		updateDisplField(displField, deltaField);
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

bool Demons::correlationCoef() {
	cv::MatND staticImageHist, deformedImageHist;
	int channels[] = {0};
	int histSize = 256;
    float range[] = { 0, 255 };
    const float* ranges[] = { range };
	calcHist(&staticImage_, 1, channels, cv::Mat(), staticImageHist, 1, &histSize, ranges);
	calcHist(&deformedImage_, 1, channels, cv::Mat(), deformedImageHist, 1, &histSize, ranges);
	// std::cout << cv::compareHist(staticImageHist, deformedImageHist, CV_COMP_CORREL) << "\n";
	return cv::compareHist(staticImageHist, deformedImageHist, CV_COMP_CORREL) >= CORRCOEFcriteria;
}

bool Demons::rootMeanSquareError() {
	cv::Mat diff;
	diff = deformedImage_.clone();
	diff = diff - staticImage_;
	int rows = diff.rows, cols = diff.cols;
	double rmse = 0.0;
	for(int row = 0; row < rows; row++) {
		uchar* diffRow = diff.ptr(row);
		for(int col = 0; col < cols; col++) {
			rmse += diffRow[col]*diffRow[col];
		}
	}
	rmse = std::sqrt(rmse/(rows*cols));
	return rmse < RMSEcriteria;
}

bool Demons::stopCriteria(std::vector<float> &norm, VectorField displField, VectorField deltaField) {
	float newNorm = deltaField.sumOfAbs()/displField.sumOfAbs();
	// for (int i = 0; i < 10; i++) std::cout << "norm[" << i << "] = " << norm[i] << "\n";
	// std::cout << "newNorm = " << newNorm << "\n";
	// std::cout << "newNorm - norm = " << std::abs((newNorm - norm[9])) << "\n";
	if (std::abs((newNorm - norm[9])) > STOPcriteria) {
		for (int i = 9; i >= 0; i--) norm[i] = norm[i-1];
		norm[0] = newNorm;
		return false;
	}
	return true;
}

void Demons::updateDeformedImage(VectorField displField) {
	int rows = displField.getRows(), cols = displField.getCols();
	for(int row = 0; row < rows; row++) {
		uchar* deformedImageRow = deformedImage_.ptr(row);
		for(int col = 0; col < cols; col++) {
			std::vector<float> displVector = displField.getVectorAt(row, col);
			float newRow = row + displVector[0];
			float newCol = col + displVector[1];
			deformedImageRow[col] = Interpolation::bilinearInterpolation(movingImage_, newRow, newCol);
			// if (displVector[0] != 0 || displVector[1] != 0) {
			// 	std::cout << "row = " << row << " col = " << col << "\n";
			// 	std::cout << "displVector[0] = " << displVector[0] << " displVector[1] = " << displVector[1] << "\n";
			// 	std::cout << "pixel = " << movingImage_.at<uchar>(row,col) << "\n";
			// 	std::cout << "newRow = " << newRow << " newCol = " << newCol << "\n";
			// 	std::cout << "pixel = " << deformedImage_.at<uchar>(row,col) << "\n";
			// }
		}
	}
}

void Demons::updateDisplField(VectorField displField, VectorField deltaField) {
	// deltaField.applyGaussianFilter();
	displField.add(deltaField);
	// displField.applyGaussianFilter();
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
				float rowValue = gradient[0]*diff/denominator;
				float colValue = gradient[1]*diff/denominator;
				deltaField.updateVector(row, col, rowValue, colValue);
			}
		}
	}
	return deltaField;
}

VectorField Demons::findGrad() {
	int rows = staticImage_.rows, cols = staticImage_.cols;
	cv::Mat sobelRow = cv::Mat::zeros(rows, cols, CV_32F);
	cv::Mat sobelCol = cv::Mat::zeros(rows, cols, CV_32F);
	cv::Sobel(staticImage_, sobelRow, CV_32F, 0, 1);
	cv::Sobel(staticImage_, sobelCol, CV_32F, 1, 0);
	// sobelCol = normalizeSobelImage(sobelCol);
	// sobelRow = normalizeSobelImage(sobelRow);
	VectorField grad(sobelRow, sobelCol);
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
    imwrite(imageName.c_str(), deformedImage_, compression_params);
}

void Demons::printVFN(VectorField vectorField, int iteration) {
	std::string filename("VFN-Iteration");
	std::ostringstream converter;
	converter << iteration;
	filename += converter.str() + ".dat";
	VectorField normalized = vectorField.getNormalized();
	normalized.printField(filename.c_str());
	std::string vfName("VectorFieldInformation.info");
	vectorField.printFieldInfos(vfName, iteration);
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