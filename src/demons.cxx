
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <ctime>
#include <vector>
#include <sstream>
#include <opencv2/imgproc/imgproc.hpp>

#include "demons.h"

#define RMSEcriteria 10
#define CORRCOEFcriteria 0.95
#define STOPcriteria 0.0001
#define SPACING 0.16
#define POSR 130
#define POSC 131

void Demons::demons() {
	int rows = staticImage_.rows, cols = staticImage_.cols;
	// Create the deformed image
	movingImage_.convertTo(deformedImage_, CV_32F, 1);
	std::vector<float> norm(10,0.0);
	Gradient staticGradient(staticImage_);
	Gradient deformedImageGradient(deformedImage_);
	VectorField gradients = staticGradient.getSobelGradient();
	gradients.printField("Gradients.dat");
	std::string gfName("GradientInformation.info");
	gradients.printFieldInfos(gfName, 1);
	VectorField deltaField(rows, cols);
	for(int iteration = 1; iteration <= 50; iteration++) {
		deltaField = newDeltaField(gradients, deformedImageGradient);
		// if(iteration != 1 && stopCriteria(norm, displField, deltaField)) break;
		updateDisplField(displField, deltaField);
		updateDeformedImage(displField);
		std::cout << "Iteration " << iteration << "\n";
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
	int rows = deformedImage_.rows, cols = deformedImage_.cols;
	double rmse = 0.0;
	for(int row = 0; row < rows; row++) {
		const float* dRow = deformedImage_.ptr<float>(row);
		uchar* mRow = staticImage_.ptr(row);
		for(int col = 0; col < cols; col++) {
			rmse += (dRow[col]-mRow[col])*(dRow[col]-mRow[col]);
		}
	}
	rmse = std::sqrt(rmse/(rows*cols));
	return rmse < RMSEcriteria;
}

bool Demons::stopCriteria(std::vector<float> &norm, VectorField displField, VectorField deltaField) {
	float newNorm = deltaField.sumOfAbs()/displField.sumOfAbs();
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
		float* deformedImageRow = deformedImage_.ptr<float>(row);
		for(int col = 0; col < cols; col++) {
			std::vector<float> displVector = displField.getVectorAt(row, col);
			double newRow = row - displVector[0];
			double newCol = col - displVector[1];
			deformedImageRow[col] = movingInterpolator.bilinearInterpolation<float>(newRow, newCol);
		}
	}
}

void Demons::updateDisplField(VectorField displField, VectorField deltaField) {
	// deltaField.applyGaussianFilter();
	displField.add(deltaField);
	displField.applyGaussianFilter();
}

VectorField Demons::newDeltaField(VectorField gradients, Gradient deformedImageGradient) {
	int rows = gradients.getRows(), cols = gradients.getCols();
	VectorField deltaField(rows, cols);
	VectorField gradientDeformed = deformedImageGradient.getSobelGradient();
	for(int row = 0; row < rows; row++) {
		const float* dRow = deformedImage_.ptr<float>(row);
		uchar* sRow = staticImage_.ptr(row);
		for(int col = 0; col < cols; col++) {
			std::vector<float> sGrad = gradients.getVectorAt(row, col);
			std::vector<float> dGrad = gradientDeformed.getVectorAt(row, col);
			float diff = dRow[col] - sRow[col];
			// float k = std::sqrt(SPACING);
			float denominator = (diff*diff) + (sGrad[0]+dGrad[0])*(sGrad[0]+dGrad[0]) + (sGrad[1]+dGrad[1])*(sGrad[1]+dGrad[1]);
			if (denominator > 0.0) {
				float rowValue = 2*(sGrad[0]+dGrad[0])*diff/denominator;
				float colValue = 2*(sGrad[1]+dGrad[1])*diff/denominator;
				deltaField.updateVector(row, col, rowValue, colValue);
			}
		}
	}
	return deltaField;
}

cv::Mat Demons::getRegistration() {
	return deformedImage_;
}