
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
	Gradient staticGradient(staticImage_);
	Gradient deformedImageGradient(deformedImage_);
	VectorField gradients = staticGradient.getSobelGradient();
	VectorField deltaField(rows, cols);

	for(int iteration = 1; iteration <= 50; iteration++) {
		deltaField = newDeltaField(gradients, deformedImageGradient);
		updateDisplField(displField, deltaField);
		updateDeformedImage(displField);
		std::cout << "Iteration " << iteration << "\n";
	}
	std::cout << "termino rapa\n";
}

void Demons::updateDeformedImage(VectorField displField) {
	for(int row = 0; row < rows; row++) {
		double* deformedImageRow = deformedImage_.ptr<double>(row);
		for(int col = 0; col < cols; col++) {
			std::vector<double> displVector = displField.getVectorAt(row, col);
			double newRow = row - displVector[0];
			double newCol = col - displVector[1];
			deformedImageRow[col] = movingInterpolator.bilinearInterpolation<double>(newRow, newCol);
		}
	}
}

void Demons::updateDisplField(VectorField displField, VectorField deltaField) {
	displField.add(deltaField);
	displField.applyGaussianFilter();
}

cv::Mat Demons::getRegistration() {
	return deformedImage_;
}