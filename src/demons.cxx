
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <ctime>
#include <vector>
#include <sstream>
#include <opencv2/imgproc/imgproc.hpp>

#include "demons.h"

#define SPACING 1
#define IMAGEDIMENSION 2
#define RMSEcriteria 10
#define CORRCOEFcriteria 0.95
#define STOPcriteria 0.0001
#define POSR 20
#define POSC 20

void Demons::run() {
	Gradient staticGradient(staticImage_);
	VectorField gradients = staticGradient.getBasicGradient();
	VectorField deltaField(rows, cols);
	int dimensions;
	for (dimensions = 0; dimensions < IMAGEDIMENSION; dimensions++) normalizer = SPACING*SPACING;
	normalizer /= dimensions;

	for(int iteration = 1; iteration <= 50; iteration++) {
		deltaField = newDeltaField(gradients);
		updateDisplField(displField, deltaField);
		debug(iteration, deltaField, gradients);
		std::cout << "Iteration " << iteration << "\n";
	}
	std::cout << "termino rapa\n";
}

double Demons::getDeformedImageValueAt(int row, int col) {
    std::vector<double> displVector = displField.getVectorAt(row, col);
    double newRow = row - displVector[0];
    double newCol = col - displVector[1];
    return movingInterpolator.bilinearInterpolation<double>(newRow, newCol);
}

void Demons::debug(int iteration, VectorField deltaField, VectorField gradients) {
	// ImageFunctions::printAround(staticImage_, POSR, POSC);
	// ImageFunctions::printAround(movingImage_, POSR, POSC);
	// gradients.printFieldAround(POSR,POSC);
	// deltaField.printFieldAround(POSR,POSC);
	// displField.printFieldAround(POSR,POSC);

	std::string filename("VFN-Iteration");
	std::ostringstream converter;
	converter << iteration;
	filename += converter.str() + ".dat";
	VectorField normalized = displField.getNormalized();
	normalized.printField(filename.c_str());
}

void Demons::updateDisplField(VectorField displField, VectorField deltaField) {
	displField.add(deltaField);
	displField.applyGaussianFilter(3, 1);
}

VectorField Demons::getDisplField() {
	return displField;
}