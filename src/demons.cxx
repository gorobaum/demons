#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <ctime>
#include <vector>
#include <sstream>

#include "demons.h"

#define RMSEcriteria 10
#define CORRCOEFcriteria 0.95
#define STOPcriteria 0.0001
#define POSX 128
#define POSY 128
#define POSZ 64

void Demons::run() {
	VectorField staticGradient = staticImage_.getGradient();
	VectorField deltaField(dimensions, 0.0);

	for (int iteration = 1; iteration <= numOfIterations_; iteration++) {
		std::cout << "Iteration " << iteration << "\n";
		deltaField = newDeltaField(staticGradient);
		updateDisplField(displField, deltaField);
		debug(iteration, deltaField, staticGradient);
	}
	std::cout << "termino rapa\n";
}

double Demons::getDeformedImageValueAt(int x, int y, int z) {
    std::vector<double> displVector = displField.getVectorAt(x, y, z);
    double newX = x - displVector[0];
    double newY = y - displVector[1];
    double newZ = z - displVector[1];
    return movingInterpolator.trilinearInterpolation<double>(newX, newY, newZ);
}

void Demons::debug(int iteration, VectorField deltaField, VectorField gradients) {
	// ImageFunctions::printAround(staticImage_, POSR, POSC);
	// ImageFunctions::printAround(movingImage_, POSR, POSC);
	// gradients.printAround(POSX,POSY,POSZ);
	// staticImage_.printAround(POSX,POSY,POSZ);
	// deltaField.printFieldAround(POSR,POSC);
	// displField.printFieldAround(POSR,POSC);

	// std::string filename("VFN-Iteration");
	// std::ostringstream converter;
	// converter << iteration;
	// filename += converter.str() + ".dat";
	// VectorField normalized = displField.getNormalized();
	// normalized.printField(filename.c_str());
}

void Demons::updateDisplField(VectorField &displField, VectorField &deltaField) {
	displField.add(deltaField);
	displField.applyGaussianFilter(gauKernelSize_, gauDeviation_);
}

VectorField Demons::getDisplField() {
	return displField;
}