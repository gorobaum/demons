#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <ctime>
#include <vector>
#include <sstream>

#include "demons.h"

#define SPACING 1
#define RMSEcriteria 10
#define CORRCOEFcriteria 0.95
#define STOPcriteria 0.0001
#define POSR 20
#define POSC 20

void Demons::run() {
	VectorField staticGradient = staticImage_.getGradient();
	VectorField deltaField(dimensions, 0.0);

	for(int iteration = 1; iteration <= 50; iteration++) {
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
    return movingInterpolator.NNInterpolation<double>(newX, newY, newZ);
}

void Demons::debug(int iteration, VectorField deltaField, VectorField gradients) {
	// ImageFunctions::printAround(staticImage_, POSR, POSC);
	// ImageFunctions::printAround(movingImage_, POSR, POSC);
	// gradients.printFieldAround(POSR,POSC);
	// deltaField.printFieldAround(POSR,POSC);
	// displField.printFieldAround(POSR,POSC);

	// std::string filename("VFN-Iteration");
	// std::ostringstream converter;
	// converter << iteration;
	// filename += converter.str() + ".dat";
	// VectorField normalized = displField.getNormalized();
	// normalized.printField(filename.c_str());
}

void Demons::updateDisplField(VectorField &displField, VectorField deltaField) {
	displField.add(deltaField);
	displField.applyGaussianFilter(3, 1);
}

VectorField Demons::getDisplField() {
	return displField;
}