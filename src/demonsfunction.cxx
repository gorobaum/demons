#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <ctime>
#include <vector>
#include <sstream>

#include "demonsfunction.h"
#include "profiler.h"

#define RMSEcriteria 10
#define CORRCOEFcriteria 0.95
#define STOPcriteria 0.0001
#define POSX 128
#define POSY 128
#define POSZ 64

void DemonsFunction::run() {
	VectorField staticGradient = staticImage_.getGradient();
	VectorField deltaField(dimensions, 0.0);

	for (int iteration = 1; iteration <= numOfIterations_; iteration++) {
		Profiler profiler("loop");
		std::cout << "Iteration " << iteration << "\n";
		deltaField = newDeltaField(staticGradient);
		updateDisplField(displField, deltaField);
		// debug(iteration, deltaField, staticGradient);
	}
	std::cout << "termino rapa\n";
}

double DemonsFunction::getDeformedImageValueAt(int x, int y, int z) {
    std::vector<double> displVector = displField.getVectorAt(x, y, z);
    double newX = x - displVector[0];
    double newY = y - displVector[1];
    double newZ = z - displVector[1];
    return movingInterpolator.trilinearInterpolation<double>(newX, newY, newZ);
}

void DemonsFunction::setExecutionParameters(int numOfIterations, int pyramidSize) {
	numOfIterations_ = numOfIterations;
	pyramidSize_ = pyramidSize;
}

void DemonsFunction::setGaussianParameters(double gauKernelSize, double gauDeviation) {
	gauKernelSize_ = gauKernelSize;
	gauDeviation_ = gauDeviation;
}

void DemonsFunction::debug(int iteration, VectorField deltaField, VectorField gradients) {
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

void DemonsFunction::updateDisplField(VectorField &displField, VectorField &deltaField) {
	displField.add(deltaField);
	displField.applyGaussianFilter(gauKernelSize_, gauDeviation_);
}

VectorField DemonsFunction::getDisplField() {
	return displField;
}