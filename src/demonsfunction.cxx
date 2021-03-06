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
	for (int pSize = 1; pSize <= pyramidSize_; pSize++) {
		createDemons(pSize);
		for (int iteration = 1; iteration <= numOfIterations_; iteration++) {
			Profiler profiler("loop");
			std::cout << "Iteration " << iteration << "\n";
			deltaField = newDeltaField(staticGradient);
			updateDisplField(displField, deltaField);
			// debug(iteration, deltaField, staticGradient);
		}
	}
	createDemons(0);
	for (int iteration = 1; iteration <= numOfIterations_; iteration++) {
		Profiler profiler("loop");
		std::cout << "Iteration " << iteration << "\n";
		deltaField = newDeltaField(staticGradient);
		updateDisplField(displField, deltaField);
		debug(iteration, deltaField, staticGradient);
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

std::vector<int> DemonsFunction::calculateVolumeOfInfluence(int x, int y, int z, int xStep, int yStep, int zStep) {
	std::vector<int> areaOfInfluence(3, 1);
	if (x+xStep >= dimensions[0]) areaOfInfluence[0] = (dimensions[0]-x)/2;
	else areaOfInfluence[0] = (xStep/2 == 0 ? 1 : xStep/2);
	if (y+yStep >= dimensions[1]) areaOfInfluence[1] = (dimensions[1]-y)/2;
	else areaOfInfluence[1] = (yStep/2 == 0 ? 1 : yStep/2);
	if (z+zStep >= dimensions[2]) areaOfInfluence[2] = (dimensions[2]-z)/2;
	else areaOfInfluence[2] = (zStep/2 == 0 ? 1 : zStep/2);
	return areaOfInfluence;
}

void DemonsFunction::createDemons(int currentPyramidLevel) {
	int xStep = 1, yStep = 1, zStep = 1;
	if (currentPyramidLevel > 0) {
		int divisionFactor = std::pow(2, currentPyramidLevel);
		xStep = dimensions[0]/divisionFactor;
		yStep = dimensions[1]/divisionFactor;
		zStep = dimensions[2]/divisionFactor;
		// std::cout << "divisionFactor = " << divisionFactor << "\n";
		// std::cout << "xStep = " << xStep << "\n";
		// std::cout << "yStep = " << yStep << "\n";
		// std::cout << "zStep = " << zStep << "\n";
	}
	for (int x = 0; x < dimensions[0]; x+=xStep)
		for (int y = 0; y < dimensions[1]; y+=yStep)
			for (int z = 0; z < dimensions[2]; z+=zStep) {
				std::vector<int> volumeOfInfluence = calculateVolumeOfInfluence(x,y,z,xStep,yStep,zStep);
				std::vector<int> position;
				position.push_back(x+volumeOfInfluence[0]);
				position.push_back(y+volumeOfInfluence[1]);
				position.push_back(z+volumeOfInfluence[2]);
				// std::cout << "(x,y,z) = (" << x << "," << y << "," << z << ")\n";
				// std::cout << "VoI(x,y,z) = (" << volumeOfInfluence[0] << "," << volumeOfInfluence[1] << "," << volumeOfInfluence[2] << ")\n";
				Demon demon(position, volumeOfInfluence);
				demons.push_back(demon);
			}
}

void DemonsFunction::debug(int iteration, VectorField deltaField, VectorField gradients) {
	// ImageFunctions::printAround(staticImage_, POSR, POSC);
	// ImageFunctions::printAround(movingImage_, POSR, POSC);
	// gradients.printAround(POSX,POSY,POSZ);
	// staticImage_.printAround(POSX,POSY,POSZ);
	// deltaField.printAround(POSX,POSY,POSZ);
	// displField.printAround(POSX,POSY,POSZ);

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