#ifndef DEMONS_VECTORFIELD_H_
#define DEMONS_VECTORFIELD_H_

#include <vector>

#include "demon.h"

class VectorField {
	public:
		typedef std::vector<std::vector<std::vector<std::vector<float> > > > VectorField3D;
		VectorField(VectorField3D vectorField);
		VectorField(std::vector<int> dimensions, float defaultValue);
		VectorField::VectorField3D createVectorField3D(std::vector<int> dimensions, float defaultValue);
		std::vector<float> getVectorAt(int x, int y, int z);
		void updateVector(Demon demon, std::vector<float> newValues);
		void updateVector(int x, int y, int z, std::vector<float> newValues);
		void applyGaussianFilter(int kernelSize, float deviation);
		std::vector<float> generateGaussianFilter2D(int kernelSize, float deviation);
		VectorField getNormalized();
		void printAround(int x, int y, int z);
		void printField(std::string filename);
		void printFieldInfos(std::string filename, int iteration);
		void printFieldImage(int iteration, std::vector<int> compression_params);
		void add(VectorField adding);
		float sumOfAbs();
		std::vector<int> getDimensions() {return dimensions;}
	private:
		VectorField3D vectorField;
		float vectorNorm(std::vector<float> v);
		std::vector<float> getInfos();
		std::vector<float> zeroVector = std::vector<float>(3, 0.0);
		std::vector<int> dimensions;
};

#endif
