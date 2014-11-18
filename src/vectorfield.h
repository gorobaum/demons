#ifndef DEMONS_VECTORFIELD_H_
#define DEMONS_VECTORFIELD_H_

#include <vector>

#include "demon.h"

class VectorField {
	public:
		typedef std::vector<std::vector<std::vector<std::vector<double> > > > VectorField3D;
		VectorField(VectorField3D vectorField);
		VectorField(std::vector<int> dimensions, double defaultValue);
		VectorField::VectorField3D createVectorField3D(std::vector<int> dimensions, double defaultValue);
		std::vector<double> getVectorAt(int x, int y, int z);
		void updateVector(Demon demon, std::vector<double> newValues);
		void updateVector(int x, int y, int z, std::vector<double> newValues);
		void applyGaussianFilter(int kernelSize, double deviation);
		std::vector<double> generateGaussianFilter2D(int kernelSize, double deviation);
		VectorField getNormalized();
		void printAround(int x, int y, int z);
		void printField(std::string filename);
		void printFieldInfos(std::string filename, int iteration);
		void printFieldImage(int iteration, std::vector<int> compression_params);
		void add(VectorField adding);
		double sumOfAbs();
		std::vector<int> getDimensions() {return dimensions;}
	private:
		VectorField3D vectorField;
		double vectorNorm(std::vector<double> v);
		std::vector<double> getInfos();
		std::vector<double> zeroVector = std::vector<double>(3, 0.0);
		std::vector<int> dimensions;
};

#endif
