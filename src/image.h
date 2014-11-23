#ifndef IMAGE_H_
#define IMAGE_H_

#include <string>
#include <memory>
#include <vector>
#include <iostream>

#include "vectorfield.h"

template <class T>
class Image {
public:
	typedef std::vector<std::vector<std::vector<T> > > ImageVoxels;
	typedef std::vector<std::vector<T> > ImagePixels;
	Image(std::vector<int> dims) :
		dimensions(dims) {
			imageData_ = createCube(dims[0], dims[1], dims[2]);
		}
	T& operator() (const size_t& i, const size_t& j, const size_t& k);
	std::vector<int> getDimensions() {return dimensions;}
	int getDim() {return dimensions.size();}
	T getPixelAt(float x, float y, float z);
	T getPixelAt(int x, int y, int z);
	void printAround(int x, int y, int z);
	VectorField getGradient();
private:
	ImageVoxels imageData_;
	std::vector<int> dimensions;
	ImageVoxels createCube(size_t dim0, size_t dim1, size_t dim2);
};

template <class T>
typename Image<T>::ImageVoxels Image<T>::createCube(size_t dim0, size_t dim1, size_t dim2) {
	using std::vector;
	return ImageVoxels(dim0, vector<vector<T> >(dim1, vector<T>(dim2)));
}

template <class T>
T& Image<T>::operator() (const size_t& i, const size_t& j, const size_t& k) {
	return imageData_[i][j][k];
}

template <class T>
T Image<T>::getPixelAt (int x, int y, int z) {
	T value = T();
	int nearX = x, nearY = y, nearZ = z;
	if (x < 0) nearX = 0;
	if (x >= dimensions[0]) nearX = dimensions[0]-1;
	if (y < 0) nearY = 0;
	if (y >= dimensions[1]) nearY = dimensions[1]-1;
	if (z < 0) nearZ = 0;
	if (z >= dimensions[2]) nearZ = dimensions[2]-1;
	value = imageData_[nearX][nearY][nearZ];
	return value;
}

template <class T>
void Image<T>::printAround(int x, int y, int z) {
	for (int aroundX = x-1; aroundX <= x+1; aroundX++) {
		std::cout << "Plane x = " << aroundX << "\n";
		for (int aroundY = y-1; aroundY <= y+1; aroundY++) {
			std::cout << "\n";
			for (int aroundZ = z-1; aroundZ <= z+1; aroundZ++) {
				T macaco = getPixelAt(aroundX,aroundY,aroundZ);
				std::cout << macaco << "\t";
			}
		}
		std::cout << "\n";
	}
}

template <class T>
VectorField Image<T>::getGradient() {
	VectorField gradient(dimensions, 0.0);
	for (int x = 0; x < dimensions[0]; x++)
		for (int y = 0; y < dimensions[1]; y++)
			for (int z = 0; z < dimensions[2]; z++) {
				std::vector<float> gradVector(3, 0.0);
				gradVector[0] += getPixelAt(x-1,y,z)*(-0.5);
				gradVector[0] += getPixelAt(x+1,y,z)*(0.5);
				gradVector[1] += getPixelAt(x,y-1,z)*(-0.5);
				gradVector[1] += getPixelAt(x,y+1,z)*(0.5);
				gradVector[2] += getPixelAt(x,y,z-1)*(-0.5);
				gradVector[2] += getPixelAt(x,y,z+1)*(0.5);
				gradient.updateVector(x, y, z, gradVector);
			}
	return gradient;
}

#endif