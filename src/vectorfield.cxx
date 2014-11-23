#define _USE_MATH_DEFINES
#include <exception>
#include <iostream>
#include <fstream>
#include <cmath>
#include <set>

#include "vectorfield.h"
#include "profiler.h"

#define EPSILON 1e-5

VectorField::VectorField(VectorField3D vectorField) {
	std::vector<int> dimensions;
	dimensions.push_back(vectorField.size());
	dimensions.push_back(vectorField[0].size());
	dimensions.push_back(vectorField[0][0].size());
    this->vectorField = vectorField;
}

VectorField::VectorField(std::vector<int> dimensions, float defaultValue) {
	this->dimensions = dimensions;
	vectorField = createVectorField3D(dimensions, defaultValue);
}

VectorField::VectorField3D VectorField::createVectorField3D(std::vector<int> dimensions, float defaultValue) {
	using std::vector;
	return VectorField3D(dimensions[0], vector<vector<vector<float> > >(dimensions[1], vector<vector<float> >(dimensions[2], vector<float>(3, defaultValue))));
}

std::vector<float> VectorField::getVectorAt(int x, int y, int z) {
	if ((x < 0 || x > dimensions[0]-1) || (y < 0 || y > dimensions[1]-1) || (z < 0 || z > dimensions[2]-1)) 
		return std::vector<float>(3, 0.0);
	return vectorField[x][y][z];
}

void VectorField::updateVector(Demon demon, std::vector<float> newValues) {
	std::vector<int> position = demon.getPosition();
	std::vector<int> volumeOfInfluence = demon.getVolumeOfInfluence();
	for (int x = position[0]-volumeOfInfluence[0]; x < position[0]+volumeOfInfluence[0]; x++)
		for (int y = position[1]-volumeOfInfluence[1]; y < position[1]+volumeOfInfluence[1]; y++)
			for (int z = position[2]-volumeOfInfluence[2]; z < position[2]+volumeOfInfluence[2]; z++){
				// std::cout << "x = " << position[0] << "\n";
				// std::cout << "y = " << position[1] << "\n";
				// std::cout << "z = " << position[2] << "\n";
				// std::cout << "volumeOfInfluence[0] = " << volumeOfInfluence[0] << "\n";
				// std::cout << "volumeOfInfluence[1] = " << volumeOfInfluence[1] << "\n";
				// std::cout << "volumeOfInfluence[2] = " << volumeOfInfluence[2] << "\n";
				updateVector(x,y,z, newValues);
			}
}

void VectorField::updateVector(int x, int y, int z, std::vector<float> newValues) {
	vectorField[x][y][z] = newValues;
}

// float VectorField::vectorNorm(std::vector<float> v) {
// 	return sqrt(pow(v[0],2)+pow(v[1],2));
// }

// VectorField VectorField::getNormalized() {
// 	VectorField normalized(rows_, cols_);
// 	for(int row = 0; row < rows_; row++) {
// 	    for(int col = 0; col < cols_; col++) {
// 	    	std::vector<float> vector = vectorField[row][col];
// 	    	float normalizedRow = 0.0, normalizedCol = 0.0;
// 	    	if ((vector[0]*vector[1]) != 0.0) {
// 	    		normalizedRow = vector[0]/vectorNorm(vector);
// 	    		normalizedCol = vector[1]/vectorNorm(vector);
// 	    	}
// 	    	normalized.updateVector(row, col, normalizedRow, normalizedCol);
// 	    }
// 	}
// 	return normalized;
// }

std::vector<float> VectorField::generateGaussianFilter2D(int kernelSize, float deviation) {
	std::vector<float> gaussianFilter(kernelSize, 0.0);
	float sum = 0.0;
	for (int i = 0; i < kernelSize; i++) {
		float firstTerm = (i-(kernelSize-1)/2.0);
		float squaredFirstTerm = std::pow(firstTerm,2);
		float secondTerm = deviation;
		float squaredSecondTerm = 2.0*std::pow(secondTerm,2);
		float finalTerm = -1.0*squaredFirstTerm/squaredSecondTerm;
		gaussianFilter[i] = std::exp(finalTerm);
		sum += gaussianFilter[i];
	}
	for (int i = 0; i < kernelSize; i++)
		gaussianFilter[i] /= sum;
	return gaussianFilter;
}

bool isZero(std::vector<float> vector) {
	if ((vector[0] <= EPSILON) && (vector[1] <= EPSILON) && (vector[2] <= EPSILON)) return true;
	return false;
}

void VectorField::applyGaussianFilter(int kernelSize, float deviation) {
	Profiler profiler("applyGaussianFilter");
	std::vector<float> gaussianKernel = generateGaussianFilter2D(kernelSize, deviation);
	VectorField3D auxVectorField = vectorField;
	for(int x = 0; x < dimensions[0]; x++)
		for(int y = 0; y < dimensions[1]; y++)
			for(int z = 0; z < dimensions[2]; z++) {
				std::vector<float> vector = vectorField[x][y][z];
				if (isZero(vector)) continue;
				std::vector<float> newValues(3, 0.0);
				for (int i = -1; i <= 1; i++) {
					std::vector<float> auxVector = getVectorAt(x+i,y,z);
					newValues[0] += auxVector[0]*gaussianKernel[i+1];
					newValues[1] += auxVector[1]*gaussianKernel[i+1];
					newValues[2] += auxVector[2]*gaussianKernel[i+1];
				}
				auxVectorField[x][y][z] = newValues;
			}
	vectorField = auxVectorField;
	for(int x = 0; x < dimensions[0]; x++)
		for(int y = 0; y < dimensions[1]; y++)
			for(int z = 0; z < dimensions[2]; z++) {
				std::vector<float> vector = vectorField[x][y][z];
				if (isZero(vector)) continue;
				std::vector<float> newValues(3, 0.0);
				for (int i = -1; i <= 1; i++) {
					std::vector<float> auxVector = getVectorAt(x,y+i,z);
					newValues[0] += auxVector[0]*gaussianKernel[i+1];
					newValues[1] += auxVector[1]*gaussianKernel[i+1];
					newValues[2] += auxVector[2]*gaussianKernel[i+1];
				}
				auxVectorField[x][y][z] = newValues;
			}
	vectorField = auxVectorField;
	for(int x = 0; x < dimensions[0]; x++)
		for(int y = 0; y < dimensions[1]; y++)
			for(int z = 0; z < dimensions[2]; z++) {
				std::vector<float> vector = vectorField[x][y][z];
				if (isZero(vector)) continue;
				std::vector<float> newValues(3, 0.0);
				for (int i = -1; i <= 1; i++) {
					std::vector<float> auxVector = getVectorAt(x,y,z+i);
					newValues[0] += auxVector[0]*gaussianKernel[i+1];
					newValues[1] += auxVector[1]*gaussianKernel[i+1];
					newValues[2] += auxVector[2]*gaussianKernel[i+1];
				}
				auxVectorField[x][y][z] = newValues;
			}
	vectorField = auxVectorField;
}

void VectorField::add(VectorField adding) {
	Profiler profiler("add");
	int x, y, z, i;
	for(x = 0; x < dimensions[0]; x++)
		for(y = 0; y < dimensions[1]; y++)
			for(z = 0; z < dimensions[2]; z++) {
				for (i = 0; i < 3; i++)
	        		vectorField[x][y][z][i] = vectorField[x][y][z][i] + adding.vectorField[x][y][z][i];
			}
}

// float VectorField::sumOfAbs() {
// 	float total = 0.0;
// 	for(int row = 0; row < rows_; row++) {
// 		for(int col = 0; col < cols_; col++) {
// 			total += std::abs(vectorCol_.at<float>(row, col)) + std::abs(vectorRow_.at<float>(row, col));
// 		}
// 	}
// 	return total;
// }

void VectorField::printAround(int x, int y, int z) {
	for (int aroundX = x-1; aroundX <= x+1; aroundX++) {
		std::cout << "Plane x = " << aroundX << "\n";
		for (int aroundY = y-1; aroundY <= y+1; aroundY++) {
			std::cout << "\n";
			for (int aroundZ = z-1; aroundZ <= z+1; aroundZ++) {
				std::vector<float> macaco = getVectorAt(aroundX,aroundY,aroundZ);
				std::cout << "(" << macaco[0] << ", " << macaco[1] << ", " << macaco[2] << ")\t";
			}
		}
		std::cout << "\n";
	}
}

// // void VectorField::printFieldImage(int iteration, std::vector<int> compression_params) {
// // 	cv::Mat abs_grad_col, abs_grad_row;
// // 	std::string filenamebase("DFI-Iteration"), flCol, flRow;
// // 	std::ostringstream converter;
// // 	converter << iteration;
// // 	filenamebase += converter.str();
// // 	flCol += filenamebase + "x.jpg";
// // 	flRow += filenamebase + "y.jpg";
// // 	convertScaleAbs(vectorRow_, abs_grad_row, 255);
// // 	convertScaleAbs(vectorCol_, abs_grad_col, 255);
// // 	imwrite(flRow.c_str(), abs_grad_row, compression_params);
// // 	imwrite(flCol.c_str(), abs_grad_col, compression_params);
// // }

// // std::vector<float> VectorField::getInfos() {
// // 	std::multiset<float> magnitudes;
// // 	int size = (rows_*cols_);
// // 	float max= 0.0, min = 0.0, mean = 0.0, median = 0.0, deviation = 0.0;
// 	// for(int row = 0; row < rows_; row++) {
// 	//     for(int col = 0; col < cols_; col++) {
// 	//     	std::vector<float> vector = getVectorAt(row, col);
//  //    		float mag = std::sqrt(vector[0]*vector[0] + vector[1]*vector[1]);
//  //    		if (max < mag) max = mag;
//  //    		if (min > mag) min = mag;
//  //    		mean += mag;
// 	// 		magnitudes.insert(mag);
// 	//     }
// 	// }
// // 	mean /= size;
// // 	int count = 1;
// // 	std::multiset<float>::iterator it;
// // 	for (it=magnitudes.begin(); it!=magnitudes.end(); ++it) {
// //     	deviation += std::pow((*it - mean),2);
// //     	if (count == size/2) median = *it;
// //     	count++;
// // 	}
// // 	deviation /= size;
// // 	deviation = std::sqrt(deviation);
// // 	std::vector<float> results;
// // 	results.push_back(min);
// // 	results.push_back(max);
// // 	results.push_back(median);
// // 	results.push_back(mean);
// // 	results.push_back(deviation);
// // 	return results;
// // }

// // void VectorField::printFieldInfos(std::string filename, int iteration) {
// // 	std::ofstream myfile;
// // 	if (iteration <= 1) myfile.open(filename);
// // 	else myfile.open(filename, std::ios_base::app);
// // 	myfile << "Iteration " << iteration << "\n";
// // 	std::vector<float> results = getInfos();
// // 	myfile << "Min = " << results[0] << " Max = \t" << results[1] << " Median = \t" << results[2] << " Mean = \t" << results[3] << " Standard Deviaon = \t" << results[4] << "\n";
// // 	myfile.close();
// // }

// // void VectorField::printField(std::string filename) {
// // 	std::ofstream myfile;
// // 	myfile.open(filename);
// // 	float minValCol, maxValCol;
// // 	float minValRow, maxValRow;
// // 	minMaxLoc(vectorCol_, &minValCol, &maxValCol);
// // 	minMaxLoc(vectorRow_, &minValRow, &maxValRow);
// // 	for(int row = 0; row < rows_; row++) {
// // 	    for(int col = 0; col < cols_; col++) {
// // 	    	std::vector<float> vector = getVectorAt(row, col);
// //     		float redCol = 255*(vector[1]-minValCol)/(maxValCol-minValCol);
// // 			float blueCol = 255*(maxValCol-vector[1])/(maxValCol-minValCol);
// // 			float redRow = 255*(vector[0]-minValRow)/(maxValRow-minValRow);
// // 			float blueRow = 255*(maxValRow-vector[0])/(maxValRow-minValRow);
// // 			int red = (redCol + redRow)/2;
// // 			int blue = (blueCol + blueRow)/2;
// // 			myfile << col << " " << (vectorCol_.rows - row) << " " << vector[1] << " " << vector[0] << " " <<  red << " " << 0 << " " << blue << "\n";
// // 	    }
// // 	}
// // 	myfile.close();
// // }