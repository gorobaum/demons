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

VectorField::VectorField(std::vector<int> dimensions, double defaultValue) {
	this->dimensions = dimensions;
	vectorField = createVectorField3D(dimensions, defaultValue);
}

VectorField::VectorField3D VectorField::createVectorField3D(std::vector<int> dimensions, double defaultValue) {
	using std::vector;
	return VectorField3D(dimensions[0], vector<vector<vector<double> > >(dimensions[1], vector<vector<double> >(dimensions[2], vector<double>(3, defaultValue))));
}

std::vector<double> VectorField::getVectorAt(int x, int y, int z) {
	if ((x < 0 || x > dimensions[0]-1) || (y < 0 || y > dimensions[1]-1) || (z < 0 || z > dimensions[2]-1)) 
		return std::vector<double>(3, 0.0);
	return vectorField[x][y][z];
}

void VectorField::updateVector(int x, int y, int z, std::vector<double> newValues) {
	vectorField[x][y][z] = newValues;
}

// double VectorField::vectorNorm(std::vector<double> v) {
// 	return sqrt(pow(v[0],2)+pow(v[1],2));
// }

// VectorField VectorField::getNormalized() {
// 	VectorField normalized(rows_, cols_);
// 	for(int row = 0; row < rows_; row++) {
// 	    for(int col = 0; col < cols_; col++) {
// 	    	std::vector<double> vector = vectorField[row][col];
// 	    	double normalizedRow = 0.0, normalizedCol = 0.0;
// 	    	if ((vector[0]*vector[1]) != 0.0) {
// 	    		normalizedRow = vector[0]/vectorNorm(vector);
// 	    		normalizedCol = vector[1]/vectorNorm(vector);
// 	    	}
// 	    	normalized.updateVector(row, col, normalizedRow, normalizedCol);
// 	    }
// 	}
// 	return normalized;
// }

std::vector<double> VectorField::generateGaussianFilter2D(int kernelSize, double deviation) {
	std::vector<double> gaussianFilter(kernelSize, 0.0);
	double sum = 0.0;
	for (int i = 0; i < kernelSize; i++) {
		double firstTerm = (i-(kernelSize-1)/2.0);
		double squaredFirstTerm = std::pow(firstTerm,2);
		double secondTerm = deviation;
		double squaredSecondTerm = 2.0*std::pow(secondTerm,2);
		double finalTerm = -1.0*squaredFirstTerm/squaredSecondTerm;
		gaussianFilter[i] = std::exp(finalTerm);
		sum += gaussianFilter[i];
	}
	for (int i = 0; i < kernelSize; i++)
		gaussianFilter[i] /= sum;
	return gaussianFilter;
}

bool isZero(std::vector<double> vector) {
	if ((vector[0] <= EPSILON) && (vector[1] <= EPSILON) && (vector[2] <= EPSILON)) return true;
	return false;
}

void VectorField::applyGaussianFilter(int kernelSize, double deviation) {
	std::vector<double> gaussianKernel = generateGaussianFilter2D(kernelSize, deviation);
	VectorField3D auxVectorField = vectorField;
	int x, y, z;
	int size = dimensions[0]*dimensions[1]*dimensions[2]/4;
	#pragma omp parallel for collapse(3) schedule(dynamic, size)
	for(x = 0; x < dimensions[0]; x++)
		for(y = 0; y < dimensions[1]; y++)
			for(z = 0; z < dimensions[2]; z++) {
				std::vector<double> vector = vectorField[x][y][z];
				if (isZero(vector)) continue;
				std::vector<double> newValues(3, 0.0);
				for (int i = -1; i <= 1; i++) {
					std::vector<double> auxVector = getVectorAt(x+i,y,z);
					newValues[0] += auxVector[0]*gaussianKernel[i+1];
					newValues[1] += auxVector[1]*gaussianKernel[i+1];
					newValues[2] += auxVector[2]*gaussianKernel[i+1];
				}
				auxVectorField[x][y][z] = newValues;
			}
	vectorField = auxVectorField;
	#pragma omp parallel for collapse(3) schedule(dynamic, size)
	for(x = 0; x < dimensions[0]; x++)
		for(y = 0; y < dimensions[1]; y++)
			for(z = 0; z < dimensions[2]; z++) {
				std::vector<double> vector = vectorField[x][y][z];
				if (isZero(vector)) continue;
				std::vector<double> newValues(3, 0.0);
				for (int i = -1; i <= 1; i++) {
					std::vector<double> auxVector = getVectorAt(x,y+i,z);
					newValues[0] += auxVector[0]*gaussianKernel[i+1];
					newValues[1] += auxVector[1]*gaussianKernel[i+1];
					newValues[2] += auxVector[2]*gaussianKernel[i+1];
				}
				auxVectorField[x][y][z] = newValues;
			}
	vectorField = auxVectorField;
	#pragma omp parallel for collapse(3) schedule(dynamic, size)
	for(x = 0; x < dimensions[0]; x++)
		for(y = 0; y < dimensions[1]; y++)
			for(z = 0; z < dimensions[2]; z++) {
				std::vector<double> vector = vectorField[x][y][z];
				if (isZero(vector)) continue;
				std::vector<double> newValues(3, 0.0);
				for (int i = -1; i <= 1; i++) {
					std::vector<double> auxVector = getVectorAt(x,y,z+i);
					newValues[0] += auxVector[0]*gaussianKernel[i+1];
					newValues[1] += auxVector[1]*gaussianKernel[i+1];
					newValues[2] += auxVector[2]*gaussianKernel[i+1];
				}
				auxVectorField[x][y][z] = newValues;
			}
	vectorField = auxVectorField;
}

void VectorField::add(VectorField adding) {
	int x, y, z, i;
	int size = dimensions[0]*dimensions[1]*dimensions[2]/4;
	#pragma omp parallel for collapse(4) schedule(dynamic, size)
	for(x = 0; x < dimensions[0]; x++)
		for(y = 0; y < dimensions[1]; y++)
			for(z = 0; z < dimensions[2]; z++) {
				for (i = 0; i < 3; i++)
	        		vectorField[x][y][z][i] = vectorField[x][y][z][i] + adding.vectorField[x][y][z][i];
			}
}

// // double VectorField::sumOfAbs() {
// // 	double total = 0.0;
// // 	for(int row = 0; row < rows_; row++) {
// // 		for(int col = 0; col < cols_; col++) {
// // 			total += std::abs(vectorCol_.at<double>(row, col)) + std::abs(vectorRow_.at<double>(row, col));
// // 		}
// // 	}
// // 	return total;
// // }

void VectorField::printAround(int x, int y, int z) {
	for (int aroundX = x-1; aroundX <= x+1; aroundX++) {
		std::cout << "Plane x = " << aroundX << "\n";
		for (int aroundY = y-1; aroundY <= y+1; aroundY++) {
			std::cout << "\n";
			for (int aroundZ = z-1; aroundZ <= z+1; aroundZ++) {
				std::vector<double> macaco = getVectorAt(aroundX,aroundY,aroundZ);
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

// // std::vector<double> VectorField::getInfos() {
// // 	std::multiset<double> magnitudes;
// // 	int size = (rows_*cols_);
// // 	double max= 0.0, min = 0.0, mean = 0.0, median = 0.0, deviation = 0.0;
// 	// for(int row = 0; row < rows_; row++) {
// 	//     for(int col = 0; col < cols_; col++) {
// 	//     	std::vector<double> vector = getVectorAt(row, col);
//  //    		double mag = std::sqrt(vector[0]*vector[0] + vector[1]*vector[1]);
//  //    		if (max < mag) max = mag;
//  //    		if (min > mag) min = mag;
//  //    		mean += mag;
// 	// 		magnitudes.insert(mag);
// 	//     }
// 	// }
// // 	mean /= size;
// // 	int count = 1;
// // 	std::multiset<double>::iterator it;
// // 	for (it=magnitudes.begin(); it!=magnitudes.end(); ++it) {
// //     	deviation += std::pow((*it - mean),2);
// //     	if (count == size/2) median = *it;
// //     	count++;
// // 	}
// // 	deviation /= size;
// // 	deviation = std::sqrt(deviation);
// // 	std::vector<double> results;
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
// // 	std::vector<double> results = getInfos();
// // 	myfile << "Min = " << results[0] << " Max = \t" << results[1] << " Median = \t" << results[2] << " Mean = \t" << results[3] << " Standard Deviaon = \t" << results[4] << "\n";
// // 	myfile.close();
// // }

// // void VectorField::printField(std::string filename) {
// // 	std::ofstream myfile;
// // 	myfile.open(filename);
// // 	double minValCol, maxValCol;
// // 	double minValRow, maxValRow;
// // 	minMaxLoc(vectorCol_, &minValCol, &maxValCol);
// // 	minMaxLoc(vectorRow_, &minValRow, &maxValRow);
// // 	for(int row = 0; row < rows_; row++) {
// // 	    for(int col = 0; col < cols_; col++) {
// // 	    	std::vector<double> vector = getVectorAt(row, col);
// //     		double redCol = 255*(vector[1]-minValCol)/(maxValCol-minValCol);
// // 			double blueCol = 255*(maxValCol-vector[1])/(maxValCol-minValCol);
// // 			double redRow = 255*(vector[0]-minValRow)/(maxValRow-minValRow);
// // 			double blueRow = 255*(maxValRow-vector[0])/(maxValRow-minValRow);
// // 			int red = (redCol + redRow)/2;
// // 			int blue = (blueCol + blueRow)/2;
// // 			myfile << col << " " << (vectorCol_.rows - row) << " " << vector[1] << " " << vector[0] << " " <<  red << " " << 0 << " " << blue << "\n";
// // 	    }
// // 	}
// // 	myfile.close();
// // }