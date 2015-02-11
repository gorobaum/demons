#define _USE_MATH_DEFINES
#include <exception>
#include <iostream>
#include <fstream>
#include <cmath>
#include <set>

#include "vectorfield.h"

#define EPSILON 1e-5

VectorField::VectorField(VectorField2D vectorField) {
	std::vector<int> dimensions;
	dimensions.push_back(vectorField.size());
	dimensions.push_back(vectorField[0].size());
    this->vectorField = vectorField;
}

VectorField::VectorField(std::vector<int> dimensions, double defaultValue) {
	this->dimensions = dimensions;
	vectorField = createVectorField2D(dimensions, defaultValue);
}

VectorField::VectorField2D VectorField::createVectorField2D(std::vector<int> dimensions, double defaultValue) {
	using std::vector;
	return VectorField2D(dimensions[0], vector<vector<double> >(dimensions[1], vector<double>(2, defaultValue)));
}

std::vector<double> VectorField::getVectorAt(int x, int y) {
	if ((x < 0 || x > dimensions[0]-1) || (y < 0 || y > dimensions[1]-1)) 
		return std::vector<double>(2, 0.0);
	return vectorField[x][y];
}

void VectorField::updateVector(int x, int y, std::vector<double> newValues) {
	vectorField[x][y] = newValues;
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
	VectorField2D auxVectorField = vectorField;
	for (int x = 0; x < dimensions[0]; x++)
		for (int y = 0; y < dimensions[1]; y++) {
			std::vector<double> vector = vectorField[x][y];
			if (isZero(vector)) continue;
			std::vector<double> newValues(2, 0.0);
			for (int i = -1; i <= 1; i++) {
				std::vector<double> auxVector = getVectorAt(x+i,y);
				newValues[0] += auxVector[0]*gaussianKernel[i+1];
				newValues[1] += auxVector[1]*gaussianKernel[i+1];
			}
			auxVectorField[x][y] = newValues;
		}
	vectorField = auxVectorField;
	for (int x = 0; x < dimensions[0]; x++)
		for (int y = 0; y < dimensions[1]; y++) {
			std::vector<double> vector = vectorField[x][y];
			if (isZero(vector)) continue;
			std::vector<double> newValues(2, 0.0);
			for (int i = -1; i <= 1; i++) {
				std::vector<double> auxVector = getVectorAt(x,y+i);
				newValues[0] += auxVector[0]*gaussianKernel[i+1];
				newValues[1] += auxVector[1]*gaussianKernel[i+1];
			}
			auxVectorField[x][y] = newValues;
		}
	vectorField = auxVectorField;
}

void VectorField::add(VectorField adding) {
	for (int x = 0; x < dimensions[0]; x++)
		for (int y = 0; y < dimensions[1]; y++)
			for (int i = 0; i < 2; i++)
        		vectorField[x][y][i] = vectorField[x][y][i] + adding.vectorField[x][y][i];
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

void VectorField::printAround(int row, int col) {
	for (int auxRow = row-1; auxRow <= row+1; auxRow++) {
		for (int auxCol = col-1; auxCol <= col+1; auxCol++) {
			std::vector<double> macaco = getVectorAt(auxRow,auxCol);
			std::cout << "(" << macaco[0] << ")(" << macaco[1] << ") ";
		}
		std::cout << "\n";
	}
	std::cout << "\n";
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