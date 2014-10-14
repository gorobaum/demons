#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <fstream>
#include <cmath>
#include <set>

#include "vectorfield.h"
#include "imagefunctions.h"

VectorField::VectorField(cv::Mat vectorRow, cv::Mat vectorCol) {
	rows_ = vectorCol.rows;
	cols_ = vectorCol.cols;
	Field auxField;
	for(int row = 0; row < rows_; row++) {
		std::vector<std::vector<double>> cols;
        double* vrRow = vectorRow.ptr<double>(row);
        double* vcRow = vectorCol.ptr<double>(row);
        for(int col = 0; col < cols_; col++) {
            std::vector<double> newVector;
            newVector.push_back(vrRow[col]);
            newVector.push_back(vcRow[col]);
            cols.push_back(newVector);
        }
        auxField.push_back(cols);
    }
    vectorField = auxField;
}

VectorField::VectorField(int rows, int cols) {
	rows_ = rows;
	cols_ = cols;
	Field auxField;
	for(int row = 0; row < rows_; row++) {
		std::vector<std::vector<double>> cols;
        for(int col = 0; col < cols_; col++) {
            std::vector<double> newVector(2, 0.0);
            cols.push_back(newVector);
        }
        auxField.push_back(cols);
    }
    vectorField = auxField;
}

int VectorField::getRows() {
	return rows_;
}	

int VectorField::getCols() {
	return cols_;
}		

std::vector<double> VectorField::getVectorAt(int row, int col) {
	std::vector<double> retVector(2, 0.0);
	if (row >= 0 && row <= rows_-1 && col >= 0 && col <= cols_-1) retVector = vectorField[row][col];
	return retVector;
}

void VectorField::updateVector(int row, int col, double rowValue, double colValue) {
	if (row >= 0 && row <= rows_-1 && col >= 0 && col <= cols_-1) {
		vectorField[row][col][0] = rowValue;
		vectorField[row][col][1] = colValue;
	}
}

double VectorField::vectorNorm(std::vector<double> v) {
	return sqrt(pow(v[0],2)+pow(v[1],2));
}

VectorField VectorField::getNormalized() {
	VectorField normalized(rows_, cols_);
	for(int row = 0; row < rows_; row++) {
	    for(int col = 0; col < cols_; col++) {
	    	std::vector<double> vector = vectorField[row][col];
	    	double normalizedRow = 0.0, normalizedCol = 0.0;
	    	if ((vector[0]*vector[1]) != 0.0) {
	    		normalizedRow = vector[0]/vectorNorm(vector);
	    		normalizedCol = vector[1]/vectorNorm(vector);
	    	}
	    	normalized.updateVector(row, col, normalizedRow, normalizedCol);
	    }
	}
	return normalized;
}

void VectorField::applyGaussianFilter(double kernelSize, double deviation) {
	cv::Mat gaussianKernel = cv::getGaussianKernel(kernelSize, deviation, CV_64F);
	for(int row = 0; row < rows_; row++) {
        for(int col = 0; col < cols_; col++) {
        	std::vector<double> vector = vectorField[row][col];
        	double newPixelValueRow = 0.0;
        	double newPixelValueCol = 0.0;
        	if (vector[0] != 0.0 || vector[1] != 0.0) {
	        	for (int i = -1; i <= 1; i++) {
	        		std::vector<double> auxVector = getVectorAt(row+i,col);
        			double pixelAtVectorRow = auxVector[0];
        			double pixelAtVectorCol = auxVector[1];
        			double gaussianKernelValue = ImageFunctions::getValue<double>(gaussianKernel, i+1, 0);
        			newPixelValueRow += gaussianKernelValue*pixelAtVectorRow;
        			newPixelValueCol += gaussianKernelValue*pixelAtVectorCol;
	        	}
	        	updateVector(row, col, newPixelValueRow, newPixelValueCol);
        	}
    	}
    }
    for(int row = 0; row < rows_; row++) {
        for(int col = 0; col < cols_; col++) {
        	std::vector<double> vector = vectorField[row][col];
        	double newPixelValueRow = 0.0;
        	double newPixelValueCol = 0.0;
        	if (vector[0] != 0.0 || vector[1] != 0.0) {
	        	for (int i = -1; i <= 1; i++) {
	        		std::vector<double> auxVector = getVectorAt(row,col+i);
        			double pixelAtVectorRow = auxVector[0];
        			double pixelAtVectorCol = auxVector[1];
        			double gaussianKernelValue = ImageFunctions::getValue<double>(gaussianKernel, i+1, 0);
        			newPixelValueRow += gaussianKernelValue*pixelAtVectorRow;
        			newPixelValueCol += gaussianKernelValue*pixelAtVectorCol;
	        	}
	        	updateVector(row, col, newPixelValueRow, newPixelValueCol);
        	}
    	}
    }
}

void VectorField::add(VectorField adding) {
	for(int row = 0; row < rows_; row++) {
        for(int col = 0; col < cols_; col++) {
        	double vectorRow = vectorField[row][col][0] + adding.vectorField[row][col][0];
        	double vectorCol = vectorField[row][col][1] + adding.vectorField[row][col][1];
        	updateVector(row, col, vectorRow, vectorCol);
        }
    }
}

// double VectorField::sumOfAbs() {
// 	double total = 0.0;
// 	for(int row = 0; row < rows_; row++) {
// 		for(int col = 0; col < cols_; col++) {
// 			total += std::abs(vectorCol_.at<double>(row, col)) + std::abs(vectorRow_.at<double>(row, col));
// 		}
// 	}
// 	return total;
// }

void VectorField::printFieldAround(int row, int col) {
	for (int auxRow = row-1; auxRow <= row+1; auxRow++) {
		for (int auxCol = col-1; auxCol <= col+1; auxCol++) {
			std::vector<double> macaco = getVectorAt(auxRow,auxCol);
			std::cout << "(" << macaco[0] << ")(" << macaco[1] << ") ";
		}
		std::cout << "\n";
	}
	std::cout << "\n";
}

// void VectorField::printFieldImage(int iteration, std::vector<int> compression_params) {
// 	cv::Mat abs_grad_col, abs_grad_row;
// 	std::string filenamebase("DFI-Iteration"), flCol, flRow;
// 	std::ostringstream converter;
// 	converter << iteration;
// 	filenamebase += converter.str();
// 	flCol += filenamebase + "x.jpg";
// 	flRow += filenamebase + "y.jpg";
// 	convertScaleAbs(vectorRow_, abs_grad_row, 255);
// 	convertScaleAbs(vectorCol_, abs_grad_col, 255);
// 	imwrite(flRow.c_str(), abs_grad_row, compression_params);
// 	imwrite(flCol.c_str(), abs_grad_col, compression_params);
// }

// std::vector<double> VectorField::getInfos() {
// 	std::multiset<double> magnitudes;
// 	int size = (rows_*cols_);
// 	double max= 0.0, min = 0.0, mean = 0.0, median = 0.0, deviation = 0.0;
	// for(int row = 0; row < rows_; row++) {
	//     for(int col = 0; col < cols_; col++) {
	//     	std::vector<double> vector = getVectorAt(row, col);
 //    		double mag = std::sqrt(vector[0]*vector[0] + vector[1]*vector[1]);
 //    		if (max < mag) max = mag;
 //    		if (min > mag) min = mag;
 //    		mean += mag;
	// 		magnitudes.insert(mag);
	//     }
	// }
// 	mean /= size;
// 	int count = 1;
// 	std::multiset<double>::iterator it;
// 	for (it=magnitudes.begin(); it!=magnitudes.end(); ++it) {
//     	deviation += std::pow((*it - mean),2);
//     	if (count == size/2) median = *it;
//     	count++;
// 	}
// 	deviation /= size;
// 	deviation = std::sqrt(deviation);
// 	std::vector<double> results;
// 	results.push_back(min);
// 	results.push_back(max);
// 	results.push_back(median);
// 	results.push_back(mean);
// 	results.push_back(deviation);
// 	return results;
// }

// void VectorField::printFieldInfos(std::string filename, int iteration) {
// 	std::ofstream myfile;
// 	if (iteration <= 1) myfile.open(filename);
// 	else myfile.open(filename, std::ios_base::app);
// 	myfile << "Iteration " << iteration << "\n";
// 	std::vector<double> results = getInfos();
// 	myfile << "Min = " << results[0] << " Max = \t" << results[1] << " Median = \t" << results[2] << " Mean = \t" << results[3] << " Standard Deviaon = \t" << results[4] << "\n";
// 	myfile.close();
// }

// void VectorField::printField(std::string filename) {
// 	std::ofstream myfile;
// 	myfile.open(filename);
// 	double minValCol, maxValCol;
// 	double minValRow, maxValRow;
// 	minMaxLoc(vectorCol_, &minValCol, &maxValCol);
// 	minMaxLoc(vectorRow_, &minValRow, &maxValRow);
// 	for(int row = 0; row < rows_; row++) {
// 	    for(int col = 0; col < cols_; col++) {
// 	    	std::vector<double> vector = getVectorAt(row, col);
//     		double redCol = 255*(vector[1]-minValCol)/(maxValCol-minValCol);
// 			double blueCol = 255*(maxValCol-vector[1])/(maxValCol-minValCol);
// 			double redRow = 255*(vector[0]-minValRow)/(maxValRow-minValRow);
// 			double blueRow = 255*(maxValRow-vector[0])/(maxValRow-minValRow);
// 			int red = (redCol + redRow)/2;
// 			int blue = (blueCol + blueRow)/2;
// 			myfile << col << " " << (vectorCol_.rows - row) << " " << vector[1] << " " << vector[0] << " " <<  red << " " << 0 << " " << blue << "\n";
// 	    }
// 	}
// 	myfile.close();
// }