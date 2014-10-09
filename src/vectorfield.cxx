#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <fstream>
#include <cmath>
#include <cstdlib>
#include <set>

#include "vectorfield.h"
#include "imagefunctions.h"

VectorField::VectorField(cv::Mat &vectorRow, cv::Mat &vectorCol) {
	rows_ = vectorCol.rows;
	cols_ = vectorCol.cols;
	vectorRow_ = vectorRow.clone();
  	vectorCol_ = vectorCol.clone();
}

VectorField::VectorField(int rows, int cols) {
	rows_ = rows;
	cols_ = cols;
	vectorRow_ = cv::Mat::zeros(rows, cols, CV_64F);
  	vectorCol_ = cv::Mat::zeros(rows, cols, CV_64F);
}

int VectorField::getRows() {
	return rows_;
}	

int VectorField::getCols() {
	return cols_;
}	

cv::Mat VectorField::getMatRow() {
	return vectorRow_;
}	

cv::Mat VectorField::getMatCol() {
	return vectorCol_;
}	

std::vector<double> VectorField::getVectorAt(int row, int col) {
	std::vector<double> auxVec;
	auxVec.push_back(ImageFunctions::getValue<double>(vectorRow_, row, col));
	auxVec.push_back(ImageFunctions::getValue<double>(vectorCol_, row, col));
	return auxVec;
}

void VectorField::updateVector(int row, int col, double rowValue, double colValue) {
	vectorRow_.at<double>(row, col) = rowValue;
	vectorCol_.at<double>(row, col) = colValue;
}

// void VectorField::applyGaussianFilter() {
// 	GaussianBlur(vectorRow_, vectorRow_, cv::Size(3,3), 1);
// 	GaussianBlur(vectorCol_, vectorCol_, cv::Size(3,3), 1);
// }

double VectorField::vectorNorm(std::vector<double> v) {
	return sqrt(pow(v[0],2)+pow(v[1],2));
}

VectorField VectorField::getNormalized() {
	VectorField normalized(rows_, cols_);
	for(int row = 0; row < rows_; row++) {
	    for(int col = 0; col < cols_; col++) {
	    	std::vector<double> vector = getVectorAt(row, col);
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

void VectorField::applyGaussianFilterCol(cv::Mat image) {
	cv::Mat gaussianKernel = cv::getGaussianKernel(3, 0.7, CV_64F);
	for(int row = 0; row < rows_; row++) {
        for(int col = 0; col < cols_; col++) {
        	double newPixelValue = 0.0;
        	double vectorAt = ImageFunctions::getValue<double>(image, row, col);
        	if (vectorAt != 0.0) {
	        	for (int i = -1; i <= 1; i++) {
	        		double pixelAt = ImageFunctions::getValue<double>(image, row+i, col);
        			double gaussianKernelValue = ImageFunctions::getValue<double>(gaussianKernel, i+1, 0);
        			newPixelValue += gaussianKernelValue*pixelAt;
	        	}
	        	image.at<double>(row, col) = newPixelValue;
        	}
    	}
    }
}

void VectorField::applyGaussianFilterRow(cv::Mat image) {
	cv::Mat gaussianKernel = cv::getGaussianKernel(3, 0.7, CV_64F);
	for(int row = 0; row < rows_; row++) {
        for(int col = 0; col < cols_; col++) {
        	double newPixelValue = 0.0;
        	double vectorAt = ImageFunctions::getValue<double>(image, row, col);
        	if (vectorAt != 0.0) {
	        	for (int i = -1; i <= 1; i++) {
	        		double pixelAt = ImageFunctions::getValue<double>(image, row, col+i);
        			double gaussianKernelValue = ImageFunctions::getValue<double>(gaussianKernel, i+1, 0);
        			newPixelValue += gaussianKernelValue*pixelAt;
	        	}
	        	image.at<double>(row, col) = newPixelValue;
        	}
    	}
    }
}

void VectorField::applyGaussianFilter() {
	applyGaussianFilterRow(vectorRow_);
	applyGaussianFilterCol(vectorRow_);
	applyGaussianFilterRow(vectorCol_);
	applyGaussianFilterCol(vectorCol_);
}

void VectorField::add(VectorField adding) {
	vectorRow_ = vectorRow_ + adding.vectorRow_;
	vectorCol_ = vectorCol_ + adding.vectorCol_;
}

double VectorField::sumOfAbs() {
	double total = 0.0;
	for(int row = 0; row < rows_; row++) {
		for(int col = 0; col < cols_; col++) {
			total += std::abs(vectorCol_.at<double>(row, col)) + std::abs(vectorRow_.at<double>(row, col));
		}
	}
	return total;
}

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

void VectorField::printFieldImage(int iteration, std::vector<int> compression_params) {
	cv::Mat abs_grad_col, abs_grad_row;
	std::string filenamebase("DFI-Iteration"), flCol, flRow;
	std::ostringstream converter;
	converter << iteration;
	filenamebase += converter.str();
	flCol += filenamebase + "x.jpg";
	flRow += filenamebase + "y.jpg";
	convertScaleAbs(vectorRow_, abs_grad_row, 255);
	convertScaleAbs(vectorCol_, abs_grad_col, 255);
	imwrite(flRow.c_str(), abs_grad_row, compression_params);
	imwrite(flCol.c_str(), abs_grad_col, compression_params);
}

std::vector<double> VectorField::getInfos() {
	std::multiset<double> magnitudes;
	int size = (rows_*cols_);
	double max= 0.0, min = 0.0, mean = 0.0, median = 0.0, deviation = 0.0;
	for(int row = 0; row < rows_; row++) {
	    for(int col = 0; col < cols_; col++) {
	    	std::vector<double> vector = getVectorAt(row, col);
    		double mag = std::sqrt(vector[0]*vector[0] + vector[1]*vector[1]);
    		if (max < mag) max = mag;
    		if (min > mag) min = mag;
    		mean += mag;
			magnitudes.insert(mag);
	    }
	}
	mean /= size;
	int count = 1;
	std::multiset<double>::iterator it;
	for (it=magnitudes.begin(); it!=magnitudes.end(); ++it) {
    	deviation += std::pow((*it - mean),2);
    	if (count == size/2) median = *it;
    	count++;
	}
	deviation /= size;
	deviation = std::sqrt(deviation);
	std::vector<double> results;
	results.push_back(min);
	results.push_back(max);
	results.push_back(median);
	results.push_back(mean);
	results.push_back(deviation);
	return results;
}

void VectorField::printFieldInfos(std::string filename, int iteration) {
	std::ofstream myfile;
	if (iteration <= 1) myfile.open(filename);
	else myfile.open(filename, std::ios_base::app);
	myfile << "Iteration " << iteration << "\n";
	std::vector<double> results = getInfos();
	myfile << "Min = " << results[0] << " Max = \t" << results[1] << " Median = \t" << results[2] << " Mean = \t" << results[3] << " Standard Deviaon = \t" << results[4] << "\n";
	myfile.close();
}

void VectorField::printField(std::string filename) {
	std::ofstream myfile;
	myfile.open(filename);
	double minValCol, maxValCol;
	double minValRow, maxValRow;
	minMaxLoc(vectorCol_, &minValCol, &maxValCol);
	minMaxLoc(vectorRow_, &minValRow, &maxValRow);
	for(int row = 0; row < rows_; row++) {
	    for(int col = 0; col < cols_; col++) {
	    	std::vector<double> vector = getVectorAt(row, col);
    		double redCol = 255*(vector[1]-minValCol)/(maxValCol-minValCol);
			double blueCol = 255*(maxValCol-vector[1])/(maxValCol-minValCol);
			double redRow = 255*(vector[0]-minValRow)/(maxValRow-minValRow);
			double blueRow = 255*(maxValRow-vector[0])/(maxValRow-minValRow);
			int red = (redCol + redRow)/2;
			int blue = (blueCol + blueRow)/2;
			myfile << col << " " << (vectorCol_.rows - row) << " " << vector[1] << " " << vector[0] << " " <<  red << " " << 0 << " " << blue << "\n";
	    }
	}
	myfile.close();
}