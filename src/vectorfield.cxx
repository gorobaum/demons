#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <fstream>
#include <cmath>
#include <set>

#include "vectorfield.h"

VectorField::VectorField(cv::Mat &vectorRow, cv::Mat &vectorCol) {
	rows_ = vectorCol.rows;
	cols_ = vectorCol.cols;
	vectorRow_ = vectorRow.clone();
  	vectorCol_ = vectorCol.clone();
}

VectorField::VectorField(int rows, int cols) {
	rows_ = rows;
	cols_ = cols;
	vectorRow_ = cv::Mat::zeros(rows, cols, CV_32F);
  	vectorCol_ = cv::Mat::zeros(rows, cols, CV_32F);
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

std::vector<float> VectorField::getVectorAt(int row, int col) {
	std::vector<float> auxVec;
	auxVec.push_back(getValue(vectorRow_, row, col));
	auxVec.push_back(getValue(vectorCol_, row, col));
	return auxVec;
}

void VectorField::updateVector(int row, int col, float rowValue, float colValue) {
	vectorRow_.at<float>(row, col) = rowValue;
	vectorCol_.at<float>(row, col) = colValue;
}

void VectorField::applyGaussianFilter() {
	GaussianBlur(vectorRow_, vectorRow_, cv::Size(5, 5), 1, 1);
	GaussianBlur(vectorCol_, vectorCol_, cv::Size(5, 5), 1, 1);
}

float VectorField::vectorNorm(std::vector<float> v) {
	return sqrt(pow(v[0],2)+pow(v[1],2));
}

VectorField VectorField::getNormalized() {
	VectorField normalized(rows_, cols_);
	for(int row = 0; row < rows_; row++) {
	    for(int col = 0; col < cols_; col++) {
	    	std::vector<float> vector = getVectorAt(row, col);
	    	float normalizedRow = 0.0, normalizedCol = 0.0;
	    	if ((vector[0]*vector[1]) != 0.0) {
	    		normalizedRow = vector[0]/vectorNorm(vector);
	    		normalizedCol = vector[1]/vectorNorm(vector);
	    	}
	    	normalized.updateVector(row, col, normalizedRow, normalizedCol);
	    }
	}
	return normalized;
}

void VectorField::add(VectorField adding) {
	vectorRow_ = vectorRow_ + adding.vectorRow_;
	vectorCol_ = vectorCol_ + adding.vectorCol_;
}

float VectorField::sumOfAbs() {
	float total = 0.0;
	for(int row = 0; row < rows_; row++) {
		for(int col = 0; col < cols_; col++) {
			total += std::abs(vectorCol_.at<float>(row, col)) + std::abs(vectorRow_.at<float>(row, col));
		}
	}
	return total;
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
	    	std::vector<float> vector = getVectorAt(row, col);
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
	    	std::vector<float> vector = getVectorAt(row, col);
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

float VectorField::getValue(cv::Mat image, int row, int col) {
	if (col > image.cols-1 || col < 0)
		return 0;
	else if (row > image.rows-1 || row < 0)
		return 0;
	else {
		return image.at<float>(row, col);
	}
}