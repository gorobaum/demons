#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <fstream>
#include <cmath>
#include <set>

#include "vectorfield.h"

VectorField::VectorField (cv::Mat &vectorX, cv::Mat &vectorY) {
	rows_ = vectorX.rows;
	cols_ = vectorX.cols;
  	vectorX_ = vectorX.clone();
	vectorY_ = vectorY.clone();
}

VectorField::VectorField (int rows, int cols) {
	rows_ = rows;
	cols_ = cols;
  	vectorX_ = cv::Mat::zeros(rows, cols, CV_32F);
	vectorY_ = cv::Mat::zeros(rows, cols, CV_32F);
}

int VectorField::getRows() {
	return rows_;
}	

int VectorField::getCols() {
	return cols_;
}	

std::vector<float> VectorField::getVectorAt(int row, int col) {
	std::vector<float> auxVec;
	auxVec.push_back(getValue(vectorX_, row, col));
	auxVec.push_back(getValue(vectorY_, row, col));
	return auxVec;
}

void VectorField::updateVector(int row, int col, float xValue, float yValue) {
	vectorX_.at<float>(row, col) = xValue;
	vectorY_.at<float>(row, col) = yValue;
}

void VectorField::applyGaussianFilter() {
	GaussianBlur(vectorX_, vectorX_, cv::Size(3, 3), 0.1, 0.1);
	GaussianBlur(vectorY_, vectorY_, cv::Size(3, 3), 0.1, 0.1);
}

float VectorField::vectorNorm(std::vector<float> v) {
	return sqrt(pow(v[0],2)+pow(v[1],2));
}

VectorField VectorField::getNormalized() {
	VectorField normalized(rows_, cols_);
	for(int row = 0; row < rows_; row++) {
	    for(int col = 0; col < cols_; col++) {
	    	std::vector<float> vector = getVectorAt(row, col);
	    	float normalizedX = 0.0, normalizedY = 0.0;
	    	if ((vector[0]*vector[1]) != 0.0) {
	    		normalizedX = vector[0]/vectorNorm(vector);
	    		normalizedY = vector[1]/vectorNorm(vector);
	    	}
	    	normalized.updateVector(row, col, normalizedX, normalizedY);
	    }
	}
	return normalized;
}

void VectorField::add(VectorField adding) {
	vectorX_ = vectorX_ + adding.vectorX_;
	vectorY_ = vectorY_ + adding.vectorY_;
}

float VectorField::sumOfAbs() {
	float total = 0.0;
	for(int row = 0; row < rows_; row++) {
		for(int col = 0; col < cols_; col++) {
			total += std::abs(vectorX_.at<float>(row, col)) + std::abs(vectorY_.at<float>(row, col));
		}
	}
	return total;
}

void VectorField::printFieldImage(int iteration, std::vector<int> compression_params) {
	cv::Mat abs_grad_x, abs_grad_y;
	std::string filenamebase("DFI-Iteration"), flx, fly;
	std::ostringstream converter;
	converter << iteration;
	filenamebase += converter.str();
	flx += filenamebase + "x.jpg";
	fly += filenamebase + "y.jpg";
	convertScaleAbs(vectorX_, abs_grad_x, 255);
	convertScaleAbs(vectorY_, abs_grad_y, 255);
	imwrite(flx.c_str(), abs_grad_x, compression_params);
	imwrite(fly.c_str(), abs_grad_y, compression_params);
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
	if (iteration == 1) myfile.open(filename);
	else myfile.open(filename, std::ios_base::app);
	myfile << "Iteration " << iteration << "\n";
	std::vector<double> results = getInfos();
	myfile << "Min = " << results[0] << " Max = \t" << results[1] << " Median = \t" << results[2] << " Mean = \t" << results[3] << " Standard Deviaon = \t" << results[4] << "\n";
	myfile.close();
}

void VectorField::printField(std::string filename) {
	std::ofstream myfile;
	myfile.open(filename);
	double minValx, maxValx;
	double minValy, maxValy;
	minMaxLoc(vectorX_, &minValx, &maxValx);
	minMaxLoc(vectorY_, &minValy, &maxValy);
	for(int row = 0; row < rows_; row++) {
	    for(int col = 0; col < cols_; col++) {
	    	std::vector<float> vector = getVectorAt(row, col);
    		double redX = 255*(vector[0]-minValx)/(maxValx-minValx);
			double blueX = 255*(maxValx-vector[0])/(maxValx-minValx);
			double redY = 255*(vector[1]-minValy)/(maxValy-minValy);
			double blueY = 255*(maxValy-vector[1])/(maxValy-minValy);
			int red = (redX + redY)/2;
			int blue = (blueX + blueY)/2;
			myfile << col << " " << (vectorX_.rows - row) << " " << vector[0] << " " << vector[1] << " " <<  red << " " << 0 << " " << blue << "\n";
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