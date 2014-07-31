#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <fstream>
#include <cmath>

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
	GaussianBlur(vectorX_, vectorX_, cv::Size(5, 5), 0.03, 0.03);
	GaussianBlur(vectorY_, vectorY_, cv::Size(5, 5), 0.03, 0.03);
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

double VectorField::sumOfAbs() {
	cv::Mat absMat = cv::abs(vectorX_) + cv::abs(vectorY_);
	double total = 0.0;
	for(int row = 0; row < rows_; row++) {
		uchar* absMatRow = absMat.ptr(row);
		for(int col = 0; col < cols_; col++) {
			total += absMatRow[col];
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