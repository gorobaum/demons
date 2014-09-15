#include "gradient.h"

#include <opencv2/imgproc/imgproc.hpp>

VectorField Gradient::getSobelGradient() {
	int rows = image_.rows, cols = image_.cols;
	cv::Mat gradRow = cv::Mat::zeros(rows, cols, CV_64F);
	cv::Mat gradCol = cv::Mat::zeros(rows, cols, CV_64F);
	cv::Sobel(image_, gradRow, CV_64F, 0, 1);
	cv::Sobel(image_, gradCol, CV_64F, 1, 0);
	VectorField grad(gradRow, gradCol);
	return grad;
}

VectorField Gradient::getBasicGradient() {
	int rows = image_.rows, cols = image_.cols;
	cv::Mat kernelRow = cv::Mat::zeros(3, 3, CV_64F); 
	cv::Mat kernelCol = cv::Mat::zeros(3, 3, CV_64F);
	cv::Mat gradRow = cv::Mat::zeros(rows, cols, CV_64F);
	cv::Mat gradCol = cv::Mat::zeros(rows, cols, CV_64F);
	kernelRow.at<float>(1,0) = -1;
	kernelRow.at<float>(1,2) = 1;
	kernelCol.at<float>(0,1) = -1;
	kernelCol.at<float>(2,1) = 1;
	filter2D(image_, gradRow, CV_64F , kernelRow);
	filter2D(image_, gradCol, CV_64F , kernelCol);
	VectorField grad(gradRow, gradCol);
	return grad;
}

cv::Mat Gradient::normalizeSobelImage(cv::Mat sobelImage) {
	double minVal, maxVal;
	minMaxLoc(sobelImage, &minVal, &maxVal); //find minimum and maximum intensities
	cv::Mat normalized;
	sobelImage.convertTo(normalized, CV_64F, 255.0/(maxVal - minVal), -minVal * 255.0/(maxVal - minVal));
	return normalized;
}