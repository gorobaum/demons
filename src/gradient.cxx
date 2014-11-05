#include "gradient.h"
#include <iostream>
#include <vector>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>

VectorField Gradient::getBasicGradient() {
	int rows = image_.rows, cols = image_.cols;
	double m[3] = {-0.5, 0, 0.5};
	cv::Mat kernelRow = cv::Mat(3,1, CV_64F, m);
	cv::Mat kernelCol = cv::Mat(1,3, CV_64F, m);
	cv::Mat gradRow = cv::Mat::zeros(rows, cols, CV_64F);
	cv::Mat gradCol = cv::Mat::zeros(rows, cols, CV_64F);
	cv::filter2D(image_, gradRow, CV_64F, kernelRow);
	cv::filter2D(image_, gradCol, CV_64F, kernelCol);
	VectorField grad(gradRow, gradCol);
	return grad;
}