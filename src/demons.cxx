
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <opencv2/imgproc/imgproc.hpp>

#include "demons.h"
#include "interpolation.h"

cv::Mat Demons::demons() {
	Field gradients = findGrad();
	int rows = staticImage_.rows, cols = staticImage_.cols;
	// Create the deformed image
	cv::Mat deformed(rows, cols, CV_LOAD_IMAGE_GRAYSCALE);
	deformed = cv::Scalar(0);
	return deformed;
}

void Demons::updateDisplField(cv::Mat deformed, Field displField, Field gradients, int x, int y, int position) {
	
}

Demons::Field Demons::findGrad() {
	int rows = staticImage_.rows, cols = staticImage_.cols;
	cv::Mat sobelX;
	cv::Mat sobelY;
	cv::Sobel(staticImage_, sobelX, CV_32F, 1, 0);
	cv::Sobel(staticImage_, sobelY, CV_32F, 0, 1);
	sobelX = normalizeSobelImage(sobelX);
	sobelY = normalizeSobelImage(sobelY);
	
}

cv::Mat Demons::normalizeSobelImage(cv::Mat sobelImage) {
	double minVal, maxVal;
  minMaxLoc(sobelImage, &minVal, &maxVal); //find minimum and maximum intensities
  cv::Mat normalized;
  sobelImage.convertTo(normalized, CV_8U, 255.0/(maxVal - minVal), -minVal * 255.0/(maxVal - minVal));
  return normalized;
}

double Demons::getIterationTime(time_t startTime) {
	double iterationTime = difftime(time(NULL), startTime);
	totalTime += iterationTime;
	return iterationTime;
}