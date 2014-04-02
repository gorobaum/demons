
#include <cstdio>
#include <cstdlib>
#include <iostream>
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
	Field displField = createField();
	int iteration = 1;
	bool stop = false;
	float norm[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
	std::vector<int> compression_params;
	compression_params.push_back(CV_IMWRITE_JPEG_QUALITY);
	compression_params.push_back(95);
	for(int i = 0; i < 100; i++) {
		time(&startTime);
		for(int row = 0; row < rows; row++) {
			uchar* rowDeformed = deformed.ptr(row);
			for(int col = 0; col < cols; col++) {
				int position = row*rows + col;
				float newRow = row - displField[position].x;
				float newCol = col - displField[position].y;
				uchar newValue = Interpolation::bilinearInterpolation(movingImage_, newRow, newCol);
				rowDeformed[col] = newValue;
				updateDisplField(deformed.ptr(row), displField, gradients[position], row, col, position);
			}
		}
		std::cout << "Iteration " << iteration << '\n';
    std::string imageName("Iteration.jpg");
    imwrite(imageName.c_str(), deformed, compression_params);
    iteration++;
	}
	return deformed;
}

void Demons::updateDisplField(uchar* deformedRow, Demons::Field& displacement, Demons::Vector gradient, int row, int col, int position) {
	uchar* staticRow = staticImage_.ptr(row);
	float dif = (deformedRow[col] - staticRow[col]);
	float division = pow(dif,2) + pow(gradient.x,2) + pow(gradient.y,2);
	if (division > 0.0) {
		displacement[position].x = displacement[position].x + dif/division*gradient.x;
		displacement[position].y = displacement[position].y + dif/division*gradient.y;
	}
}

Demons::Field Demons::createField() {
	Field field;
	int rows = staticImage_.rows, cols = staticImage_.cols;
	for(int row = 0; row < rows; row++) 
		for(int col = 0; col < cols; col++) {
			Vector newVec(0,0);
			field.push_back(newVec);
		}
	return field;
}

Demons::Field Demons::findGrad() {
	int rows = staticImage_.rows, cols = staticImage_.cols;
	cv::Mat sobelX;
	cv::Mat sobelY;
	cv::Sobel(staticImage_, sobelX, CV_32F, 1, 0);
	cv::Sobel(staticImage_, sobelY, CV_32F, 0, 1);
	sobelX = normalizeSobelImage(sobelX);
	sobelY = normalizeSobelImage(sobelY);
	Demons::Field grad;
	for(int row = 0; row < rows; row++) {
		uchar* rowSobelX = sobelX.ptr(row);
		uchar* rowSobelY = sobelY.ptr(row);
		for(int col = 0; col < cols; col++) {
			Vector newVec(rowSobelX[col], rowSobelY[col]);
			grad.push_back(newVec);
		}
	}
	return grad;
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