#ifndef DEMONS_DEMONS_H_
#define DEMONS_DEMONS_H_

#include <vector>
#include <array>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "vectorfield.h"

class Demons {
	public:
		explicit Demons (cv::Mat staticImage, cv::Mat movingImage):
			staticImage_(staticImage), 
			movingImage_(movingImage),
			displField(staticImage.rows, staticImage.cols) {}
		void demons();
		cv::Mat getRegistration();
		VectorField getDisplField();
	private:
		cv::Mat staticImage_;
		cv::Mat movingImage_;
		cv::Mat deformedImage_;
		VectorField displField;
		std::vector<int> compression_params;
		time_t startTime;
		double totalTime;
		VectorField findGrad();
		VectorField findGradSobel();
		double getIterationTime(time_t startTime);
		cv::Mat normalizeSobelImage(cv::Mat sobelImage);
		VectorField newDeltaField(VectorField gradients);
		void updateDisplField(VectorField displField, VectorField deltaField);
		void printVFN(VectorField vectorField, VectorField deltaField, int iteration);
		void printVFI(VectorField vectorField, int iteration);
		void printDeformedImage(int iteration);
		void updateDeformedImage(VectorField displField);
		bool stopCriteria(std::vector<float> &norm, VectorField displField, VectorField deltaField);
		bool correlationCoef();
		bool rootMeanSquareError();
};

#endif