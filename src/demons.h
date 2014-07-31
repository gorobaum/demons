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
			staticImage_(staticImage), movingImage_(movingImage) {}
		void demons();
		cv::Mat getRegistration();
	private:
		cv::Mat staticImage_;
		cv::Mat movingImage_;
		cv::Mat deformedImage_;
		std::vector<int> compression_params;	
		time_t startTime;
		double totalTime;
		VectorField findGrad();
		double getIterationTime(time_t startTime);
		cv::Mat normalizeSobelImage(cv::Mat sobelImage);
		VectorField newDeltaField(VectorField gradients);
		void updateDisplField(VectorField displField, VectorField deltaField);
		void printVFN(VectorField vectorField, int iteration);
		void printVFI(VectorField vectorField, int iteration);
		void printDeformedImage(int iteration);
		void updateDeformedImage(VectorField displField);
		bool stopCriteria(std::vector<double> &norm, VectorField displField, VectorField deltaField);
};

#endif