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
		time_t startTime;
		double totalTime;
		VectorField findGrad();
		double getIterationTime(time_t startTime);
		cv::Mat normalizeSobelImage(cv::Mat sobelImage);
		void updateDisplField(VectorField displacement, std::vector<float> gradient, int row, int col);
		void printDisplField(VectorField vectorField, int iteration);
};

#endif