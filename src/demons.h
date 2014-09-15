#ifndef DEMONS_DEMONS_H_
#define DEMONS_DEMONS_H_

#include <vector>
#include <array>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "vectorfield.h"
#include "interpolation.h"
#include "gradient.h"

class Demons {
	public:
		explicit Demons (cv::Mat &staticImage, cv::Mat &movingImage):
			staticImage_(staticImage), 
			movingImage_(movingImage),
			movingInterpolator(movingImage),
			displField(staticImage.rows, staticImage.cols) {}
		void demons();
		cv::Mat getRegistration();
		VectorField getDisplField();
	private:
		cv::Mat staticImage_;
		cv::Mat movingImage_;
		cv::Mat deformedImage_;
		Interpolation movingInterpolator;
		VectorField displField;
		VectorField newDeltaField(VectorField gradients, Gradient deformedImageGradient);
		void updateDisplField(VectorField displField, VectorField deltaField);
		void updateDeformedImage(VectorField displField);
		bool stopCriteria(std::vector<float> &norm, VectorField displField, VectorField deltaField);
		bool correlationCoef();
		bool rootMeanSquareError();
};

#endif