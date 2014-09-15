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
			displField(staticImage.rows, staticImage.cols) {
				rows = staticImage_.rows;
				cols = staticImage_.cols;
				movingImage_.convertTo(deformedImage_, CV_64F, 1);
			}
		void demons();
		cv::Mat getRegistration();
		VectorField getDisplField();
	protected:
		int rows, cols;
		cv::Mat staticImage_;
		cv::Mat movingImage_;
		cv::Mat deformedImage_;
		Interpolation movingInterpolator;
		VectorField displField;
		virtual VectorField newDeltaField(VectorField gradients, Gradient deformedImageGradient) = 0;
		void updateDisplField(VectorField displField, VectorField deltaField);
		void updateDeformedImage(VectorField displField);
};

#endif