#ifndef DEMONS_DEMONS_H_
#define DEMONS_DEMONS_H_

#include <vector>
#include <array>
#include <cmath>

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
			}
		void run();
		VectorField getDisplField();
	protected:
		int rows, cols;
		double normalizer;
		cv::Mat staticImage_;
		cv::Mat movingImage_;
		Interpolation movingInterpolator;
		VectorField displField;
		virtual VectorField newDeltaField(VectorField gradients) = 0;
		void updateDisplField(VectorField displField, VectorField deltaField);
		void updateDeformedImage(VectorField displField);
		double getDeformedImageValueAt(int row, int col);
		void debug(int interation, VectorField deltaField, VectorField gradients);
};

#endif