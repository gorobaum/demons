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
		explicit Demons (cv::Mat &staticImage, cv::Mat &movingImage, std::vector<int> dimensions):
			staticImage_(staticImage), 
			movingImage_(movingImage),
			movingInterpolator(movingImage),
			dimensions_(dimensions),
			displField(dimensions, 0.0) {}
		void run();
		VectorField getDisplField();
	protected:
		double normalizer;
		cv::Mat staticImage_;
		cv::Mat movingImage_;
		Interpolation movingInterpolator;
		std::vector<int> dimensions_;
		VectorField displField;
		virtual VectorField newDeltaField(VectorField gradients) = 0;
		void updateDisplField(VectorField &displField, VectorField deltaField);
		void updateDeformedImage(VectorField displField);
		double getDeformedImageValueAt(int row, int col);
		void debug(int interation, VectorField deltaField, VectorField gradients);
};

#endif