#ifndef DEMONS_DEMONS_H_
#define DEMONS_DEMONS_H_

#include <vector>
#include <array>
#include <cmath>

#include "vectorfield.h"
#include "interpolation.h"
#include "image.h"

class Demons {
	public:
		explicit Demons (Image<unsigned char> &staticImage, Image<unsigned char> &movingImage, int numOfIterations, double gauKernelSize, double gauDeviation, std::vector<double> spacings):
			staticImage_(staticImage), 
			movingImage_(movingImage),
			dimensions(staticImage.getDimensions()),
			movingInterpolator(movingImage),
			displField(dimensions, 0.0),
			numOfIterations_(numOfIterations),
			gauKernelSize_(gauKernelSize),
			gauDeviation_(gauDeviation) {
				for (int i = 0; i < 3; i++)
					spacing += spacings[i]*spacings[i];
			}
		void run();
		VectorField getDisplField();
	protected:
		Image<unsigned char> staticImage_;
		Image<unsigned char> movingImage_;
		std::vector<int> dimensions;
		Interpolation movingInterpolator;
		VectorField displField;
		int numOfIterations_;
		double gauKernelSize_;
		double gauDeviation_;
		double spacing;
		virtual VectorField newDeltaField(VectorField gradients) = 0;
		void updateDisplField(VectorField &displField, VectorField &deltaField);
		void updateDeformedImage(VectorField displField);
		double getDeformedImageValueAt(int x, int y, int z);
		void debug(int interation, VectorField deltaField, VectorField gradients);
};

#endif