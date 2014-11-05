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
		explicit Demons (Image<unsigned char> &staticImage, Image<unsigned char> &movingImage):
			staticImage_(staticImage), 
			movingImage_(movingImage),
			dimensions(staticImage.getDimensions()),
			movingInterpolator(movingImage),
			displField(dimensions, 0.0) {}
		void run();
		VectorField getDisplField();
	protected:
		double normalizer;
		Image<unsigned char> staticImage_;
		Image<unsigned char> movingImage_;
		std::vector<int> dimensions;
		Interpolation movingInterpolator;
		VectorField displField;
		virtual VectorField newDeltaField(VectorField gradients) = 0;
		void updateDisplField(VectorField &displField, VectorField deltaField);
		void updateDeformedImage(VectorField displField);
		double getDeformedImageValueAt(int x, int y, int z);
		void debug(int interation, VectorField deltaField, VectorField gradients);
};

#endif