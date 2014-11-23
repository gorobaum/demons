#ifndef DEMONS_DEMONSFUNCTION_H_
#define DEMONS_DEMONSFUNCTION_H_

#include <vector>
#include <array>
#include <cmath>

#include "vectorfield.h"
#include "interpolation.h"
#include "image.h"
#include "demon.h"

class DemonsFunction {
	public:
		explicit DemonsFunction (Image<float> &staticImage, Image<float> &movingImage, std::vector<float> spacings):
			staticImage_(staticImage), 
			movingImage_(movingImage),
			dimensions(staticImage.getDimensions()),
			movingInterpolator(movingImage),
			displField(dimensions, 0.0) {
				for (int i = 0; i < 3; i++)
					spacing += spacings[i]*spacings[i];
				spacing /= 3;
			}
		void run();
		void setExecutionParameters(int numOfIterations, int pyramidSize);
		void setGaussianParameters(float gauKernelSize, float gauDeviation);
		VectorField getDisplField();
	protected:
		Image<float> staticImage_;
		Image<float> movingImage_;
		std::vector<int> dimensions;
		std::vector<Demon> demons;
		Interpolation movingInterpolator;
		VectorField displField;
		int numOfIterations_;
		int pyramidSize_;
		float gauKernelSize_;
		float gauDeviation_;
		float spacing;
		virtual VectorField newDeltaField(VectorField gradients) = 0;
		void createDemons(int currentPyramidLevel);
		std::vector<int> calculateVolumeOfInfluence(int x, int y, int z, int xStep, int yStep, int zStep);
		void updateDisplField(VectorField &displField, VectorField &deltaField);
		void updateDeformedImage(VectorField displField);
		float getDeformedImageValueAt(int x, int y, int z);
		void debug(int interation, VectorField deltaField, VectorField gradients);
};

#endif