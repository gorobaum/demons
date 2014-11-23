#include "asymmetricdemonsfunction.h"

VectorField AsymmetricDemonsFunction::newDeltaField(VectorField gradients) {
	VectorField deltaField(dimensions, 0.0);
	for(int x = 0; x < dimensions[0]; x++)
		for(int y = 0; y < dimensions[1]; y++)
			for(int z = 0; z < dimensions[2]; z++) {
				std::vector<float> sGradVec = gradients.getVectorAt(x, y, z);
				float deformedValue = getDeformedImageValueAt(x, y, z);
				float diff = deformedValue - staticImage_.getPixelAt(x,y,z);
				float denominator = ((diff*diff)/spacing) + (sGradVec[0])*(sGradVec[0]) + (sGradVec[1])*(sGradVec[1]) + (sGradVec[2])*(sGradVec[2]);
				if (denominator > 0.0) {
					std::vector<float> deltaVector(3, 0.0);
					deltaVector[0] = (sGradVec[0])*diff/denominator;
					deltaVector[1] = (sGradVec[1])*diff/denominator;
					deltaVector[2] = (sGradVec[2])*diff/denominator;
					deltaField.updateVector(x, y, z, deltaVector);
				}
			}
	return deltaField;
}