#include "symmetricdemonsfunction.h"
#include "profiler.h"

#define EPSILON 1e-5

VectorField SymmetricDemonsFunction::newDeltaField(VectorField gradients) {
	Profiler profiler("New Delta Field");
	VectorField deltaField(dimensions, 0.0);
	for (std::vector<Demon>::iterator it = demons.begin(); it != demons.end(); ++it) {
		std::vector<int> pos = it->getPosition();
		int x = pos[0], y = pos[1], z = pos[2];
		std::vector<float> staticGrad = gradients.getVectorAt(x, y, z);
		std::vector<float> deformedGrad = calculateDeformedGradientAt(x, y, z);
		float deformedImageValueAt = getDeformedImageValueAt(x, y, z);
		float diff = deformedImageValueAt - staticImage_.getPixelAt(x,y,z);
		float gradX = staticGrad[0]+deformedGrad[0];
		float gradY = staticGrad[1]+deformedGrad[1];
		float gradZ = staticGrad[2]+deformedGrad[2];
		float denominator = ((diff*diff)/spacing) + (gradX*gradX) + (gradY*gradY) + (gradZ*gradZ);
		if (denominator != 0) {
			std::vector<float> deltaVector(3, 0.0);
			deltaVector[0] = 2*gradX*diff/denominator;
			deltaVector[1] = 2*gradY*diff/denominator;
			deltaVector[2] = 2*gradZ*diff/denominator;
			deltaField.updateVector(*it, deltaVector);
		}
	}
	return deltaField;
}

std::vector<float> SymmetricDemonsFunction::calculateDeformedGradientAt(int x, int y, int z) {
	std::vector<float> deformedGrad(3, 0.0);
	deformedGrad[0] += getDeformedImageValueAt(x-1, y, z)*(-0.5);
	deformedGrad[0] += getDeformedImageValueAt(x+1, y, z)*(0.5);
	deformedGrad[1] += getDeformedImageValueAt(x, y-1, z)*(-0.5);
	deformedGrad[1] += getDeformedImageValueAt(x, y+1, z)*(0.5);
	deformedGrad[2] += getDeformedImageValueAt(x, y, z-1)*(-0.5);
	deformedGrad[2] += getDeformedImageValueAt(x, y, z+1)*(0.5);
	return deformedGrad;
}
