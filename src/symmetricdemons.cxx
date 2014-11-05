#include "symmetricdemons.h"

VectorField SymmetricDemons::newDeltaField(VectorField gradients) {
	VectorField deltaField(dimensions, 0.0);
	for(int x = 0; x < dimensions[0]; x++)
		for(int y = 0; y < dimensions[1]; y++)
			for(int z = 0; z < dimensions[2]; z++) {
			std::vector<double> staticGrad = gradients.getVectorAt(x, y, z);
			std::vector<double> deformedGrad = calculateDeformedGradientAt(x, y, z);
			double deformedImageValueAt = getDeformedImageValueAt(x, y, z);
			double diff = deformedImageValueAt - (int)staticImage_.getPixelAt(x,y,z);
			double gradX = staticGrad[0]+deformedGrad[0];
			double gradY = staticGrad[1]+deformedGrad[1];
			double gradZ = staticGrad[2]+deformedGrad[2];
			double denominator = (diff*diff) + (gradX*gradX) + (gradY*gradY) + (gradZ*gradZ);
			if (denominator > 0) {
				std::vector<double> deltaVector(3, 0.0);
				deltaVector[0] = 2*gradX*diff/denominator;
				deltaVector[1] = 2*gradY*diff/denominator;
				deltaVector[2] = 2*gradZ*diff/denominator;
				deltaField.updateVector(x, y, z, deltaVector);
			}
		}
	return deltaField;
}

std::vector<double> SymmetricDemons::calculateDeformedGradientAt(int x, int y, int z) {
	std::vector<double> deformedGrad(3, 0.0);
	deformedGrad[0] += getDeformedImageValueAt(x-1, y, z)*(-0.5);
	deformedGrad[0] += getDeformedImageValueAt(x+1, y, z)*(0.5);
	deformedGrad[1] += getDeformedImageValueAt(x, y-1, z)*(-0.5);
	deformedGrad[1] += getDeformedImageValueAt(x, y+1, z)*(0.5);
	deformedGrad[2] += getDeformedImageValueAt(x, y, z-1)*(-0.5);
	deformedGrad[2] += getDeformedImageValueAt(x, y, z+1)*(0.5);
	return deformedGrad;
}