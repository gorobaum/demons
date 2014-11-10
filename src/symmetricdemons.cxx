#include "symmetricdemons.h"

VectorField SymmetricDemons::newDeltaField(VectorField gradients) {
	VectorField deltaField(dimensions, 0.0);
	int i;
	#pragma omp parallel for
		for(i = 0; i < dimensions[0]; i++)
			for(int y = 0; y < dimensions[1]; y++)
				for(int z = 0; z < dimensions[2]; z++) {
				std::vector<double> staticGrad = gradients.getVectorAt(i, y, z);
				std::vector<double> deformedGrad = calculateDeformedGradientAt(i, y, z);
				double deformedImageValueAt = getDeformedImageValueAt(i, y, z);
				double diff = deformedImageValueAt - staticImage_.getPixelAt(i,y,z);
				double gradX = staticGrad[0]+deformedGrad[0];
				double gradY = staticGrad[1]+deformedGrad[1];
				double gradZ = staticGrad[2]+deformedGrad[2];
				double denominator = ((diff*diff)/spacing) + (gradX*gradX) + (gradY*gradY) + (gradZ*gradZ);
				if (denominator > 0) {
					std::vector<double> deltaVector(3, 0.0);
					deltaVector[0] = 2*gradX*diff/denominator;
					deltaVector[1] = 2*gradY*diff/denominator;
					deltaVector[2] = 2*gradZ*diff/denominator;
					deltaField.updateVector(i, y, z, deltaVector);
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