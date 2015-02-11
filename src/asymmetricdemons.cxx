#include "asymmetricdemons.h"

VectorField AsymmetricDemons::newDeltaField(VectorField gradients) {
	VectorField deltaField(dimensions_, 0.0);
	for(int row = 0; row < dimensions_[0]; row++) {
		uchar* sRow = staticImage_.ptr(row);
		for(int col = 0; col < dimensions_[1]; col++) {
			std::vector<double> sGrad = gradients.getVectorAt(row, col);
			double deformedValue = getDeformedImageValueAt(row, col);
			double diff = deformedValue - sRow[col];
			double denominator = (diff*diff) + (sGrad[0])*(sGrad[0]) + (sGrad[1])*(sGrad[1]);
			if (denominator > 0.0) {
				std::vector<double> deltaVector(2, 0.0);
				deltaVector[0] = (sGrad[0])*diff/denominator;
				deltaVector[1] = (sGrad[1])*diff/denominator;
				deltaField.updateVector(row, col, deltaVector);
			}
		}
	}
	return deltaField;
}