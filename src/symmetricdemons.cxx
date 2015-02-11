#include <iostream>

#include "symmetricdemons.h"

VectorField SymmetricDemons::newDeltaField(VectorField gradients) {
	VectorField deltaField(dimensions_, 0.0);
	for(int row = 0; row < dimensions_[0]; row++) {
		uchar* sRow = staticImage_.ptr(row);
		for(int col = 0; col < dimensions_[1]; col++) {
			std::vector<double> staticGrad = gradients.getVectorAt(row, col);
			std::vector<double> deformedGrad = calculateDeformedGradientAt(row, col);
			double deformedImageValueAt = getDeformedImageValueAt(row, col);
			double diff = deformedImageValueAt - sRow[col];
			double gradRow = staticGrad[0]+deformedGrad[0];
			double gradCol = staticGrad[1]+deformedGrad[1];
			double denominator = (diff*diff) + (gradRow*gradRow) + (gradCol*gradCol);
			if (denominator > 0) {
				std::vector<double> deltaVector(2, 0.0);
				deltaVector[0] = 2*gradRow*diff/denominator;
				deltaVector[1] = 2*gradCol*diff/denominator;
				deltaField.updateVector(row, col, deltaVector);
			}
		}
	}
	return deltaField;
}

std::vector<double> SymmetricDemons::calculateDeformedGradientAt(int row, int col) {
	std::vector<double> deformedGrad(2, 0.0);
	deformedGrad[0] += getDeformedImageValueAt(row-1, col)*(-0.5);
	deformedGrad[0] += getDeformedImageValueAt(row+1, col)*(0.5);
	deformedGrad[1] += getDeformedImageValueAt(row, col-1)*(-0.5);
	deformedGrad[1] += getDeformedImageValueAt(row, col+1)*(0.5);
	return deformedGrad;
}