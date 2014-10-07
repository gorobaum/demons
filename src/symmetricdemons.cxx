#include "symmetricdemons.h"

VectorField SymmetricDemons::newDeltaField(VectorField gradients) {
	VectorField deltaField(rows, cols);
	for(int row = 0; row < rows; row++) {
		uchar* sRow = staticImage_.ptr(row);
		for(int col = 0; col < cols; col++) {
			std::vector<double> staticGrad = gradients.getVectorAt(row, col);
			std::vector<double> deformedGrad = calculateDeformedGradientAt(row, col);
			double deformedValue = getDeformedValue(row, col);
			double diff = deformedValue - sRow[col];
			double denominator = (diff*diff)/normalizer + ((staticGrad[0]+deformedGrad[0])*(staticGrad[0]+deformedGrad[0]) + (staticGrad[1]+deformedGrad[1])*(staticGrad[1]+deformedGrad[1]));
			if (std::abs(diff) >= 0.001 || denominator >= 1e-9) {
				double rowValue = 2*(staticGrad[0]+deformedGrad[0])*diff/denominator;
				double colValue = 2*(staticGrad[1]+deformedGrad[1])*diff/denominator;
				deltaField.updateVector(row, col, rowValue, colValue);
			}
		}
	}
	return deltaField;
}

std::vector<double> SymmetricDemons::calculateDeformedGradientAt(int row, int col) {
	std::vector<double> deformedGrad;
	double gradRow;
	double rowAbove = getDeformedValue(row-1, col);
	double rowBelow = getDeformedValue(row+1, col);
	gradRow = (rowBelow - rowAbove)/2;
	double gradCol;
	double colLeft = getDeformedValue(row, col-1);
	double colRight = getDeformedValue(row, col+1);
	gradCol	= (colRight - colLeft)/2;
	deformedGrad.push_back(gradRow);
	deformedGrad.push_back(gradCol);
	return deformedGrad;
}