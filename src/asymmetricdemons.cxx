#include "asymmetricdemons.h"

VectorField AsymmetricDemons::newDeltaField(VectorField gradients) {
	VectorField deltaField(rows, cols);
	for(int row = 0; row < rows; row++) {
		uchar* sRow = staticImage_.ptr(row);
		for(int col = 0; col < cols; col++) {
			std::vector<double> sGrad = gradients.getVectorAt(row, col);
			double deformedValue = getDeformedValue(row, col);
			double diff = deformedValue - sRow[col];
			double denominator = (diff*diff) + (sGrad[0])*(sGrad[0]) + (sGrad[1])*(sGrad[1]);
			if (denominator > 0.0) {
				double rowValue = (sGrad[0])*diff/denominator;
				double colValue = (sGrad[1])*diff/denominator;
				deltaField.updateVector(row, col, rowValue, colValue);
			}
		}
	}
	return deltaField;
}