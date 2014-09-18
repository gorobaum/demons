#include "symmetricdemons.h"

VectorField SymmetricDemons::newDeltaField(VectorField gradients, Gradient deformedImageGradient) {
	VectorField deltaField(rows, cols);
	VectorField gradientDeformed = deformedImageGradient.getBasicGradient();
	for(int row = 0; row < rows; row++) {
		const double* dRow = deformedImage_.ptr<double>(row);
		uchar* sRow = staticImage_.ptr(row);
		for(int col = 0; col < cols; col++) {
			std::vector<double> staticGrad = gradients.getVectorAt(row, col);
			std::vector<double> deformedGrad = gradientDeformed.getVectorAt(row, col);
			double diff = dRow[col] - sRow[col];
			double denominator = (diff*diff) + ((staticGrad[0]+deformedGrad[0])*(staticGrad[0]+deformedGrad[0]) + (staticGrad[1]+deformedGrad[1])*(staticGrad[1]+deformedGrad[1]));
			if (denominator > 0.0) {
				double rowValue = 2*(staticGrad[0]+deformedGrad[0])*diff/denominator;
				double colValue = 2*(staticGrad[1]+deformedGrad[1])*diff/denominator;
				deltaField.updateVector(row, col, rowValue, colValue);
			}
		}
	}
	return deltaField;
}