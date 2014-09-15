#include "symmetricdemons.h"

VectorField SymmetricDemons::newDeltaField(VectorField gradients, Gradient deformedImageGradient) {
	VectorField deltaField(rows, cols);
	VectorField gradientDeformed = deformedImageGradient.getSobelGradient();
	for(int row = 0; row < rows; row++) {
		const double* dRow = deformedImage_.ptr<double>(row);
		uchar* sRow = staticImage_.ptr(row);
		for(int col = 0; col < cols; col++) {
			std::vector<double> sGrad = gradients.getVectorAt(row, col);
			std::vector<double> dGrad = gradientDeformed.getVectorAt(row, col);
			double diff = dRow[col] - sRow[col];
			double denominator = (diff*diff) + (sGrad[0]+dGrad[0])*(sGrad[0]+dGrad[0]) + (sGrad[1]+dGrad[1])*(sGrad[1]+dGrad[1]);
			if (denominator > 0.0) {
				double rowValue = 2*(sGrad[0]+dGrad[0])*diff/denominator;
				double colValue = 2*(sGrad[1]+dGrad[1])*diff/denominator;
				deltaField.updateVector(row, col, rowValue, colValue);
			}
		}
	}
	return deltaField;
}