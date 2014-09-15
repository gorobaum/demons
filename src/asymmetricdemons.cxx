#include "asymmetricdemons.h"

VectorField AsymmetricDemons::newDeltaField(VectorField gradients, Gradient deformedImageGradient) {
	VectorField deltaField(rows, cols);
	for(int row = 0; row < rows; row++) {
		const float* dRow = deformedImage_.ptr<float>(row);
		uchar* sRow = staticImage_.ptr(row);
		for(int col = 0; col < cols; col++) {
			std::vector<float> sGrad = gradients.getVectorAt(row, col);
			float diff = dRow[col] - sRow[col];
			float denominator = (diff*diff) + (sGrad[0])*(sGrad[0]) + (sGrad[1])*(sGrad[1]);
			if (denominator > 0.0) {
				float rowValue = 2*(sGrad[0])*diff/denominator;
				float colValue = 2*(sGrad[1])*diff/denominator;
				deltaField.updateVector(row, col, rowValue, colValue);
			}
		}
	}
	return deltaField;
}