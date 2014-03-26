
#include <cstdio>
#include <cmath>
#include <iostream>

#include "deform.h"
#include "interpolation.h"

using namespace cv;

Mat Deform::applySinDeformation() {
	Mat deformatedImage(originalImage_);
	for(int row = 0; row < deformatedImage.rows; row++) {
		uchar* di = deformatedImage.ptr(row);
		uchar* oi = originalImage_.ptr(row);
    for(int col = 0; col < deformatedImage.cols; col++) {
    	uchar aux = Interpolation::NNInterpolation(originalImage_, row + 2*sin(col/16), col);
    	di[col] = aux;
    }
  }
	return deformatedImage;
}
