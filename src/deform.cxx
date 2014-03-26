
#include <cstdio>
#include <cmath>
#include <iostream>

#include "deform.h"
#include "interpolation.h"

using namespace cv;

Mat Deform::applySinDeformation() {
	Mat deformatedImage(originalImage_.rows, originalImage_.cols, CV_LOAD_IMAGE_GRAYSCALE);
	for(int row = 0; row < deformatedImage.rows; row++) {
		uchar* di = deformatedImage.ptr(row);
    for(int col = 0; col < deformatedImage.cols; col++) {
    	float newRow = row + 2*sin(col/16);
    	float newCol = col;
    	di[col] = Interpolation::NNInterpolation(originalImage_, newRow, newCol);
    }
  }
	return deformatedImage;
}
