
#include <cstdio>
#include <cmath>
#include <iostream>

#include "deform.h"
#include "interpolation.h"

using namespace cv;

Mat Deform::applySinDeformation() {
	Mat deformatedImage(originalImage_);
	for(int y = 0; y < deformatedImage.rows; y++) {
    const uchar* Mi = deformatedImage.ptr<uchar>(y);
    for(int x = 0; x < deformatedImage.cols; x++) {
			std::cout << x << ' ' << y << "\n";
			deformatedImage.at<uchar>(x,y) = Interpolation::NNInterpolation(originalImage_, x, y + 2*sin(x/16));
			std::cout << "oi\n";
		}
	}
	return deformatedImage;
}
