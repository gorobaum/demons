
#include <cstdio>
#include <cstdlib>

#include "demons.h"

CImg<float> Demons::demons() {
	Vector* gradients = findGrad();
	printf("%f\n", gradients[0].x);
	printf("%f\n", gradients[0].y);
	return staticImage_;
}

Demons::Vector* Demons::findGrad() {
	const CImg<float> maskX(3,3,1,1, 1,0,-1, 2,0,-2,1,0,-1), maskY = maskX.get_transpose();
  const CImg<float> imgX = staticImage_.get_convolve(maskX), imgY = staticImage_.get_convolve(maskY);
  int width = staticImage_.width(), height = staticImage_.height();
  /*(imgX,imgY).display("ImgX, imgY");*/
	Vector* grad = (Vector*)malloc(width*height*sizeof(Vector));
	for (int y = 0; y < staticImage_.height(); y++) {
		for (int x = 0; x < staticImage_.width(); x++) {
			Vector newVector(imgX.atXY(x, y, 0, 0, 0.0), imgY.atXY(x, y, 0, 0, 0.0));
			grad[y*height+x] = newVector;
		}
	}
	return grad;
}