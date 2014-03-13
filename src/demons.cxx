
#include <cstdio>

#include "demons.h"

CImg<float> Demons::demons() {
	printf("Dae, as ibagens são:\n");
	printf("%d\n", staticImage_.width());
	printf("%d\n", movingImage_.width());
	findGrad();
	return staticImage_;
}

Demons::Vector* Demons::findGrad() {
	const CImg<float> maskX(3,3,1,1, 1,0,-1, 2,0,-2,1,0,-1), maskY = maskX.get_transpose();
  	const CImg<float> imgX = staticImage_.get_convolve(maskX), imgY = staticImage_.get_convolve(maskY);
  	(imgX,imgY).display("ImgX, imgY");
	Demons::Vector* macaco;
	return macaco;
}