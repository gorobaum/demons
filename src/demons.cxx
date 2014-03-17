
#include <cstdio>
#include <cstdlib>
#include <ctime>

#include "demons.h"
#include "interpolation.h"

CImg<float> Demons::demons() {
	Vector* gradients = findGrad();
	int width = staticImage_.width(), height = staticImage_.height();
	// Create the deformed image
	CImg<float> deformed(staticImage_);
	deformed.fill(0.0);
	Vector* displField = (Vector*)malloc(width*height*sizeof(Vector));
	Vector* CurrDisplField = (Vector*)malloc(width*height*sizeof(Vector));
	int iteration = 1;
	bool stop = false;
	float norm[10] = {0, 0, 0, 0 ,0 ,0 ,0 ,0 ,0 ,0};
	while (!stop) {
		time(&startTime);
		printf("Iteration number %d\n", iteration);
		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				int position = y*height+x;
				float mix = x - displField[position].x;
				float miy = y - displField[position].y;
				float newValue = Interpolation::bilinearInterpolation(movingImage_, mix, miy);
				deformed.set_linear_atXY(newValue, x, y);
			}
		}
	}
	return deformed;
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

double Demons::getIterationTime() {
	return difftime(time(NULL), startTime);
}