
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <vector>

#include "demons.h"
#include "interpolation.h"

CImg<float> Demons::demons() {
	Field gradients = findGrad();
	int width = staticImage_.width(), height = staticImage_.height();
	// Create the deformed image
	CImg<float> deformed(staticImage_);
	deformed.fill(0.0);
	Field displField(width*height, Vector(0.0, 0.0));
	time_t startTime;
	int iteration = 1;
	bool stop = false;
	std::vector<float> norm(10, 0);
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
				updateDisplField(deformed, displField, gradients,x, y, position);
			}
		}
		deformed.display("display");
		printf("Iteration %d took %f seconds.\n", iteration, getIterationTime(startTime));
		iteration++;
	}
	return deformed;
}

void Demons::updateDisplField(CImg<float> deformed, Field displField, Field gradients, int x, int y, int position) {
	float dif = (deformed.atXY(x, y, 0, 0, 0.0) - staticImage_.atXY(x, y, 0, 0, 0.0));
	float division = pow(gradients[position].x, 2) + pow(gradients[position].y, 2) + pow(dif, 2);
	if (division != 0) {
		displField[position].x += dif*gradients[position].x/division;
		displField[position].y += dif*gradients[position].y/division;
	}
}

Demons::Field Demons::findGrad() {
	const CImg<float> maskX(3,3,1,1, 1,0,-1, 2,0,-2,1,0,-1), maskY = maskX.get_transpose();
  const CImg<float> imgX = staticImage_.get_convolve(maskX), imgY = staticImage_.get_convolve(maskY);
  int width = staticImage_.width(), height = staticImage_.height();
  /*(imgX,imgY).display("ImgX, imgY");*/
	Field grad(width*height, Vector(0.0,0.0));
	for (int y = 0; y < staticImage_.height(); y++) {
		for (int x = 0; x < staticImage_.width(); x++) {
			Vector newVector(imgX.atXY(x, y, 0, 0, 0.0), imgY.atXY(x, y, 0, 0, 0.0));
			grad[y*height+x] = newVector;
		}
	}
	return grad;
}

double Demons::getIterationTime(time_t startTime) {
	double iterationTime = difftime(time(NULL), startTime);
	totalTime += iterationTime;
	return iterationTime;
}