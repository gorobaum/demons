#ifndef DEMONS_DEMONS_H_
#define DEMONS_DEMONS_H_

#include <vector>
#include <array>

#include "CImg.h"

using namespace cimg_library;

class Demons {
	public:
		explicit Demons (CImg<float> staticImage, CImg<float> movingImage):
			staticImage_(staticImage), movingImage_(movingImage) {}
		CImg<float> demons();
	private:
		CImg<float> staticImage_;
		CImg<float> movingImage_;
		struct Vector {
			float x;
			float y;
			Vector(float a=0, float b=0):
					 x(a),
           y(b){}
		};
		Vector* findGrad();
};

#endif