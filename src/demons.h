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
		float totalTime;
		struct Vector {
			float x;
			float y;
			Vector(float a=0, float b=0):
					 x(a),
           y(b){}
		};
		typedef std::vector<Vector> Field;
		Field findGrad();
		double getIterationTime(time_t startTime);
		void updateDisplField(CImg<float> deformed, Field displField, Field gradients, int x, int y, int position);
};

#endif