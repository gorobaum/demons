#ifndef DEMONS_GRADIENT_H_
#define DEMONS_GRADIENT_H_

#include <vector>
#include <array>

#include "vectorfield.h"
#include "image.h"

class Gradient {
public:
	Gradient(Image<unsigned char> &image) : image_(image) {};
	VectorField getBasicGradient();
private:
	Image<unsigned char> &image_;
};

#endif