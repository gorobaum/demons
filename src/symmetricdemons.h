#ifndef DEMONS_SYMMECTRICDEMONS_H_
#define DEMONS_SYMMECTRICDEMONS_H_

#include <vector>
#include <array>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "demons.h"

class SymmetricDemons : public Demons {
	using Demons::Demons;
	private:
		VectorField newDeltaField(VectorField gradients, Gradient deformedImageGradient);
};

#endif