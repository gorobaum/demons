#ifndef DEMONS_ASYMMECTRICDEMONS_H_
#define DEMONS_ASYMMECTRICDEMONS_H_

#include <vector>
#include <array>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "demons.h"

class AsymmetricDemons : public Demons {
	using Demons::Demons;
	private:
		VectorField newDeltaField(VectorField gradients);
};

#endif