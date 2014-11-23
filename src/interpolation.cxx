#ifndef DEMONS_INTERPOLATION_CXX_
#define DEMONS_INTERPOLATION_CXX_

#include <cstdio>
#include <cmath>
#include <iostream>

#include "interpolation.h"

int Interpolation::getNearestInteger(float number) {
	if ((number - floor(number)) <= 0.5) return floor(number);
	return floor(number) + 1.0;
}

#endif
