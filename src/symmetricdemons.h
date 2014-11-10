#ifndef DEMONS_SYMMECTRICDEMONS_H_
#define DEMONS_SYMMECTRICDEMONS_H_

#include <vector>
#include <array>
#include <omp.h>

#include "demons.h"

class SymmetricDemons : public Demons {
	using Demons::Demons;
	private:
		VectorField newDeltaField(VectorField gradients);
		std::vector<double> calculateDeformedGradientAt(int x, int y, int z);
};

#endif