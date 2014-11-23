#ifndef DEMONS_SYMMECTRICDEMONSFUNCTION_H_
#define DEMONS_SYMMECTRICDEMONSFUNCTION_H_

#include <vector>
#include <array>
#include <omp.h>

#include "demonsfunction.h"

class SymmetricDemonsFunction : public DemonsFunction {
	using DemonsFunction::DemonsFunction;
	private:
		VectorField newDeltaField(VectorField gradients);
		std::vector<float> calculateDeformedGradientAt(int x, int y, int z);
};

#endif