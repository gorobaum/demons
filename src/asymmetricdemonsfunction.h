#ifndef DEMONS_ASYMMECTRICDEMONSFUNCTION_H_
#define DEMONS_ASYMMECTRICDEMONSFUNCTION_H_

#include <vector>
#include <array>

#include "demonsfunction.h"

class AsymmetricDemonsFunction : public DemonsFunction {
	using DemonsFunction::DemonsFunction;
	private:
		VectorField newDeltaField(VectorField gradients);
};

#endif