#ifndef DEMONS_ASYMMECTRICDEMONS_H_
#define DEMONS_ASYMMECTRICDEMONS_H_

#include <vector>
#include <array>

#include "demonsfunction.h"

class AsymmetricDemonsFunction : public DemonsFunction {
	using DemonsFunction::DemonsFunction;
	private:
		VectorField newDeltaField(VectorField gradients);
};

#endif