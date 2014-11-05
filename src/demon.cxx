#include <string>
#include <iostream>
#include <vector>
#include <cstring>

#include "vectorfield.h"
#include "image.h"
#include "interpolation.h"

int main(int argc, char** argv) {
    std::vector<int> dimensions;
    dimensions.push_back(3);
    dimensions.push_back(3);
    dimensions.push_back(3);
    VectorField teste(dimensions, 3);
    teste.printAround(1,1,1);
    teste.applyGaussianFilter(3,1);
    teste.printAround(1,1,1);
    VectorField sum1(dimensions, 3.0);
    VectorField sum2(dimensions, 3.0);
    sum1.add(sum2);
    sum1.printAround(1,1,1);

    Image<unsigned char> image(dimensions);
    image.printAround(1,1,1);
    Interpolation interpolator(image);
    image(1,1,1) = 10;
    std::cout << (int)interpolator.NNInterpolation<unsigned char>(1.1, 1.1, 1.1) << "\n";

    VectorField grad = image.getGradient();
    grad.printAround(1,1,1);
	return 0;

}