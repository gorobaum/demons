#include <string>
#include <iostream>
#include <vector>
#include <cstring>

#include "vectorfield.h"
#include "image.h"

int main(int argc, char** argv) {
    std::vector<int> dimensions;
    dimensions.push_back(3);
    dimensions.push_back(3);
    dimensions.push_back(3);
    VectorField teste(dimensions, 3.0);
    teste.printFieldAround(1,1,1);
    teste.applyGaussianFilter(3,1);
    teste.printFieldAround(1,1,1);
    VectorField sum1(dimensions, 3.0);
    VectorField sum2(dimensions, 3.0);
    sum1.add(sum2);
    sum1.printFieldAround(1,1,1);

    Image<unsigned char> image(dimensions);
    image.printAround(1,1,1);
	return 0;

}