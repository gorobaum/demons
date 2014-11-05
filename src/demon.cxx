#include <string>
#include <iostream>
#include <vector>
#include <cstring>

#include "vectorfield.h"
#include "image.h"
#include "interpolation.h"
#include "demons.h"
#include "asymmetricdemons.h"
#include "symmetricdemons.h"

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

    Image<unsigned char> staticImage(dimensions);
    Image<unsigned char> movingImage(dimensions);
    for(int x = 0; x < dimensions[0]; x++)
        for(int y = 0; y < dimensions[1]; y++)
            for(int z = 0; z < dimensions[2]; z++){
                staticImage(x,y,z) = 10;
                movingImage(x,y,z) = 20;
            }

    SymmetricDemons sDemons(staticImage, movingImage);
    sDemons.run();
    VectorField result = sDemons.getDisplField();
    result.printAround(1,1,1);
    
    Interpolation staticInterpolator(staticImage);
    staticImage(2,1,1) = 20;
    std::cout << "Interpolation = " << staticInterpolator.trilinearInterpolation<double>(2.9,1,1) << "\n";
	return 0;

}