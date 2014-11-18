#include <string>
#include <iostream>
#include <vector>
#include <cstring>

#include "vectorfield.h"
#include "image.h"
#include "interpolation.h"
#include "demonsfunction.h"
#include "symmetricdemonsfunction.h"
#include "asymmetricdemonsfunction.h"

int main(int argc, char** argv) {
    std::vector<int> dimensions;
    dimensions.push_back(30);
    dimensions.push_back(30);
    dimensions.push_back(30);
    
    Image<unsigned char> staticImage(dimensions);
    Image<unsigned char> movingImage(dimensions);
    for(int x = 0; x < dimensions[0]; x++)
        for(int y = 0; y < dimensions[1]; y++)
            for(int z = 0; z < dimensions[2]; z++){
                staticImage(x,y,z) = 10;
                movingImage(x,y,z) = 30;
            }

    std::vector<double> spacing(3, 1.0);
    SymmetricDemonsFunction sDemons(staticImage, movingImage, spacing);
    sDemons.setExecutionParameters(50, 2);
    sDemons.setGaussianParameters(3, 1);
    sDemons.run();
    VectorField result = sDemons.getDisplField();
    result.printAround(1,1,1);
    
    Interpolation staticInterpolator(staticImage);
    staticImage(2,1,1) = 20;
    staticImage(2,2,1) = 30;
    staticImage(1,2,1) = 40;
    std::cout << "Interpolation = " << staticInterpolator.trilinearInterpolation<double>(1.5,1.5,1) << "\n";
	return 0;

}