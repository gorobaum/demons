#include <string>
#include <iostream>
#include <vector>
#include <cstring>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "deformation.h"
#include "interpolation.h"

using namespace cv;

int main(int argc, char** argv) {
	if (argc < 2) {
		std::cout << "Precisa passar o nome dos arquivos coração! \n";    
		return 0;
	}
	vector<int> compression_params;
    compression_params.push_back(CV_IMWRITE_JPEG_QUALITY);
    compression_params.push_back(95);
    char* extension;
	if (strcmp(argv[1], "-d") == 0) {
		Mat originalImage;
        originalImage = imread(argv[4], CV_LOAD_IMAGE_GRAYSCALE);
        extension =  std::strrchr(argv[4], '.');

        if(!originalImage.data) {
            std::cout <<  "Could not open or find the image" << std::endl ;
            return -1;
        }
        
        Deform deform(originalImage);
        double amp = atof(argv[2]);
        double freq = atof(argv[3]);
        Mat deformed = deform.applySinDeformation(amp, freq);
        string movingName = "moving";
        movingName.append(extension);
        imwrite(movingName, deformed, compression_params);
	} else if (strcmp(argv[1], "-r") == 0) {
        Mat originalImage;
        originalImage = imread(argv[3], CV_LOAD_IMAGE_GRAYSCALE);
        extension =  std::strrchr(argv[3], '.');

        if(!originalImage.data) {
            std::cout <<  "Could not open or find the image" << std::endl ;
            return -1;
        }
        
        Deform deform(originalImage);
        double angle = atof(argv[2]);
        Mat deformed = deform.rotate(angle);
        string movingName = "moving";
        movingName.append(extension);
        imwrite(movingName, deformed, compression_params);
    } else if (strcmp(argv[1], "-t") == 0) {
        Mat originalImage;
        originalImage = imread(argv[4], CV_LOAD_IMAGE_GRAYSCALE);
        extension =  std::strrchr(argv[4], '.');

        if(!originalImage.data) {
            std::cout <<  "Could not open or find the image" << std::endl ;
            return -1;
        }
        
        Deform deform(originalImage);
        double width = atof(argv[2]);
        double height = atof(argv[3]);
        Mat deformed = deform.translation(width, height);
        string movingName = "moving";
        movingName.append(extension);
        imwrite(movingName, deformed, compression_params);
    }
	return 0;
}
