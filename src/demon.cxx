#include <string>
#include <iostream>
#include <vector>
#include <cstring>

#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "demons.h"
#include "symmetricdemons.h"
#include "asymmetricdemons.h"
#include "vectorfield.h"
#include "interpolation.h"
#include "imagefunctions.h"

using namespace cv;

int main(int argc, char** argv) {
	if (argc < 3) {
		std::cout << "Precisa passar o nome dos arquivos coração! \n";    
		return 0;
	}
	vector<int> compression_params;
    compression_params.push_back(CV_IMWRITE_JPEG_QUALITY);
    compression_params.push_back(95);
	Mat staticImage = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
	Mat movingImage = imread(argv[2], CV_LOAD_IMAGE_GRAYSCALE);
    // GaussianBlur(staticImage, staticImage, cv::Size(3,3), 1.0);
    // GaussianBlur(movingImage, movingImage, cv::Size(3,3), 1.0);
    // Mat lut = ImageFunctions::histogramMatching(staticImage, movingImage);
    // cv::LUT(movingImage, lut, movingImage);
    Mat deformed;
    std::string fileName;
    char* extension = std::strrchr(argv[2], '.');

    SymmetricDemons symmetricDemons(staticImage, movingImage);
    symmetricDemons.demons();
    deformed = symmetricDemons.getRegistration();
    fileName = argv[3];
    fileName += "symmetric";
    fileName += extension;
    imwrite(fileName.c_str(), deformed, compression_params);

    AsymmetricDemons asymmetricDemons(staticImage, movingImage);
    asymmetricDemons.demons();
    deformed = asymmetricDemons.getRegistration();
    fileName = argv[3];
    fileName += "asymmetric";
    fileName += extension;
    imwrite(fileName.c_str(), deformed, compression_params);
	return 0;
}