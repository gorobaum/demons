#include <string>
#include <iostream>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "demons.h"
#include "vectorfield.h"
#include "interpolation.h"
#include "imagefunctions.h"

using namespace cv;

int main(int argc, char** argv) {
	if (argc < 2) {
		std::cout << "Precisa passar o nome dos arquivos coração! \n";    
		return 0;
	}
	vector<int> compression_params;
    compression_params.push_back(CV_IMWRITE_JPEG_QUALITY);
    compression_params.push_back(95);
	Mat staticImage = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
	Mat movingImage = imread(argv[2], CV_LOAD_IMAGE_GRAYSCALE);
    Mat originalMovingImage = movingImage.clone();
    Mat lut = ImageFunctions::histogramMatching(staticImage, movingImage);
    cv::LUT(movingImage, lut, movingImage);
    Demons demons(staticImage, movingImage);
    demons.demons();
    Mat deformed = demons.getRegistration();
    // Mat deformed = demons.getRegistration();
    std::string imageName("deformed.jpg");
    imwrite(imageName.c_str(), deformed, compression_params);
	return 0;
}