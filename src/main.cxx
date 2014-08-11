#include <string>
#include <iostream>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "deform.h"
#include "demons.h"
#include "vectorfield.h"
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
	if (strcmp(argv[1], "-d") == 0) {
		Mat originalImage;
        originalImage = imread(argv[4], CV_LOAD_IMAGE_GRAYSCALE);

        if(!originalImage.data) {
            std::cout <<  "Could not open or find the image" << std::endl ;
            return -1;
        }
        
        Deform deform(originalImage);
        double amp = atof(argv[2]);
        double freq = atof(argv[3]);
        Mat deformed = deform.applySinDeformation(amp, freq);

        imwrite("moving.jpg", deformed, compression_params);
	} else if (strcmp(argv[1], "-r") == 0) {
        Mat originalImage;
        originalImage = imread(argv[3], CV_LOAD_IMAGE_GRAYSCALE);

        if(!originalImage.data) {
            std::cout <<  "Could not open or find the image" << std::endl ;
            return -1;
        }
        
        Deform deform(originalImage);
        double angle = atof(argv[2]);
        Mat deformed = deform.rotate(angle);

        imwrite("moving.jpg", deformed, compression_params);
    } else if (strcmp(argv[1], "-t") == 0) {
        Mat originalImage;
        originalImage = imread(argv[4], CV_LOAD_IMAGE_GRAYSCALE);

        if(!originalImage.data) {
            std::cout <<  "Could not open or find the image" << std::endl ;
            return -1;
        }
        
        Deform deform(originalImage);
        double width = atof(argv[2]);
        double height = atof(argv[3]);
        Mat deformed = deform.translation(width, height);

        imwrite("moving.jpg", deformed, compression_params);
    } else if (strcmp(argv[1], "-T") == 0) {
        int rows = 3, cols = 4;
        cv::Mat teste = cv::Mat::zeros(rows, cols, CV_32F);
        for(int row = 0; row < rows; row++) {
            for(int col = 0; col < cols; col++) {
                teste.at<float>(row, col) = row*0.1;
            }
        }
        for(int row = 0; row < rows; row++) {
            for(int col = 0; col < cols; col++) {
                std::cout << teste.at<float>(row, col) << " ";
            }
            std::cout << "\n";
        }
        cv::Scalar inter = Interpolation::bilinearInterpolation(teste, 2.5, 2);
        std::cout << inter << "\n";
    } else {
		Mat staticImage;
		Mat movingImage;
        staticImage = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
        movingImage = imread(argv[2], CV_LOAD_IMAGE_GRAYSCALE);
        Demons demons(staticImage, movingImage);
        demons.demons();
        Mat deformed = demons.getRegistration();
        std::string imageName("deformed.jpg");
        imwrite(imageName.c_str(), deformed, compression_params);
	}
	return 0;
}