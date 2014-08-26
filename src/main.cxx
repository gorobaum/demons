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

Mat histogramMatching(Mat staticImage, Mat movingImage) {
    cv::Mat staticImageHist, movingImageHist;
    int channels[] = {0};
    int histSize = 256;
    float range[] = { 0, 255 };
    const float* ranges[] = { range };
    calcHist(&staticImage, 1, channels, cv::Mat(), staticImageHist, 1, &histSize, ranges);
    calcHist(&movingImage, 1, channels, cv::Mat(), movingImageHist, 1, &histSize, ranges);
    Mat freqSIH(1, 256, CV_32F);
    Mat freqMIH(1, 256, CV_32F);
    freqSIH = staticImageHist.clone();
    freqMIH = movingImageHist.clone();
    for(int i = 1; i < 256; i++) {
        freqSIH.at<float>(i) += freqSIH.at<float>(i-1);
        freqMIH.at<float>(i) += freqMIH.at<float>(i-1);
    }
    freqSIH /= freqSIH.total()*256;
    freqMIH /= freqMIH.total()*256;
    cv::Mat LUT = Mat::zeros(1, 256, CV_8U);
    int j = 0, i = 0;
    double mean = (freqSIH.at<float>(j+1) - freqSIH.at<float>(j))/2;
    while(i < 256) {
        while((freqMIH.at<float>(i) - freqSIH.at<float>(j)) <= mean) {
            LUT.at<uchar>(i) = j;
            i++;
            if (i >= 256) break;
        }
        j++;
        if (j == 255) {
            while(i < 256) {
                LUT.at<uchar>(i) = j;  
                i++;
            } 
            break;
        }
        mean = (freqSIH.at<float>(j+1) - freqSIH.at<float>(j))/2;
    }
    return LUT;
}

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
        cv::Scalar inter = Interpolation::bilinearInterpolation(teste, 2.5, 2, false);
        std::cout << inter << "\n";
    } else {
		Mat staticImage = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
		Mat movingImage = imread(argv[2], CV_LOAD_IMAGE_GRAYSCALE);
        Mat lut = histogramMatching(staticImage, movingImage);
        cv::LUT(movingImage, lut, movingImage);
        std::string mn("moving_M.jpg");
        imwrite(mn.c_str(), movingImage, compression_params);
        Demons demons(staticImage, movingImage);
        demons.demons();
        Mat deformed = demons.getRegistration();
        std::string imageName("deformed.jpg");
        imwrite(imageName.c_str(), deformed, compression_params);
	}
	return 0;
}