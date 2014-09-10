#include <string>
#include <iostream>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "demons.h"
#include "vectorfield.h"
#include "interpolation.h"

using namespace cv;

Mat histogramMatching(Mat staticImage, Mat movingImage) {
    Mat staticImageHist, movingImageHist;
    int channels[] = {0};
    int histSize = 256;
    float range[] = { 0, 255 };
    const float* ranges[] = { range };
    cv::calcHist(&staticImage, 1, channels, cv::Mat(), staticImageHist, 1, &histSize, ranges);
    cv::calcHist(&movingImage, 1, channels, cv::Mat(), movingImageHist, 1, &histSize, ranges);
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
	Mat staticImage = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
	Mat movingImage = imread(argv[2], CV_LOAD_IMAGE_GRAYSCALE);
    Mat originalMovingImage = movingImage.clone();
    Mat lut = histogramMatching(staticImage, movingImage);
    cv::LUT(movingImage, lut, movingImage);
    Demons demons(staticImage, movingImage);
    demons.demons();
    Mat deformed = demons.getRegistration();
    // Mat deformed = demons.getRegistration();
    std::string imageName("deformed.jpg");
    imwrite(imageName.c_str(), deformed, compression_params);
	return 0;
}