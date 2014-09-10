#include "imagefunctions.h"

cv::Mat ImageFunctions::histogramMatching(cv::Mat staticImage, cv::Mat movingImage) {
	cv::Mat staticImageHist, movingImageHist;
	int channels[] = {0};
	int histSize = 256;
	float range[] = { 0, 255 };
	const float* ranges[] = { range };
	cv::calcHist(&staticImage, 1, channels, cv::Mat(), staticImageHist, 1, &histSize, ranges);
	cv::calcHist(&movingImage, 1, channels, cv::Mat(), movingImageHist, 1, &histSize, ranges);
	cv::Mat freqSIH(1, 256, CV_32F);
	cv::Mat freqMIH(1, 256, CV_32F);
	freqSIH = staticImageHist.clone();
	freqMIH = movingImageHist.clone();
	for(int i = 1; i < 256; i++) {
	    freqSIH.at<float>(i) += freqSIH.at<float>(i-1);
	    freqMIH.at<float>(i) += freqMIH.at<float>(i-1);
	}
	freqSIH /= freqSIH.total()*256;
	freqMIH /= freqMIH.total()*256;
	cv::Mat LUT = cv::Mat::zeros(1, 256, CV_8U);
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
