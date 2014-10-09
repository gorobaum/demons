#ifndef DEMONS_IMAGEFUNCTIONS_H_
#define DEMONS_IMAGEFUNCTIONS_H_

#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

class ImageFunctions {
public:
	template<typename T> static T getValue(const cv::Mat &image, int row, int col);
	static cv::Mat histogramMatching(cv::Mat staticImage, cv::Mat movingImage);
	static void printAround(cv::Mat image, int row, int col);
};

template<typename T>
T ImageFunctions::getValue(const cv::Mat &image, int row, int col) {
	if (col > image.cols-1 || col < 0)
		return 0;
	else if (row > image.rows-1 || row < 0)
		return 0;
	else {
		return image.at<T>(row, col);
	}
}

#endif