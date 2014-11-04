#include "imagefunctions.h"

void ImageFunctions::printAround(cv::Mat image, int row, int col) {
	for (int auxRow = row-1; auxRow <= row+1; auxRow++) {
		for (int auxCol = col-1; auxCol <= col+1; auxCol++) {
			uchar pixelValue = getValue<uchar>(image, auxRow, auxCol);
			std::cout << (int)pixelValue << " ";
		}
		std::cout << "\n";
	}
	std::cout << "\n";
}