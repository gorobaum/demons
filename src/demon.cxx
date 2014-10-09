#include <string>
#include <iostream>
#include <vector>
#include <cstring>

#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "symmetricdemons.h"
#include "asymmetricdemons.h"
#include "vectorfield.h"
#include "interpolation.h"
#include "imagefunctions.h"

using namespace cv;

cv::Mat applyVectorField(cv::Mat image, VectorField displacementField) {
    int rows = image.rows, cols = image.cols;
    cv::Mat result = cv::Mat::zeros(rows, cols, image.depth());
    Interpolation imageInterpolator(image);
    for(int row = 0; row < rows; row++) {
        uchar* resultRow = result.ptr<uchar>(row);
        for(int col = 0; col < cols; col++) {
            std::vector<double> displVector = displacementField.getVectorAt(row, col);
            double newRow = row - displVector[0];
            double newCol = col - displVector[1];
            resultRow[col] = imageInterpolator.bilinearInterpolation<uchar>(newRow, newCol);
        }
    }
    return result;
}

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
    std::string fileName;
    char* extension = std::strrchr(argv[2], '.');

    // AsymmetricDemons asymmetricDemons(staticImage, movingImage);
    // asymmetricDemons.run();
    // VectorField displacementField = asymmetricDemons.getDisplField();
    // cv::Mat result = applyVectorField(movingImage, displacementField);
    // fileName = argv[3];
    // fileName += "asymmetric";
    // fileName += extension;
    // imwrite(fileName.c_str(), result, compression_params);

    SymmetricDemons symmetricDemons(staticImage, movingImage);
    symmetricDemons.run();
    VectorField displacementField = symmetricDemons.getDisplField();
    cv::Mat result = applyVectorField(movingImage, displacementField);
    fileName = argv[3];
    fileName += "symmetric";
    fileName += extension;
    imwrite(fileName.c_str(), result, compression_params);
	return 0;

}