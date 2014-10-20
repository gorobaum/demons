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

#include <nifti1_io.h>

using namespace cv;

cv::Mat applyVectorField(cv::Mat image, VectorField displacementField) {
    int rows = image.rows, cols = image.cols;
    cv::Mat result = cv::Mat::zeros(rows, cols, CV_8U);
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
    nifti_image * nim = nifti_image_read(argv[1], 1);
    std::cout << "Número de dimensões = " << nim->ndim << "\n";
	return 0;

}