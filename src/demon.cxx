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

#include "image/image.hpp"

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
	if (argc < 3) {
		std::cout << "Precisa passar o nome dos arquivos coração! \n";    
		return 0;
	}
    image::basic_image<short,3> image_data,label_data;
    image::io::nifti nifti_parser, out_file;

    if(nifti_parser.load_from_file(argv[1]))
        nifti_parser >> image_data;
    
    out_file << image_data;
    out_file.save_to_file(argv[3]);

    // AsymmetricDemons asymmetricDemons(staticImage, movingImage);
    // asymmetricDemons.run();
    // VectorField displacementField = asymmetricDemons.getDisplField();
    // cv::Mat result = applyVectorField(movingImage, displacementField);
    // fileName = argv[3];
    // fileName += "asymmetric";
    // fileName += extension;
    // imwrite(fileName.c_str(), result, compression_params);

    // SymmetricDemons symmetricDemons(staticImage, movingImage);
    // symmetricDemons.run();
    // VectorField displacementField = symmetricDemons.getDisplField();
    // cv::Mat result = applyVectorField(movingImage, displacementField);
    // fileName = argv[3];
    // fileName += "symmetric";
    // fileName += extension;
    // imwrite(fileName.c_str(), result, compression_params);
	return 0;

}