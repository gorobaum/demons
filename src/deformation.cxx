
#include <cstdio>
#include <cmath>
#include <iostream>
#include <vector>

#include "deformation.h"
#include "interpolation.h"

cv::Mat Deform::applySinDeformation(double amp, double freq) {
    cv::Mat deformatedImage(originalImage_.rows, originalImage_.cols, CV_LOAD_IMAGE_GRAYSCALE);
    Interpolation imageInterpolator(originalImage_);
    for(int row = 0; row < deformatedImage.rows; row++) {
        uchar* di = deformatedImage.ptr(row);
        for(int col = 0; col < deformatedImage.cols; col++) {
            float newRow = row + amp*sin(col/freq);
            float newCol = col ;
            di[col] = imageInterpolator.bilinearInterpolation<uchar>(newRow, newCol, false);
        }
    }
    return deformatedImage;
}

cv::Mat Deform::rotate(double angle) {
    cv::Mat deformatedImage(originalImage_.rows, originalImage_.cols, CV_LOAD_IMAGE_GRAYSCALE);
    double col = originalImage_.cols/2., row = originalImage_.rows/2.;
    cv::Point2f pt(col, row);
    cv::Mat r = cv::getRotationMatrix2D(pt, angle, 1.0);
    cv::warpAffine(originalImage_, deformatedImage, r, originalImage_.size());

    return deformatedImage;
}

cv::Mat Deform::translation(int width, int height) {
    cv::Mat deformatedImage(originalImage_.rows, originalImage_.cols, CV_LOAD_IMAGE_GRAYSCALE);
    cv::Mat translationMatrix = (cv::Mat_<double>(2,3) << 1, 0, width, 0, 1, height);
    warpAffine(originalImage_, deformatedImage, translationMatrix, originalImage_.size());
    return deformatedImage;
}