
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <ctime>
#include <vector>
#include <opencv2/imgproc/imgproc.hpp>
#include <sstream>

#include "demons.h"
#include "interpolation.h"

#define RMSEcriteria 10
#define CORRCOEFcriteria 0.95
#define STOPcriteria 0.0001
#define SPACING 0.16
#define POSR 130
#define POSC 131

void Demons::demons() {
	int rows = staticImage_.rows, cols = staticImage_.cols;
	// Create the deformed image
	movingImage_.convertTo(deformedImage_, CV_32F, 1);
	std::vector<float> norm(10,0.0);
	VectorField gradients = findGradSobel(staticImage_);
	gradients.printField("Gradients.dat");
	std::string gfName("GradientInformation.info");
	gradients.printFieldInfos(gfName, 1);
	// for(int i = POSR - 1; i < POSR + 2; i ++) {
	// 	for(int j = POSC - 1; j < POSC + 2; j++) {
	// 		std::cout << (int)staticImage_.at<uchar>(i,j) << "\t";
	// 	}
	// 	std::cout << "\n";
	// }
	// std::cout << "\n";
	// for(int i = POSR - 1; i < POSR + 2; i ++) {
	// 	for(int j = POSC - 1; j < POSC + 2; j++) {
	// 		std::cout << (int)movingImage_.at<uchar>(i,j) << "\t";
	// 	}
	// 	std::cout << "\n";
	// }
	// std::cout << "Gradient[" << POSR << "][" << POSC << "] = [" << gradients.getVectorAt(POSR,POSC)[0] << "][" << gradients.getVectorAt(POSR,POSC)[1] << "]\n";
	// gradients.getNormalized().printField("GradientsN.dat");
	VectorField deltaField(rows, cols);
	int iteration = 1;
	compression_params.push_back(CV_IMWRITE_JPEG_QUALITY);
	compression_params.push_back(95);
	while(iteration <= 50) {
		// for(int i = POSR - 1; i < POSR + 2; i ++) {
		// 	for(int j = POSC - 1; j < POSC + 2; j++) {
		// 		std::cout << (int)deformedImage_.at<float>(i,j) << "\t";
		// 	}
		// 	std::cout << "\n";
		// }
		time(&startTime);
		deltaField = newDeltaField(gradients);
		// if(iteration != 1 && stopCriteria(norm, displField, deltaField)) break;
		updateDisplField(displField, deltaField);
		updateDeformedImage(displField);
		double iterTime = getIterationTime(startTime);
		printVFN(displField, deltaField ,iteration);
		printVFI(displField, iteration);
		printDeformedImage(iteration);
		std::cout << "Iteration " << iteration << " took " << iterTime << " seconds.\n";
		iteration++;
	}
	std::cout << "termino rapa\n";
}

bool Demons::correlationCoef() {
	cv::MatND staticImageHist, deformedImageHist;
	int channels[] = {0};
	int histSize = 256;
    float range[] = { 0, 255 };
    const float* ranges[] = { range };
	calcHist(&staticImage_, 1, channels, cv::Mat(), staticImageHist, 1, &histSize, ranges);
	calcHist(&deformedImage_, 1, channels, cv::Mat(), deformedImageHist, 1, &histSize, ranges);
	// std::cout << cv::compareHist(staticImageHist, deformedImageHist, CV_COMP_CORREL) << "\n";
	return cv::compareHist(staticImageHist, deformedImageHist, CV_COMP_CORREL) >= CORRCOEFcriteria;
}

bool Demons::rootMeanSquareError() {
	int rows = deformedImage_.rows, cols = deformedImage_.cols;
	double rmse = 0.0;
	for(int row = 0; row < rows; row++) {
		const float* dRow = deformedImage_.ptr<float>(row);
		uchar* mRow = staticImage_.ptr(row);
		for(int col = 0; col < cols; col++) {
			rmse += (dRow[col]-mRow[col])*(dRow[col]-mRow[col]);
		}
	}
	rmse = std::sqrt(rmse/(rows*cols));
	return rmse < RMSEcriteria;
}

bool Demons::stopCriteria(std::vector<float> &norm, VectorField displField, VectorField deltaField) {
	float newNorm = deltaField.sumOfAbs()/displField.sumOfAbs();
	// for (int i = 0; i < 10; i++) std::cout << "norm[" << i << "] = " << norm[i] << "\n";
	// std::cout << "deltaField.sumOfAbs() = " << deltaField.sumOfAbs() << "\n";
	// std::cout << "displField.sumOfAbs() = " << displField.sumOfAbs() << "\n";
	// std::cout << "newNorm = " << newNorm << "\n";
	// std::cout << "newNorm - norm = " << std::abs((newNorm - norm[9])) << "\n";
	if (std::abs((newNorm - norm[9])) > STOPcriteria) {
		for (int i = 9; i >= 0; i--) norm[i] = norm[i-1];
		norm[0] = newNorm;
		return false;
	}
	return true;
}

void Demons::updateDeformedImage(VectorField displField) {
	int rows = displField.getRows(), cols = displField.getCols();
	for(int row = 0; row < rows; row++) {
		float* deformedImageRow = deformedImage_.ptr<float>(row);
		for(int col = 0; col < cols; col++) {
			std::vector<float> displVector = displField.getVectorAt(row, col);
			double newRow = row - displVector[0];
			double newCol = col - displVector[1];
			bool print = false;
			// if (col == POSC && row == POSR) {
			// 	std::cout << "newRow = " << newRow << " newCol = " << newCol << "\n";
			// 	print = true;
			// }
			deformedImageRow[col] = Interpolation::fbilinearInterpolation(movingImage_, newRow, newCol, print);
			// if (col == POSC && row == POSR) std::cout << "Interpolation = " << deformedImageRow[col] << "\n";
		}
	}
}

void Demons::updateDisplField(VectorField displField, VectorField deltaField) {
	// deltaField.applyGaussianFilter();
	// std::cout << "DisplField[" << POSR << "][" << POSC << "] = [" << displField.getVectorAt(POSR,POSC)[0] << "][" << displField.getVectorAt(POSR,POSC)[1] << "]\n";
	displField.add(deltaField);
	// std::cout << "DisplField[" << POSR << "][" << POSC << "] = [" << displField.getVectorAt(POSR,POSC)[0] << "][" << displField.getVectorAt(POSR,POSC)[1] << "]\n";
	displField.applyGaussianFilter();
}

VectorField Demons::newDeltaField(VectorField gradients) {
	int rows = gradients.getRows(), cols = gradients.getCols();
	VectorField deltaField(rows, cols);
	VectorField gradientDeformed = findGradSobel(deformedImage_);
	for(int row = 0; row < rows; row++) {
		const float* dRow = deformedImage_.ptr<float>(row);
		uchar* sRow = staticImage_.ptr(row);
		for(int col = 0; col < cols; col++) {
			std::vector<float> sGrad = gradients.getVectorAt(row, col);
			std::vector<float> dGrad = gradientDeformed.getVectorAt(row, col);
			float diff = dRow[col] - sRow[col];
			// float k = std::sqrt(SPACING);
			float denominator = (diff*diff) + (sGrad[0]+dGrad[0])*(sGrad[0]+dGrad[0]) + (sGrad[1]+dGrad[1])*(sGrad[1]+dGrad[1]);
			// if (row == POSR && col == POSC) {
			// 	std::cout << "deformedImage_[" << row << "][" << col << "] = " << dRow[col] << "\n";
			// 	std::cout << "staticImage_[" << row << "][" << col << "] = " << (int)sRow[col] << "\n";
			// 	std::cout << "Diff = " << diff << "\n";
			// 	std::cout << "Gradient[" << POSR << "][" << POSC << "] = [" << gradients.getVectorAt(POSR,POSC)[0] << "][" << gradients.getVectorAt(POSR,POSC)[1] << "]\n";
			// 	std::cout << "Denominator = " << denominator << "\n";
			// }
			if (denominator > 0.0) {
				float rowValue = 2*(sGrad[0]+dGrad[0])*diff/denominator;
				float colValue = 2*(sGrad[1]+dGrad[1])*diff/denominator;
				deltaField.updateVector(row, col, rowValue, colValue);
			}
		}
	}
	// std::cout << "DeltaField[" << POSR << "][" << POSC << "] = [" << deltaField.getVectorAt(POSR,POSC)[0] << "][" << deltaField.getVectorAt(POSR,POSC)[1] << "]\n";
	return deltaField;
}

VectorField Demons::findGrad(cv::Mat image) {
	int rows = image.rows, cols = image.cols;
	cv::Mat kernelRow = cv::Mat::zeros(3, 3, CV_32F); 
	cv::Mat kernelCol = cv::Mat::zeros(3, 3, CV_32F);
	cv::Mat gradRow = cv::Mat::zeros(rows, cols, CV_32F);
	cv::Mat gradCol = cv::Mat::zeros(rows, cols, CV_32F);
	kernelRow.at<float>(1,0) = -1;
	kernelRow.at<float>(1,2) = 1;
	kernelCol.at<float>(0,1) = -1;
	kernelCol.at<float>(2,1) = 1;
	filter2D(image, gradRow, CV_32F , kernelRow);
	filter2D(image, gradCol, CV_32F , kernelCol);
	VectorField grad(gradRow, gradCol);
	return grad;
}

VectorField Demons::findGradSobel(cv::Mat image) {
	int rows = image.rows, cols = image.cols;
	cv::Mat gradRow = cv::Mat::zeros(rows, cols, CV_32F);
	cv::Mat gradCol = cv::Mat::zeros(rows, cols, CV_32F);
	cv::Sobel(image, gradRow, CV_32F, 0, 1);
	cv::Sobel(image, gradCol, CV_32F, 1, 0);
	// gradCol = normalizeSobelImage(gradCol);
	// gradRow = normalizeSobelImage(gradRow);
	VectorField grad(gradRow, gradCol);
	return grad;
}

cv::Mat Demons::normalizeSobelImage(cv::Mat sobelImage) {
	double minVal, maxVal;
	minMaxLoc(sobelImage, &minVal, &maxVal); //find minimum and maximum intensities
	cv::Mat normalized;
	sobelImage.convertTo(normalized, CV_32F, 255.0/(maxVal - minVal), -minVal * 255.0/(maxVal - minVal));
	return normalized;
}

void Demons::printDeformedImage(int iteration) {
	std::string imageName("Iteration");
	std::ostringstream converter;
	converter << iteration;
	imageName += converter.str() + ".jpg";
    imwrite(imageName.c_str(), deformedImage_, compression_params);
}

void Demons::printVFN(VectorField vectorField, VectorField deltaField, int iteration) {
	std::string filename("VFN-Iteration");
	std::ostringstream converter;
	converter << iteration;
	filename += converter.str() + ".dat";
	VectorField normalized = vectorField.getNormalized();
	normalized.printField(filename.c_str());
	std::string vfName("VectorFieldInformation.info");
	vectorField.printFieldInfos(vfName, iteration);
	std::string dfName("DeltaFieldInformation.info");
	deltaField.printFieldInfos(dfName, iteration);
}

void Demons::printVFI(VectorField vectorField, int iteration) {
	vectorField.printFieldImage(iteration, compression_params);
}

double Demons::getIterationTime(time_t startTime) {
	double iterationTime = difftime(time(NULL), startTime);
	totalTime += iterationTime;
	return iterationTime;
}

cv::Mat Demons::getRegistration() {
	return deformedImage_;
}

VectorField Demons::getDisplField() {
	return displField;
}