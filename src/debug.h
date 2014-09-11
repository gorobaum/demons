#ifndef DEMONS_DEBUG_H_
#define DEMONS_DEBUG_H_

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <string>

#include "vectorfield.h"

class Debug {
public:
	explicit Debug(cv::Mat deformedImage, VectorField displField, VectorField deltaField):
		deformedImage_(deformedImage), displField_(displField), deltaField_(deltaField) {
				rows_ = deformedImage_.rows;
				cols_ = deformedImage_.cols;
				compression_params.push_back(CV_IMWRITE_JPEG_QUALITY);
				compression_params.push_back(95);
		}
	void printVectorField();
	void debugIteration(int iteration);
private:
	std::vector<int> compression_params;
	cv::Mat deformedImage_;
	VectorField displField_;
	VectorField deltaField_;
	int printRow;
	int printCol;
	int rows_;
	int cols_;
	void printVectorFieldPosition(VectorField vectorField, std::string vectorFieldName);
	void saveImage(cv::Mat image, std::string filename);
	void saveVectorField(VectorField displField, VectorField deltaField, int iteration);
	void saveStatisticInformation(VectorField vectorField, std::string filename, int iteration);
	void printFieldImage(VectorField vectorField, int iteration);
	std::vector<double> getInfos(VectorField vectorField);
	void printImageNeighbourhood(cv::Mat image, std::string imageInformation);
	void savePlot(std::string filename);
};

#endif