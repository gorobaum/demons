#ifndef DEMONS_VECTORFIELD_H_
#define DEMONS_VECTORFIELD_H_

#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

class VectorField {
	public:
		VectorField (cv::Mat vectorX, cv::Mat vectorY);
		VectorField (int rows, int cols);
		std::vector<uchar> getVectorAt(int row, int col);
		void updateVector(int row, int col, uchar xValue, uchar yValue);
		void applyGaussianFilter();
	private:
		cv::Mat vectorX_;
		cv::Mat vectorY_;
		uchar* ptrRowX;
		uchar* ptrRowY;
		int currentRow;
		void updateCurrentRow(int newRow);
		void startRow();
};







#endif