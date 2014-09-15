#ifndef DEMONS_VECTORFIELD_H_
#define DEMONS_VECTORFIELD_H_

#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

class VectorField {
	public:
		VectorField(cv::Mat &vectorRow, cv::Mat &vectorCol);
		VectorField(int rows, int cols);
		std::vector<double> getVectorAt(int row, int col);
		void updateVector(int row, int col, double rowValue, double colValue);
		void applyGaussianFilter();
		VectorField getNormalized();
		void printField(std::string filename);
		void printFieldInfos(std::string filename, int iteration);
		void printFieldImage(int iteration, std::vector<int> compression_params);
		void add(VectorField adding);
		double sumOfAbs();
		int getRows();
		int getCols();
		cv::Mat getMatRow();
		cv::Mat getMatCol();
	private:
		cv::Mat vectorCol_;
		cv::Mat vectorRow_;
		double vectorNorm(std::vector<double> v);
		std::vector<double> getInfos();
		int rows_;
		int cols_;
};

#endif
