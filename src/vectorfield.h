#ifndef DEMONS_VECTORFIELD_H_
#define DEMONS_VECTORFIELD_H_

#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

class VectorField {
	public:
		VectorField(cv::Mat &vectorRow, cv::Mat &vectorCol);
		VectorField(int rows, int cols);
		std::vector<float> getVectorAt(int row, int col);
		void updateVector(int row, int col, float rowValue, float colValue);
		void applyGaussianFilter();
		VectorField getNormalized();
		void printField(std::string filename);
		void printFieldInfos(std::string filename, int iteration);
		void printFieldImage(int iteration, std::vector<int> compression_params);
		void add(VectorField adding);
		float sumOfAbs();
		int getRows();
		int getCols();
	private:
		cv::Mat vectorCol_;
		cv::Mat vectorRow_;
		float getValue(cv::Mat image, int row, int col);
		float vectorNorm(std::vector<float> v);
		std::vector<double> getInfos();
		int rows_;
		int cols_;
};

#endif