#ifndef DEMONS_VECTORFIELD_H_
#define DEMONS_VECTORFIELD_H_

#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

class VectorField {
	public:
		VectorField (cv::Mat &vectorX, cv::Mat &vectorY);
		VectorField (int rows, int cols);
		std::vector<float> getVectorAt(int row, int col);
		void updateVector(int row, int col, float xValue, float yValue);
		void applyGaussianFilter();
		VectorField getNormalized();
		void printField(std::string filename);
		void printFieldImage(int iteration, std::vector<int> compression_params);
		int getRows();
		int getCols();
	private:
		cv::Mat vectorX_;
		cv::Mat vectorY_;
		float getValue(cv::Mat image, int row, int col);
		float vectorNorm(std::vector<float> v);
		int rows_;
		int cols_;
};

#endif