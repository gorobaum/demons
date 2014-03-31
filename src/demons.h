#ifndef DEMONS_DEMONS_H_
#define DEMONS_DEMONS_H_

#include <vector>
#include <array>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

class Demons {
	public:
		explicit Demons (cv::Mat staticImage, cv::Mat movingImage):
			staticImage_(staticImage), movingImage_(movingImage) {}
		cv::Mat demons();
	private:
		cv::Mat staticImage_;
		cv::Mat movingImage_;
		time_t startTime;
		float totalTime;
		struct Vector {
			float x;
			float y;
			Vector(float a=0, float b=0):
					 x(a), y(b){}
		};
		typedef std::vector<Vector> Field;
		Field findGrad();
		Field createField();
		double getIterationTime(time_t startTime);
		cv::Mat normalizeSobelImage(cv::Mat sobelImage);
		void updateDisplField(uchar* deformedRow, Demons::Vector displacement, Demons::Vector gradient, int row, int col);
};

#endif