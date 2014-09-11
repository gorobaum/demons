#include "debug.h"

void Debug::debugIteration(int iteration) {
	printImageNeighbourhood(deformedImage_, "Deformed Image in the iteration "+iteration);
	printVectorFieldPosition(deltaField_, "DeltaField");
	printVectorFieldPosition(displField_, "DisplacementField");
	saveVectorField(displField_, deltaField_ ,iteration);
	printFieldImage(displField_, iteration);
	std::string filename = "DeformedImage"+iteration+".png";
	saveImage(deformedImage, filename);
}

void Debug::printImageNeighbourhood(cv::Mat image, std::string imageInformation) {
	std::cout << imageInformation + "\n";
	for(int i = printRow - 1; i < printRow + 2; i ++) {
		for(int j = printCol - 1; j < printCol + 2; j++) {
			std::cout << (int)image.at<uchar>(i,j) << "\t";
		}
		std::cout << "\n";
	}
}

void Debug::printVectorFieldPosition(VectorField vectorField, std::string vectorFieldName) {
	std::cout << vectorFieldName + "[" << printRow << "][" << printCol << "] = [" << vectorField.getVectorAt(printRow,printCol)[0] << "][" << vectorField.getVectorAt(printRow,printCol)[1] << "]\n";
}

void Debug::saveImage(cv::Mat image, std::string filename) {
    imwrite(filename.c_str(), image, compression_params);
}

void Debug::saveVectorField(VectorField displField, VectorField deltaField, int iteration) {
	std::string filename("VFN-Iteration");
	std::ostringstream converter;
	converter << iteration;
	filename += converter.str() + ".dat";
	savePlot(displField.getNormalized(), filename.c_str());
	std::string vfName("VectorFieldInformation.info");
	saveStatisticInformation(displField, vfName, iteration);
	std::string dfName("DeltaFieldInformation.info");
	saveStatisticInformation(deltaField, dfName, iteration);
}

void Debug::saveStatisticInformation(VectorField vectorField, std::string filename, int iteration) {
	std::ofstream myfile;
	if (iteration <= 1) myfile.open(filename);
	else myfile.open(filename, std::ios_base::app);
	myfile << "Iteration " << iteration << "\n";
	std::vector<double> results = getInfos(vectorField);
	myfile << "Min = " << results[0] << " Max = \t" << results[1] << " Median = \t" << results[2] << " Mean = \t" << results[3] << " Standard Deviaon = \t" << results[4] << "\n";
	myfile.close();
}


void Debug::printFieldImage(VectorField vectorField, int iteration) {
	cv::Mat abs_grad_col, abs_grad_row;
	std::string filenamebase("DFI-Iteration"), flCol, flRow;
	std::ostringstream converter;
	converter << iteration;
	filenamebase += converter.str();
	flCol += filenamebase + "x.jpg";
	flRow += filenamebase + "y.jpg";
	convertScaleAbs(vectorField.getMatRow(), abs_grad_row, 255);
	convertScaleAbs(vectorField.getMatCol(), abs_grad_col, 255);
	imwrite(flRow.c_str(), abs_grad_row, compression_params);
	imwrite(flCol.c_str(), abs_grad_col, compression_params);
}

std::vector<double> Debug::getInfos(VectorField vectorField) {
	std::multiset<double> magnitudes;
	int size = (rows_*cols_);
	double max= 0.0, min = 0.0, mean = 0.0, median = 0.0, deviation = 0.0;
	for(int row = 0; row < rows_; row++) {
	    for(int col = 0; col < cols_; col++) {
	    	std::vector<float> vector = vectorField.getVectorAt(row, col);
    		double mag = std::sqrt(vector[0]*vector[0] + vector[1]*vector[1]);
    		if (max < mag) max = mag;
    		if (min > mag) min = mag;
    		mean += mag;
			magnitudes.insert(mag);
	    }
	}
	mean /= size;
	int count = 1;
	std::multiset<double>::iterator it;
	for (it=magnitudes.begin(); it!=magnitudes.end(); ++it) {
    	deviation += std::pow((*it - mean),2);
    	if (count == size/2) median = *it;
    	count++;
	}
	deviation /= size;
	deviation = std::sqrt(deviation);
	std::vector<double> results;
	results.push_back(min);
	results.push_back(max);
	results.push_back(median);
	results.push_back(mean);
	results.push_back(deviation);
	return results;
}

void Debug::savePlot(std::string filename) {
	std::ofstream myfile;
	myfile.open(filename);
	double minValCol, maxValCol;
	double minValRow, maxValRow;
	minMaxLoc(vectorCol_, &minValCol, &maxValCol);
	minMaxLoc(vectorRow_, &minValRow, &maxValRow);
	for(int row = 0; row < rows_; row++) {
	    for(int col = 0; col < cols_; col++) {
	    	std::vector<float> vector = getVectorAt(row, col);
    		double redCol = 255*(vector[1]-minValCol)/(maxValCol-minValCol);
			double blueCol = 255*(maxValCol-vector[1])/(maxValCol-minValCol);
			double redRow = 255*(vector[0]-minValRow)/(maxValRow-minValRow);
			double blueRow = 255*(maxValRow-vector[0])/(maxValRow-minValRow);
			int red = (redCol + redRow)/2;
			int blue = (blueCol + blueRow)/2;
			myfile << col << " " << (vectorCol_.rows - row) << " " << vector[1] << " " << vector[0] << " " <<  red << " " << 0 << " " << blue << "\n";
	    }
	}
	myfile.close();
}