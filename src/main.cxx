
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <iostream>

#include <opencv2/opencv.hpp>

int main(int argc, char** argv) {
	if (argc <= 2) {
		std::cout << "Precisa passar o nome dos arquivos coração! \n";
		return 0;
	}
	if (strcmp(argv[1], "-d") == 0) {
		std::cout << "Dae deformations \n";
	} else {
		std::cout << "Dae demonsations \n";
	}
	return 0;
}