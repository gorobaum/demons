
#include <cstdlib>
#include <cstdio>

#include "CImg.h"
using namespace cimg_library;

int main(int argc, char** argv) {
	if (argc != 2) {
		printf("Precisa passar o nome da imagem né coração?\n");
		return -1;
	} 
	CImg<unsigned char> image("teste.jpg");
	CImgDisplay main_disp(image,"Click a point");
	while (!main_disp.is_closed()) {}
	return 0;
}