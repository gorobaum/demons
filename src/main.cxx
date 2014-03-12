
#include <cstdlib>
#include <cstdio>

#include "CImg.h"
#include "deform.h"

using namespace cimg_library;

int main(int argc, char** argv) {
	CImg<float> image("teste.jpg");
	CImgDisplay main_disp(image,"Click a point");
	Deform deform;
	deform.applySinDeformation(image);
	while (!main_disp.is_closed()) {}
	return 0;
}