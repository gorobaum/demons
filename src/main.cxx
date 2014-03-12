
#include <cstdlib>
#include <cstdio>

#include "CImg.h"
#include "deform.h"

using namespace cimg_library;

int main(int argc, char** argv) {
	CImg<float> image("teste.jpg");
	Deform deform(image);
	CImg<float> deformated = deform.applySinDeformation();
	CImgDisplay main_disp(deformated,"Click a point");
	while (!main_disp.is_closed()) {}
	return 0;
}