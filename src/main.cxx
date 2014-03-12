
#include <cstdlib>
#include <cstdio>

#include "CImg.h"
#include "deform.h"

using namespace cimg_library;

int main(int argc, char** argv) {
	if (argc != 2) {
		printf("Dae, precisa passar a ibagem q vc quer usar né coração?\n");
		return 0;
	}
	CImg<float> image(argv[1]);
	Deform deform(image);
	const CImg<float> modified = deform.applySinDeformation();
	modified.save_jpeg("modified-teste.jpg");
	return 0;
}