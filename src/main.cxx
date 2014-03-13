
#include <cstdlib>
#include <cstdio>
#include <cstring>

#include "CImg.h"
#include "deform.h"

using namespace cimg_library;

int main(int argc, char** argv) {
	bool deformMode = false;
	for (int args = 1; args < argc; args++) {
		if (strcmp(argv[1], "-d") == 0) deformMode = true;
	}
	if (deformMode) {
		CImg<float> image(argv[2]);
		Deform deform(image);
		const CImg<float> modified = deform.applySinDeformation();
		modified.save_jpeg("modified-teste.jpg");
	} else {

	}
	return 0;
}