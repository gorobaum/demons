
#include <cstdlib>
#include <cstdio>
#include <cstring>

#include "CImg.h"
#include "deform.h"
#include "demons.h"

using namespace cimg_library;

int main(int argc, char** argv) {
	if (strcmp(argv[1], "-d") == 0) {
		CImg<float> image(argv[2]);
		Deform deform(image);
		const CImg<float> modified = deform.applySinDeformation();
		modified.save_jpeg("modified-teste.jpg");
	} else {
		CImg<float> staticImage(argv[1]);
		CImg<float> movingImage(argv[2]);
		Demons demons(staticImage, movingImage);
		CImg<float> registred = demons.demons();
		registred.save_jpeg("registred.jpg");
	}
	return 0;
}