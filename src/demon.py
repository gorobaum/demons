import sys
import callDemon
import numpy as np
import nibabel as nib

if len(sys.argv) <= 3:
	print "Please use as <Original Image Header> <Modified Image Header> <New Image Name>"
	sys.exit()
else:
	try:
		staticImage = nib.load(sys.argv[1])
		movingImage = nib.load(sys.argv[1])
	except:
		print "Unable to load image!"
	outputData = np.ones(staticImage.get_data().shape, staticImage.get_data().dtype)
	callDemon.calldemon(staticImage, staticImage, outputData)
	outputImage = nib.Nifti1Image(outputData, staticImage.get_affine())
	print outputData[0][0][0]
	outputImage.to_filename(sys.argv[3]);
