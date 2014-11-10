import sys
import callDemon
import numpy as np
import nibabel as nib
import time

def runDemon(staticImageName, movingImageName, outputName, iterations, kernelSize, deviation, execution):
	try:
		staticImage = nib.load(staticImageName)
		movingImage = nib.load(movingImageName)
	except:
		print "Unable to load images in execution", execution
	outputData = np.ones(staticImage.get_data().shape, staticImage.get_data().dtype)
	spacing = staticImage.get_header()['pixdim'][1:4]
	start_time = time.time()
	callDemon.calldemon(staticImage, movingImage, outputData, spacing[0], spacing[1], spacing[2], int(iterations), float(kernelSize), float(deviation))
	print "--- seconds ---", (time.time() - start_time)
	outputImage = nib.Nifti1Image(outputData, staticImage.get_affine())
	outputImage.to_filename(outputName);

if len(sys.argv) <= 1:
	print "Please use as <Conf file path>"
	sys.exit()
else:
	try:
		confFile = open(sys.argv[1], "r")
	except:
		print "Unable to load config file!"
	execution = 1
	while True:
		staticImageName = confFile.readline().rstrip('\n')
		if staticImageName == "":
			break
		movingImageName = confFile.readline().rstrip('\n')
		if movingImageName == "":
			break
		outputName = confFile.readline().rstrip('\n')
		if outputName == "":
			break
		iterations = confFile.readline().rstrip('\n')
		if iterations == "":
			break
		kernelSize = confFile.readline().rstrip('\n')
		if kernelSize == "":
			break
		deviation = confFile.readline().rstrip('\n')
		if deviation == "":
			break
		runDemon(staticImageName, movingImageName, outputName, iterations, kernelSize, deviation, execution)
		execution += 1