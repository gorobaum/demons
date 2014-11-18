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
	aoutputData = np.ones(staticImage.get_data().shape, staticImage.get_data().dtype)
	soutputData = np.ones(staticImage.get_data().shape, staticImage.get_data().dtype)
	spacing = staticImage.get_header()['pixdim'][1:4]
	start_time = time.time()
	callDemon.calldemon(staticImage, movingImage, aoutputData, soutputData, spacing[0], spacing[1], spacing[2], int(iterations), float(kernelSize), float(deviation))
	print "Total execution time: ", (time.time() - start_time)
	outputImage = nib.Nifti1Image(aoutputData, staticImage.get_affine(), staticImage.get_header())
	outputImage.to_filename("a"+outputName);
	outputImage = nib.Nifti1Image(soutputData, staticImage.get_affine(), staticImage.get_header())
	outputImage.to_filename("s"+outputName);

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