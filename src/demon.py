import sys
import callDemon
import nibabel as nib

if len(sys.argv) <= 2:
	print "Please use as <Original Image Header> <Modified Image Header>"
	sys.exit()
else:
	try:
		staticImage = nib.load(sys.argv[1])
		movingImage = nib.load(sys.argv[1])
	except:
		print "Unable to load image!"
	print staticImage.get_data()[0][1]
	print staticImage.get_data().dtype
	callDemon.calldemon(staticImage, movingImage)