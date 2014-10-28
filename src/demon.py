import sys
import callDemon
import nibabel as nib

if len(sys.argv) <= 2:
	print "Please use as <Original Image Header> <Modified Image Header>"
	sys.exit()
else:
	try:
		staticHeader = nib.load(sys.argv[1])
	except:
		print "Unable to load image!"
	print staticHeader.get_data()[0][0][0]
	callDemon.calldemon(staticHeader)