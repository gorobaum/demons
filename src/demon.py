import sys
import nibabel as nib

if len(sys.argv) <= 2:
	print "Please use as <Original Image Header> <Modified Image Header>"
	sys.exit()
else:
	try:
		staticHeader = nib.load(sys.argv[1])
	except:
		print "Unable to load image!"

	data = staticHeader.get_data()

	print data[45][45]

