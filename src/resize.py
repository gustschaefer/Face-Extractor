import cv2 
import os
import glob
import argparse
from imutils import paths

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input_dir", required=True,
	help="path to input folder")
ap.add_argument("-o", "--output_dir", required=True)
ap.add_argument("-s", "--new_size", default=(300, 300),
	help="dimensions of resized images")
args = vars(ap.parse_args())

if os.path.isdir(args["output_dir"]) == False:
	os.mkdir(args["output_dir"]) 

folderlen = len(args["input_dir"])
imagePaths = list(paths.list_images(args["input_dir"]))

for (num, imagePath) in enumerate(imagePaths):
	image = cv2.imread(imagePath)
	print("[INFO] processing image {}/{}, size: {}, new size: {}".format(num +1 , len(imagePaths), 
		image.shape[:2], args["new_size"]))

	image = cv2.resize(image, args["new_size"])
	cv2.imwrite(args["output_dir"] + imagePath[folderlen:], image)
	#cv2.imshow('image', image)
	#cv2.waitKey(50)
#cv2.destroyAllWindows()