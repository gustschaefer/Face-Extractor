# USAGE
# python detect_faces.py --image rooster.jpg --prototxt deploy.prototxt.txt --model res10_300x300_ssd_iter_140000.caffemodel

# import the necessary packages
import numpy as np
import argparse
import cv2
import os
from imutils import paths

max_crop_folder = "C:/Users/Lucimara Santos/Documents/LABIC/face-detection-crop/max-crop"

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input folder")
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
ap.add_argument("-cf", "--max_crop_folder", default = max_crop_folder,
	help="path to salve image crops")
args = vars(ap.parse_args())

def get_bbox(image, index_max_conf):
	box = detections[0, 0, index_max_conf, 3:7] * np.array([w, h, w, h])
	(startX, startY, endX, endY) = box.astype("int")

	org = image.copy()

	text = "{:.2f}%".format(max_conf * 100)
	print("Confidence: {}".format(text))

	y = startY - 10 if startY - 10 > 10 else startY + 10
	cv2.rectangle(image, (startX, startY), (endX, endY),
		(0, 0, 255), 2)

	return image

print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

print("[INFO] quantifying faces...")
imagePaths = list(paths.list_images(args["dataset"]))

for (num, imagePath) in enumerate(imagePaths):
	print("[INFO] processing image {}/{}".format(num + 1, len(imagePaths)))
	print(imagePath)

	image = cv2.imread(imagePath)
	(h, w) = image.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
		(300, 300), (104.0, 177.0, 123.0))

	net.setInput(blob)
	detections = net.forward()

	# loop over the detections
	confidence = []
	for i in range(0, detections.shape[2]):
		confidence.append(detections[0, 0, i, 2])

	index_max_conf = confidence.index(max(confidence)) 
	max_conf = max(confidence) # the region with the highest prob. to have a face

	#cpy = image.copy()
	#get_bbox(cpy, index_max_conf)

	crop_img = image[startY:endY, startX:endX]

	name = str(num) + ".jpg"
	cv2.imwrite(os.path.join(args["max_crop_folder"], name), crop_img)


