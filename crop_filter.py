import numpy as np
import argparse
import cv2
import os
from imutils import paths
import get_image_infos as gii

filtered_crop_face = "C:/*****/face-detection-crop/filtered-crop/face"
filtered_crop_not = "C:/*****/face-detection-crop/filtered-crop/not"
filtered_txt = "C:/*****/face-detection-crop/filtered-crop/filter-crop.txt"
not_filtered_txt = "C:/*****/face-detection-crop/filtered-crop/not-filter-crop.txt"

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input folder")
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")

ap.add_argument("-ff", "--filtered_crop_face", default=filtered_crop_face,
	help="path to output filtered faces")
ap.add_argument("-nff", "--filtered_crop_not", default=filtered_crop_not,
	help="path to output crops")
ap.add_argument("-ftxt", "--filtered_txt", default=filtered_txt,
	help="path to output filtered faces txt file")
ap.add_argument("-ntxt", "--not_filtered_txt", default=not_filtered_txt,
	help="path to output not filtered faces txt file")

ap.add_argument("-c", "--confidence", type=float, default=0.9,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

print("[INFO] quantifying faces...")
imagePaths = list(paths.list_images(args["dataset"]))

confs_dict = []

filtered_faces_infos = []
NA_infos = []

shapes_filter = []
shapes_NA = []

for (num, imagePath) in enumerate(imagePaths):
	print("[INFO] processing image {}/{}".format(num + 1, len(imagePaths)))
	print(imagePath)

	image = cv2.imread(imagePath)
	(h, w) = image.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
		(300, 300), (104.0, 177.0, 123.0))

	net.setInput(blob)
	detections = net.forward()

	confidence = []
	for i in range(0, detections.shape[2]):
		confidence.append(detections[0, 0, i, 2])

    # get maximum confidence (since the cropped image has only one face)
	max_conf = max(confidence) 
    
	text = "{:.2f}%".format(max_conf * 100)
    
	confs_dict.append((str(num) + ".jpg", text))

	if max_conf > args["confidence"]:   
		name = "filter"+str(num)+".jpg"
		print("Image was filtered\n")
        
		cv2.imwrite(os.path.join(args["filtered_crop_face"], name), image)
        
        info_line = gii.get_infos(os.path.join(args["filtered_crop_face"], name), 
								 image, str(text))
		filtered_faces_infos.append(info_line)

		shapes_filter.append(image.shape[:2])

	else:
		name = "NA"+str(num)+".jpg"
		print("NOT FACE\n")
		cv2.imwrite(os.path.join(args["filtered_crop_not"], name), image)
		NA_infos.append(gii.get_infos(os.path.join(args["filtered_crop_not"], name), image, 
						str(text)))

		shapes_NA.append(image.shape[:2])


gii.create_txt(args["filtered_txt"], filtered_faces_infos, title="filtered faces infos", shapes_filter)

gii.create_txt(args["not_filtered_txt"], NA_infos, title="not filtered faces infos", shapes_NA)


