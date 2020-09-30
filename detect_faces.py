# python detect_faces.py --image rooster.jpg --prototxt deploy.prototxt.txt --model res10_300x300_ssd_iter_140000.caffemodel

# import the necessary packages
import numpy as np
import argparse
import cv2
import os
from imutils import paths
import get_image_infos as gii

bbox_folder = "C:/Users/*****/face-detection-crop/first-filter/bboxes"
multi_crop_folder = "C:/Users/*****/face-detection-crop/first-filter/multi-crop-raw"
not_face_folder = "C:/*****/face-detection-crop/first-filter/not"
txt_path = "C:/*****/face-detection-crop/first-filter/raw-crop.txt"

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset folder")
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")

ap.add_argument("-b", "--bbox_folder", default=bbox_folder,
	help="path to output bboxes folder")
ap.add_argument("-r", "--multi_crop_folder", default=multi_crop_folder,
	help="path to output crops folder")
ap.add_argument("-nt", "--not_face_folder", default=not_face_folder,
	help="path to output not faces folder")
ap.add_argument("-txt", "--txt_path", default=txt_path,
	help="path to output txt file")

ap.add_argument("-c", "--confidence", type=float, default=0.15,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# load serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

print("[INFO] quantifying faces...")
imagePaths = list(paths.list_images(args["dataset"]))

# number of crop images 
num_crop = 0 

# list containing all image infos, each one in formeted line (gii.get_infos)
file_infos = [] 

# list containing all shapes used to calculate the median, max and min resolution
shapes = [] 

for (num, imagePath) in enumerate(imagePaths):
    
	print("[INFO] processing image {}/{}".format(num + 1, len(imagePaths)))

	total_faces = 0 # Total number of possible faces in one image

	# load the input image and construct an input blob for the image
	# by resizing to a fixed 300x300 pixels and then normalizing it
	image = cv2.imread(imagePath)
	original_image = image.copy()

	(h, w) = image.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
		                        (300, 300), (104.0, 177.0, 123.0))

	# pass the blob through the network and obtain the detections and
	# predictions
	net.setInput(blob)
	detections = net.forward()

	# loop over the detections
	for i in range(0, detections.shape[2]):

		# extract the confidence (i.e., probability) associated with the
		# prediction
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by  `confidence`
		if confidence > args["confidence"]:

			# compute the (x, y)-coordinates of the bounding box for the
			# object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")
	 
			# draw the bounding box of the face along with the associated
			# probability
			text = "{:.2f}%".format(confidence * 100)
            
			y = startY - 10 if startY - 10 > 10 else startY + 10
			cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)

			cv2.putText(image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, 
                       (0, 0, 255), 2)

			# set the name of each image in the first filter stage
			name_crop = "raw"+str(num_crop) +".jpg"
            # get a copy of the image in order to keep the original preserverd
			org = original_image.copy() 
			
            # increase the bbox size
			startX, startY, endX, endY = startX-2, startY-10, endX+2, endY+10
            # crop image based on bbox coords
			crop_img = org[startY:endY, startX:endX] 

            # check if the croped image it's not empty
			if crop_img.shape[0] != 0 and crop_img.shape[1] != 0: 
                # save crop in folder
				cv2.imwrite(os.path.join(args["multi_crop_folder"], name_crop), 
                            crop_img) 
                
				num_crop = num_crop + 1
				total_faces = total_faces + 1
                
                # save image shape
				shapes.append(crop_img.shape[:2])
                # save the line correspondingly to the image
                info_line = gii.get_infos(os.path.join(args["multi_crop_folder"],
                                          name_crop), crop_img, str(text))
                # save the formated line into the list
				file_infos.append(info_line)

		else: # nothing above the standard confidence was detected
			break

	if total_faces > 0: # the image has faces
		name = "img" + str(num) + " - boxes" + str(total_faces) + ".jpg" 
		cv2.imwrite(os.path.join(args["bbox_folder"], name), image)

	else: # no faces detected
		cv2.imwrite(os.path.join(args["not_face_folder"], str(num) + ".jpg"), image)

# create the .txt file using the file_infos list 
# (which contains the Infos about all the images with detected faces)
gii.create_txt(args["txt_path"], file_infos, title="first filter (raw) infos", 
               shapes=shapes)







