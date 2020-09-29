

# USAGE
# python detect_faces.py --image rooster.jpg --prototxt deploy.prototxt.txt --model res10_300x300_ssd_iter_140000.caffemodel

# import the necessary packages
import numpy as np
import argparse
import cv2
import os


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
#ap.add_argument("-c", "--confidence", type=float, default=0.5,
#	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# load the input image and construct an input blob for the image
# by resizing to a fixed 300x300 pixels and then normalizing it
image = cv2.imread(args["image"])
(h, w) = image.shape[:2]
#blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
	(300, 300), (104.0, 177.0, 123.0))

# pass the blob through the network and obtain the detections and
# predictions
print("[INFO] computing object detections...")
net.setInput(blob)
detections = net.forward()

confidence = []
# loop over the detections
for i in range(0, detections.shape[2]):
	confidence.append(detections[0, 0, i, 2])

index_max_conf = confidence.index(max(confidence))
max_conf = max(confidence)

print("\nlen {}, index {}, max {}\n".format(len(confidence), index_max_conf, max_conf))
print(confidence[:5])	
# compute the (x, y)-coordinates of the bounding box for the object
box = detections[0, 0, index_max_conf, 3:7] * np.array([w, h, w, h])
(startX, startY, endX, endY) = box.astype("int")

# draw the bounding box of the face along with the associated
# probability
org = image.copy()

text = "{:.2f}%".format(max_conf * 100)
y = startY - 10 if startY - 10 > 10 else startY + 10

startX, startY, endX, endY = startX-2, startY-10, endX+2, endY+10

cv2.rectangle(image, (startX, startY), (endX, endY),
	(0, 0, 255), 2)

crop_img = org[startY:endY, startX:endX]
cv2.imwrite("C:/Users/Lucimara Santos/Documents/LABIC/face-detection-crop/test/crop.jpg", crop_img)

cv2.putText(image, text, (startX, y),
cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

# show the output image
print("SIZE: ", crop_img.shape[:2])

crop_img = cv2.resize(crop_img, (300, 300))

cv2.imshow("Output", image)
cv2.imshow("crop", crop_img)
cv2.waitKey(0)

