
from imutils import paths
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os

dataset = 'dataset/train'
embeddings = 'output/embeddings.pickle'
detector = 'face_detection_model'

embedding_model = 'openface_nn4.small2.v1.t7'
protoPath = os.path.sep.join(detector, "deploy.prototxt"])
modelPath = os.path.sep.join(detector,
	"res10_300x300_ssd_iter_140000.caffemodel"])
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)



embedder = cv2.dnn.readNetFromTorch(embedding_model)



imagePaths = list(paths.list_images(dataset))

knownEmbeddings = []
knownNames = []

# initialize the total number of faces processed
total = 0

# loop over the image paths
for (i, imagePath) in enumerate(imagePaths):
	# extract the person name from the image path
	print("Processing image {}/{}".format(i + 1,
		len(imagePaths)))
	name = imagePath.split(os.path.sep)[-2]

	
	image = cv2.imread(imagePath)
	image = imutils.resize(image, width=600)
	(h, w) = image.shape[:2]

	
	imageBlob = cv2.dnn.blobFromImage(
		cv2.resize(image, (300, 300)), 1.0, (300, 300),
		(104.0, 177.0, 123.0), swapRB=False, crop=False)

	detector.setInput(imageBlob)
	detections = detector.forward()

	
	if len(detections) > 0:
		
		i = np.argmax(detections[0, 0, :, 2])
		confidence = detections[0, 0, i, 2]

		
		if confidence > args["confidence"]:
			
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			
			face = image[startY:endY, startX:endX]
			(fH, fW) = face.shape[:2]

			
			if fW < 20 or fH < 20:
				continue

			
			faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
				(96, 96), (0, 0, 0), swapRB=True, crop=False)
			embedder.setInput(faceBlob)
			vec = embedder.forward()

			
			knownNames.append(name)
			knownEmbeddings.append(vec.flatten())
			total += 1


print("Serializing {} encodings...".format(total))
data = {"embeddings": knownEmbeddings, "names": knownNames}
f = open(args["embeddings"], "wb")
f.write(pickle.dumps(data))
f.close()
