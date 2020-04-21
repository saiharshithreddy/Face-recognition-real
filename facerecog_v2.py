import sys
from flask import Flask, render_template, url_for, request, jsonify
from flask_cors  import CORS, cross_origin
import numpy as np
import pandas as pd
import pickle
from sklearn.externals import joblib
from scipy.spatial import distance as dist

from imutils.video import VideoStream
import argparse
import imutils
from imutils import face_utils, paths
import time
import cv2
import os
import dlib
import json
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial import distance


app = Flask(__name__)
CORS(app)
cors = CORS(app, resources={r"/api/*": {"origins": "*"}})
app.config['CORS_HEADERS'] = 'Content-Type'


@app.route('/')
def home():
	return render_template('home.html')

# request object : name
embeddings = np.zeros(128)

@app.route('/api/register', methods=['POST'])
@cross_origin()
def register():

	# request object from frontend
	content = request.data
	jsondata = json.loads(content)
	print(jsondata)
	test_description = jsondata['username']
	# =====================================

	# Face detectors 
	shape_predictor = 'shape_predictor_68_face_landmarks.dat'
	cascade = 'haarcascade_frontalface_default.xml'
	

	# load OpenCV's Haar cascade for face detection from disk
	detector = cv2.CascadeClassifier(cascade)
	
	# for detecting facial features
	detector2 = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor(shape_predictor)

	print("Starting video stream...")
	vs = VideoStream(src=0).start()

	time.sleep(2.0)
	
	total = 0
	
	time_to_recognize = 0

	# coordinates of eyes and mouth
	(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
	(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
	
	# ===========Creating a folder to store the images captured during registration=======
	directory = test_description
	
	parent_dir = "./dataset/"

	# Path
	path = os.path.join(parent_dir, directory)

	# create a directory of the person
	try:
		os.mkdir(path)
	except:
		pass
	# =========================================

	# loop over the frames from the video stream
	while True:


		time_to_recognize += 1

		frame = vs.read()
		# this is used to save the image
		orig = frame.copy()
		frame = imutils.resize(frame, width=2000)

		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		rects = detector2(gray, 0)
		
			
		for rect in rects:
			shape = predictor(gray, rect)
			shape = face_utils.shape_to_np(shape)

			# get the coordinates of the eyes 
			leftEye = shape[lStart:lEnd]
			rightEye = shape[rStart:rEnd]

			leftEyeHull = cv2.convexHull(leftEye)
			rightEyeHull = cv2.convexHull(rightEye)
			
			# mouth is not needed. 
			# mouth = shape[mStart:mEnd]
			# mouthHull = cv2.convexHull(mouth)
			# cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)

			# Draw the green borders on eyes. 
			cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
			cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

		# display
		cv2.putText(frame, "Move your head slowly", (30,60),
		cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0,0,255),2)

		# show the output frame
		cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF

		# To collect images
		if time_to_recognize > 30:
			break
		
		# captures 30 images
		elif time_to_recognize <= 30:
			p = os.path.sep.join([path, "{}.png".format(
				str(total).zfill(5))])
			cv2.imwrite(p, orig)
			total += 1

	# do a bit of cleanup
	print("{} face images stored".format(total))

	# close the video frame
	cv2.destroyAllWindows()
	vs.stop()
	
	
	# dump the facial embeddings + names to disk
	# print("[INFO] serializing {} encodings...".format(total))
	# print(knownEmbeddings/ 30)
	# data = {"embeddings": knownEmbeddings, "names": knownNames}
	# df = pd.DataFrame.from_dict(data)
	# df.to_csv('embeddings.txt', encoding='utf-8', index=False)
	embeddings = generate_embeddings('dataset')

	# Append the feature vector to the json object and return 
	jsondata['feature_vector'] = list(embeddings)
	json_object = {}
	json_object["success"] = True
	json_object["code"] = int(200)
	json_object["content"]= str(jsondata)

	response = app.response_class(
        response=json.dumps(json_object),
        mimetype='application/json'
    )
	return response

def generate_embeddings(path_to_image):
	# Generate the embedding from the images. 

	# load our serialized face detector from disk
	# print("[INFO] loading face detector...")
	protoPath = os.path.sep.join(['face_detection_model', "deploy.prototxt"])
	modelPath = os.path.sep.join(['face_detection_model',
		"res10_300x300_ssd_iter_140000.caffemodel"])
	detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

	# load our serialized face embedding model from disk
	# print("[INFO] loading face recognizer...")
	embedding_model = 'openface_nn4.small2.v1.t7'
	embedder = cv2.dnn.readNetFromTorch(embedding_model)

	# grab the paths to the input images in our dataset
	# print("[INFO] quantifying faces...")
	print(path_to_image)
	imagePaths = list(paths.list_images(path_to_image))
	
	# initialize our lists of extracted facial embeddings and
	# corresponding people names
	knownEmbeddings = np.zeros(128)
	knownNames = ""

	# initialize the total number of faces processed
	total = 0

	# loop over the image paths
	for (i, imagePath) in enumerate(imagePaths):
		# extract the person name from the image path
		print("[INFO] processing image {}/{}".format(i + 1,
			len(imagePaths)))
		name = imagePath.split(os.path.sep)[-2]

		# load the image, resize it to have a width of 600 pixels (while
		# maintaining the aspect ratio), and then grab the image
		# dimensions
		image = cv2.imread(imagePath)
		image = imutils.resize(image, width=600)
		(h, w) = image.shape[:2]

		# construct a blob from the image
		imageBlob = cv2.dnn.blobFromImage(
			cv2.resize(image, (300, 300)), 1.0, (300, 300),
			(104.0, 177.0, 123.0), swapRB=False, crop=False)

		# apply OpenCV's deep learning-based face detector to localize
		# faces in the input image
		detector.setInput(imageBlob)
		detections = detector.forward()

		# ensure at least one face was found
		if len(detections) > 0:
			# we're making the assumption that each image has only ONE
			# face, so find the bounding box with the largest probability
			i = np.argmax(detections[0, 0, :, 2])
			confidence = detections[0, 0, i, 2]

			# ensure that the detection with the largest probability also
			# means our minimum probability test (thus helping filter out
			# weak detections)
			if confidence > 0.5:
				# compute the (x, y)-coordinates of the bounding box for
				# the face
				box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
				(startX, startY, endX, endY) = box.astype("int")

				# extract the face ROI and grab the ROI dimensions
				face = image[startY:endY, startX:endX]
				(fH, fW) = face.shape[:2]

				# ensure the face width and height are sufficiently large
				if fW < 20 or fH < 20:
					continue

				# construct a blob for the face ROI, then pass the blob
				# through our face embedding model to obtain the 128-d
				# quantification of the face
				faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
					(96, 96), (0, 0, 0), swapRB=True, crop=False)
				embedder.setInput(faceBlob)
				vec = embedder.forward()

				# add the name of the person + corresponding face
				# embedding to their respective lists
				if not knownNames:
					knownNames = name
					
				knownEmbeddings += vec.flatten()
				total += 1
	return knownEmbeddings/30

def eye_aspect_ratio(eye):
		# compute the euclidean distances between the two sets of
		# vertical eye landmarks (x, y)-coordinates
		A = dist.euclidean(eye[1], eye[5])
		B = dist.euclidean(eye[2], eye[4])

		# compute the euclidean distance between the horizontal
		# eye landmark (x, y)-coordinates
		C = dist.euclidean(eye[0], eye[3])

		# compute the eye aspect ratio
		ear = (A + B) / (2.0 * C)

		# return the eye aspect ratio
		return ear

def mouth_aspect_ratio(mouth):
	# compute the euclidean distances between the two sets of
	# vertical mouth landmarks (x, y)-coordinates
	A = dist.euclidean(mouth[2], mouth[10]) # 51, 59
	B = dist.euclidean(mouth[4], mouth[8]) # 53, 57

	# compute the euclidean distance between the horizontal
	# mouth landmark (x, y)-coordinates
	C = dist.euclidean(mouth[0], mouth[6]) # 49, 55

	# compute the mouth aspect ratio
	mar = (A + B) / (2.0 * C)

	# return the mouth aspect ratio
	return mar

#==============================predict==============================
@app.route('/api/predict', methods=['POST'])
@cross_origin()
def predict():
	content = request.data
	jsondata = json.loads(content)
	# print('request_data', jsondata)
	test_description = jsondata['username']
	# print(test_description)
	
	EYE_AR_THRESH = 0.3
	# MOUTH_AR_THRESH = 0.79
	EYE_AR_CONSEC_FRAMES = 3

	# initialize the frame counters and the total number of blinks
	COUNTER = 0
	TOTAL = 0
	# Mouth = 0
	# initialize dlib's face detector (HOG-based) and then create
	# the facial landmark predictor
	shape_predictor = 'shape_predictor_68_face_landmarks.dat'
	print("[INFO] loading facial landmark predictor...")
	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor(shape_predictor)

	# grab the indexes of the facial landmarks for the left and
	# right eye, respectively
	(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
	(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
	# (mStart, mEnd) = (49, 68)

	# start the video stream thread
	print("[INFO] starting video stream thread...")

	vs = VideoStream(src=0).start()

	time.sleep(1.0)
	path = './test/'
	similarity_score = 0
	prediction = False
	
	while True:


		# grab the frame from the threaded video file stream, resize
		# it, and convert it to grayscale
		# channels)
		frame = vs.read()
		orig = frame.copy()
		frame = imutils.resize(frame, width=1450)
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		# detect faces in the grayscale frame
		rects = detector(gray, 0)
		if not rects:
			COUNTER = 0
			TOTAL = 0
			# Mouth = 0
		# loop over the face detections
		for rect in rects:
			# determine the facial landmarks for the face region, then
			# convert the facial landmark (x, y)-coordinates to a NumPy
			# array
			shape = predictor(gray, rect)
			shape = face_utils.shape_to_np(shape)

			# extract the left and right eye coordinates, then use the
			# coordinates to compute the eye aspect ratio for both eyes
			leftEye = shape[lStart:lEnd]
			rightEye = shape[rStart:rEnd]
			leftEAR = eye_aspect_ratio(leftEye)
			rightEAR = eye_aspect_ratio(rightEye)
			
			ear = (leftEAR + rightEAR) / 2.0
			
			# compute the convex hull for the left and right eye, then
			# visualize each of the eyes
			leftEyeHull = cv2.convexHull(leftEye)
			rightEyeHull = cv2.convexHull(rightEye)
			
			cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
			cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
			
			# check to see if the eye aspect ratio is below the blink
			# threshold, and if so, increment the blink frame counter
			if ear < EYE_AR_THRESH:
				COUNTER += 1

			# otherwise, the eye aspect ratio is not below the blink
			# threshold

			else:
				# if the eyes were closed for a sufficient number of
				# then increment the total number of blinks
				if COUNTER >= EYE_AR_CONSEC_FRAMES:
					TOTAL += 1



				# reset the eye frame counter
				COUNTER = 0
			

			cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 30),
				cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 2)
			
			if TOTAL >= 3:
				cv2.putText(frame, "Authentication in progress, Please wait", (100, 300),
				cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 2)
				p = os.path.sep.join([path, "test.png"])
				cv2.imwrite(p, orig)
				predicted_embedding = generate_embeddings('test')
				
				# for testing
				c = generate_embeddings('dataset')
				# print(predicted_embedding)
				
				predicted_embedding = predicted_embedding.reshape(1,128)
				
				c = c.reshape(1,128)
				# Compute the similarity between two feature vectors
				similarity_score = cosine_similarity(predicted_embedding,c)
			
				print(similarity_score[0][0])
				print("recognition done")
				prediction = True
				
		if prediction:
			break

		# show the frame
		cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF

		# if the `q` key was pressed, break from the loop
		if key == ord("q"):
			break

	# do a bit of cleanup
	cv2.destroyAllWindows()
	vs.stop()
	json_object = {}
	print("matching")
	
	if similarity_score[0][0] > 0.60 :
		# return data
		print("same person")
		json_object["success"] = True
		json_object["code"] = 200

	else:
		print("different person")
		json_object["success"] = False
		json_object["code"] = 400

	response = app.response_class(
				response=json.dumps(json_object),
				mimetype='application/json'
			)
	return response


if __name__ == '__main__':
	app.run(debug=True)
