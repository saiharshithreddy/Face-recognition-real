import sys
from flask import Flask, render_template, url_for, request, jsonify
from flask_cors  import CORS, cross_origin
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pickle
import time
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
from sklearn.svm import SVC


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
	username = jsondata['username']
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
	directory = username
	
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
			
			
			# Draw the green borders on eyes. 
			cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
			cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

		# display
		cv2.putText(frame, "Move your head slowly", (30,60),
		cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0,0,255),2)
		cv2.putText(frame, "Images taken:{}".format(total), (30,300),
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
	
	
	
	generate_embeddings('dataset')
	# load the face embeddings
	print("[INFO] loading face embeddings...")
	data = pickle.loads(open('output/embeddings.pickle', "rb").read())

	# encode the labels
	print("[INFO] encoding labels...")
	le = LabelEncoder()
	labels = le.fit_transform(data["names"])

	# train the model used to accept the 128-d embeddings of the face and
	# then produce the actual face recognition
	print("[INFO] training model...")
	recognizer = SVC(C=1.0, kernel="linear", probability=True)
	recognizer.fit(data["embeddings"], labels)

	# write the actual face recognition model to disk
	f = open('output/recognizer.pickle', "wb")
	f.write(pickle.dumps(recognizer))
	f.close()

	# write the label encoder to disk
	f = open('output/le.pickle', "wb")
	f.write(pickle.dumps(le))
	f.close()
	print(path)
	embeddings = generate_embeddings2(path)

	# Append the feature vector to the json object and return 
	jsondata['feature_vector'] = list(embeddings)
	json_object = {}
	json_object["success"] = True
	json_object["code"] = int(200)
	json_object["content"]= (jsondata)


	response = app.response_class(
        response=json.dumps(json_object),
        mimetype='application/json'
    )
	return response

def generate_embeddings2(path_to_image):
	# Generate the embedding from the images. 

	
	protoPath = os.path.sep.join(['face_detection_model', "deploy.prototxt"])
	modelPath = os.path.sep.join(['face_detection_model',
		"res10_300x300_ssd_iter_140000.caffemodel"])
	detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

	
	embedding_model = 'openface_nn4.small2.v1.t7'
	embedder = cv2.dnn.readNetFromTorch(embedding_model)

	
	print(path_to_image)
	imagePaths = list(paths.list_images(path_to_image))
	
	
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

			if confidence > 0.5:
				
				box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
				(startX, startY, endX, endY) = box.astype("int")

				
				face = image[startY:endY, startX:endX]
				(fH, fW) = face.shape[:2]

				# ensure the face width and height are sufficiently large
				if fW < 20 or fH < 20:
					continue

				faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
					(96, 96), (0, 0, 0), swapRB=True, crop=False)
				embedder.setInput(faceBlob)
				vec = embedder.forward()

				
				if not knownNames:
					knownNames = name
					
				knownEmbeddings += vec.flatten()
				total += 1
	return knownEmbeddings/30

def recognize(image_path):
		
		print("Loading face detector...")
		protoPath = os.path.sep.join(['face_detection_model', "deploy.prototxt"])
		modelPath = os.path.sep.join(['face_detection_model',
			"res10_300x300_ssd_iter_140000.caffemodel"])
		detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

		
		print("Loading face recognizer...")
		embedding_model = 'openface_nn4.small2.v1.t7'
		embedder = cv2.dnn.readNetFromTorch(embedding_model)
		recognizer = 'output/recognizer.pickle'
		
		recognizer = pickle.loads(open(recognizer, "rb").read())
		le = 'output/le.pickle'
		le = pickle.loads(open(le, "rb").read())

		
		image = cv2.imread(image_path)
		image = imutils.resize(image, width=600)
		(h, w) = image.shape[:2]

		
		imageBlob = cv2.dnn.blobFromImage(
			cv2.resize(image, (300, 300)), 1.0, (300, 300),
			(104.0, 177.0, 123.0), swapRB=False, crop=False)


		detector.setInput(imageBlob)
		detections = detector.forward()

		# loop over the detections
		for i in range(0, detections.shape[2]):
			
			
			confidence = detections[0, 0, i, 2]

			
			if confidence > 0.5:
				# compute the (x, y)-coordinates of the bounding box for the
				# face
				box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
				(startX, startY, endX, endY) = box.astype("int")

				# extract the face ROI
				face = image[startY:endY, startX:endX]
				(fH, fW) = face.shape[:2]

				# ensure the face width and height are sufficiently large
				if fW < 20 or fH < 20:
					continue

				faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96),
					(0, 0, 0), swapRB=True, crop=False)
				embedder.setInput(faceBlob)
				vec = embedder.forward()

				# perform classification to recognize the face
				preds = recognizer.predict_proba(vec)[0]
				j = np.argmax(preds)
				proba = preds[j]
				name = le.classes_[j]


		return name, proba

def generate_embeddings(path_to_image):
	
	protoPath = os.path.sep.join(['face_detection_model', "deploy.prototxt"])
	modelPath = os.path.sep.join(['face_detection_model',
		"res10_300x300_ssd_iter_140000.caffemodel"])
	detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

	
	embedding_model = 'openface_nn4.small2.v1.t7'
	embedder = cv2.dnn.readNetFromTorch(embedding_model)

	imagePaths = list(paths.list_images(path_to_image))
	
	
	knownEmbeddings = []
	knownNames = []

	
	total = 0

	# loop over the image paths
	for (i, imagePath) in enumerate(imagePaths):
		# extract the person name from the image path
		print("[INFO] processing image {}/{}".format(i + 1,
			len(imagePaths)))
		name = imagePath.split(os.path.sep)[-2]

		
		image = cv2.imread(imagePath)
		image = imutils.resize(image, width=600)
		(h, w) = image.shape[:2]

		# construct a blob from the image
		imageBlob = cv2.dnn.blobFromImage(
			cv2.resize(image, (300, 300)), 1.0, (300, 300),
			(104.0, 177.0, 123.0), swapRB=False, crop=False)

		
		detector.setInput(imageBlob)
		detections = detector.forward()

		
		if len(detections) > 0:
			
			i = np.argmax(detections[0, 0, :, 2])
			confidence = detections[0, 0, i, 2]

			
			if confidence > 0.5:
				
				box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
				(startX, startY, endX, endY) = box.astype("int")

				# extract the face ROI and grab the ROI dimensions
				face = image[startY:endY, startX:endX]
				(fH, fW) = face.shape[:2]

				# ensure the face width and height are sufficiently large
				if fW < 20 or fH < 20:
					continue

				
				faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
					(96, 96), (0, 0, 0), swapRB=True, crop=False)
				embedder.setInput(faceBlob)
				vec = embedder.forward()

				
				knownNames.append(name)
				knownEmbeddings.append(vec.flatten())
				total += 1

	# dump the facial embeddings + names to disk
	print("[INFO] serializing {} encodings...".format(total))
	data = {"embeddings": knownEmbeddings, "names": knownNames}
	f = open('output/embeddings.pickle', "wb")
	f.write(pickle.dumps(data))
	f.close()


def eye_aspect_ratio(eye):
		
		A = dist.euclidean(eye[1], eye[5])
		B = dist.euclidean(eye[2], eye[4])

		C = dist.euclidean(eye[0], eye[3])

		# compute the eye aspect ratio
		ear = (A + B) / (2.0 * C)
		return ear



#==============================predict==============================
@app.route('/api/predict', methods=['POST'])
@cross_origin()
def predict():
	content = request.data
	jsondata = json.loads(content)
	
	username = jsondata['username']
	
	
	EYE_AR_THRESH = 0.3
	
	EYE_AR_CONSEC_FRAMES = 3

	
	COUNTER = 0
	TOTAL = 0
	shape_predictor = 'shape_predictor_68_face_landmarks.dat'
	print("Loading facial landmark predictor")
	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor(shape_predictor)

	
	(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
	(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
	# (mStart, mEnd) = (49, 68)

	# start the video stream thread
	print("Starting video stream thread")

	vs = VideoStream(src=0).start()
	
	time.sleep(1.0)
	path = './test/'
	similarity_score = 0
	prediction = False
	spoofing = False
	predicted_name = ""
	similarity = 0
	start_time = time.time()
	while True:

		frame = vs.read()
	
		orig = frame.copy()
		frame = imutils.resize(frame, width=1450)
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	
		rects = detector(gray, 0)
		if not rects:
			COUNTER = 0
			TOTAL = 0

	
		# More than two persons are detected
		if len(rects) > 1:
			cv2.destroyAllWindows()
			vs.stop()
			spoofing = True
			break

		for rect in rects:
			
			shape = predictor(gray, rect)
			shape = face_utils.shape_to_np(shape)

			
			leftEye = shape[lStart:lEnd]
			rightEye = shape[rStart:rEnd]
			leftEAR = eye_aspect_ratio(leftEye)
			rightEAR = eye_aspect_ratio(rightEye)
			
			ear = (leftEAR + rightEAR) / 2.0
			
			
			leftEyeHull = cv2.convexHull(leftEye)
			rightEyeHull = cv2.convexHull(rightEye)
			
			cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
			cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
			
			
			if ear < EYE_AR_THRESH:
				COUNTER += 1
			else:
				
				if COUNTER >= EYE_AR_CONSEC_FRAMES:
					TOTAL += 1
				COUNTER = 0
			

			cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 30),
				cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 2)
			
			if TOTAL >= 3:
				
				p = os.path.sep.join([path, "test.png"])
				cv2.imwrite(p, orig)
				predicted_embedding = generate_embeddings2('test')
				trained_embedding = np.array(jsondata['feature_vector'])
				predicted_embedding = predicted_embedding.reshape(1,128)
				predicted_name, similarity = recognize('./test/test.png')
				
				print("Name: {} and similarity: {}".format(predicted_name, similarity))
				similarity_score = cosine_similarity(predicted_embedding,trained_embedding)

				print(similarity_score)
				
				prediction = True
			
		if prediction:
			break
		
		
		cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF

		
		
		end_time = time.time()
		time_elapsed = end_time - start_time
		
		# runs for only 15 secs
		if time_elapsed > 15:
			break
	
	
	json_object = {}
	if spoofing:
		json_object["success"] = False
		json_object["code"] = 400
		json_object["spoofing"] = True
	else:
		cv2.destroyWindow('Frame')
		vs.stop()
		
		print('prediction', predicted_name)
		print('name', username)
		if similarity_score[0][0] > 0.50 and predicted_name == username and similarity > 0.70:
			
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


if __name__ == "__main__":
	
	app.run(debug=True)
