import sys
from flask import Flask, render_template, url_for, request, jsonify
from flask_cors import CORS, cross_origin
import numpy as np
import pandas as pd
import pickle
from sklearn.externals import joblib

from imutils.video import VideoStream
import argparse
import imutils
from imutils import face_utils
import time
import cv2
import os
import dlib

app = Flask(__name__)
# CORS(app)
cors = CORS(app, resources={r"/api/*": {"origins": "*"}})
app.config['CORS_HEADERS'] = 'Content-Type'


@app.route('/')
def home():
    return render_template('home.html')

# request object : name


@app.route('/api/predict')
@cross_origin()
def predict():
    shape_predictor = 'shape_predictor_68_face_landmarks.dat'
    cascade = 'haarcascade_frontalface_default.xml'
    # construct the argument parser and parse the arguments
    # ap = argparse.ArgumentParser()
    # ap.add_argument("-n", "--name", required=True,
    # 	help="path to output directory")
    # args = vars(ap.parse_args())

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
    (mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

    directory = args["name"]
    parent_dir = "./dataset/"

    # Path
    path = os.path.join(parent_dir, directory)

    # create a directory of the person
    try:
    	os.mkdir(path)
    except:
    	pass


    # loop over the frames from the video stream
    while True:


    	time_to_recognize += 1

    	frame = vs.read()
    	orig = frame.copy()
    	frame = imutils.resize(frame, width=2000)

    	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    	rects = detector2(gray, 0)

    	for rect in rects:
    		shape = predictor(gray, rect)
    		shape = face_utils.shape_to_np(shape)

    		leftEye = shape[lStart:lEnd]
    		rightEye = shape[rStart:rEnd]

    		mouth = shape[mStart:mEnd]

    		mouthHull = cv2.convexHull(mouth)
    		leftEyeHull = cv2.convexHull(leftEye)
    		rightEyeHull = cv2.convexHull(rightEye)
    		# draw
    		cv2.drawContours(frame, [mouthHull], -1, (211, 211, 211), 1)
    		cv2.drawContours(frame, [leftEyeHull], -1, (211, 211, 211), 1)
    		cv2.drawContours(frame, [rightEyeHull], -1, (211, 211, 211), 1)

    	cv2.putText(frame, "Move your head slowly", (30,60),
    	cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0,0,255),2)

    	# show the output frame
    	cv2.imshow("Frame", frame)
    	key = cv2.waitKey(1) & 0xFF

    	# To collect images
    	if time_to_recognize > 50:
    		break

    	elif time_to_recognize <= 50:
    		p = os.path.sep.join([path, "{}.png".format(
    			str(total).zfill(5))])
    		cv2.imwrite(p, orig)
    		total += 1

    # do a bit of cleanup
    print("{} face images stored".format(total))

    cv2.destroyAllWindows()
    vs.stop()



if __name__ == '__main__':
    app.run(debug=True)
