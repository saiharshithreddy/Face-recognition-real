# USAGE
# python detect_blinks.py --shape-predictor shape_predictor_68_face_landmarks.dat --video blink_detection_demo.mp4
# python3 detect_blinks.py --shape-predictor shape_predictor_68_face_landmarks.dat

# import the necessary packages
from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import os

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

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
ap.add_argument("-v", "--video", type=str, default="",
	help="path to input video file")
args = vars(ap.parse_args())

# define two constants, one for the eye aspect ratio to indicate
# blink and then a second constant for the number of consecutive
# frames the eye must be below the threshold
EYE_AR_THRESH = 0.3
MOUTH_AR_THRESH = 0.79
EYE_AR_CONSEC_FRAMES = 3

# initialize the frame counters and the total number of blinks
COUNTER = 0
TOTAL = 0
Mouth = 0
# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# grab the indexes of the facial landmarks for the left and
# right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = (49, 68)
# start the video stream thread
print("[INFO] starting video stream thread...")
# vs = FileVideoStream(args["video"]).start()
#fileStream = True
vs = VideoStream(src=0).start()
#vs = VideoStream(usePiCamera=True).start()
fileStream = False
time.sleep(1.0)
path = './test/'
# loop over frames from the video stream
while True:
	# if this is a file video stream, then we need to check if
	# there any more frames left in the buffer to process
	if fileStream and not vs.more():
		break

	# grab the frame from the threaded video file stream, resize
	# it, and convert it to grayscale
	# channels)
	frame = vs.read()
	orig = frame.copy()
	frame = imutils.resize(frame, width=1450)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# detect faces in the grayscale frame
	rects = detector(gray, 0)

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
		mouth = shape[mStart:mEnd]
		mouthMAR = mouth_aspect_ratio(mouth)
		mar = mouthMAR
		# average the eye aspect ratio together for both eyes
		ear = (leftEAR + rightEAR) / 2.0
		mouthHull = cv2.convexHull(mouth)
		# compute the convex hull for the left and right eye, then
		# visualize each of the eyes
		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
		#TOTAL = 0
		#Mouth = 0
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
		if mar > MOUTH_AR_THRESH:
			Mouth += 1
			cv2.putText(frame, "Mouth is Open!", (30,60),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)
		# draw the total number of blinks on the frame along with
		# the computed eye aspect ratio for the frame
		if Mouth == 0:
			cv2.putText(frame, "Open your mouth to detect you", (100, 300),
			cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 2)

		if Mouth == 1:
			cv2.putText(frame, "Open your mouth one more time", (100, 300),
			cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 2)

		cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 2)
		#cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
		#	cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		if TOTAL == 3 and Mouth == 2:
			cv2.putText(frame, "Real person", (50, 90),
				cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 2)
			# time.sleep(2.0)
		if 3 < TOTAL <= 4 and Mouth > 2:
			cv2.putText(frame, "Now look straight at the camera", (100, 300),
		cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 2)
		if TOTAL > 4 and Mouth > 2:
			p = os.path.sep.join([path, "test.png"])
			cv2.imwrite(p, orig)
			exit()
	# show the frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
