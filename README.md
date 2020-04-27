# Face-recognition-real
Face recognition with eye detection

# Installation

1. Setup a virtual environment in the folder
```sudo pip3 install virtualenv ```  
```virtualenv myenv```  
```source myenv/bin/activate```

2. Install requirements
```pip3 install -r requirements.txt```  
  ├── sklearn  
  ├── flask  
  ├── flask_cors  
  ├── imutils  
  ├── dlib  
  ├── opencv-python  


1. Imutils: ```python3 pip install --upgrade imutils```
2. OpenCV: ```pip install opencv ``` [For MacOS](https://www.pyimagesearch.com/2016/12/19/install-opencv-3-on-macos-with-homebrew-the-easy-way/)
3. dlib:
On Ubuntu  
```sudo apt-get install build-essential cmake  ```  
```sudo apt-get install libgtk-3-dev ```  
```sudo apt-get install libboost-all-dev```

#### Folder structure
├── dataset  
│   └── harshith  
├── detect_blinks.py  
├── extract_embeddings.py  
├── face_detection_model  
│   ├── deploy.prototxt  
│   └── res10_300x300_ssd_iter_140000.caffemodel  
├── facerecog_v2.py  
├── face_registration.py  
├── flask_backend.py  
├── haarcascade_frontalface_default.xml  
├── openface_nn4.small2.v1.t7  
├── output  
│   ├── embeddings.pickle  
│   ├── le.pickle  
│   └── recognizer.pickle  
├── __pycache__  
│   └── flask.cpython-37.pyc  
├── README.md  
├── recognize.py  
├── recognize_video.py  
├── requirements.txt  
├── shape_predictor_68_face_landmarks.dat  
├── test  
│   └── test.png  
└── train_model.py  

### Tasks
- [x] Face registration (collecting images)
- [x] Face recognition with blink detection
- [ ] Speed up the video stream

Dataset: Labelled faces in the wild [Download](http://vis-www.cs.umass.edu/lfw/#download)

### Face registration
1. For using a custom dataset, run ```python3 face_registration.py --name [person_name]``` to collect images of a person in folder ```dataset/[person_name]```  
2. Extract embeddings from an image ```python3 extract_embeddings.py```. The embeddings are stored in the path ```output/embeddings.pickle```


**Added flask server** to make an api call   
```python3 facerecog_v2.py```

### Eye and mouth detection
* Using face alignment [2] to get the facial landmarks and using [1] to compute eye aspect ratio to identify eye blinking.   
* This is to avoid spoofing attacks by clearly differentiating a real person's face and an image of the person

### Face recognition
* Generate 128-d embedding of each person's image in the dataset.
* Train the SVM model for classification
* Testing: Generate the embedding of the person you want to recognize and compute the similarity between test person's embedding and pre-computed embedding.


### References
1. [Real time eye detection](http://vision.fe.uni-lj.si/cvww2016/proceedings/papers/05.pdf)
2. [Face alignment](http://www.csc.kth.se/~vahidk/papers/KazemiCVPR14.pdf)
3. [Shape predictor dataset](https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/)
4. [FaceNet](https://www.cv-foundation.org/openaccess/content_cvpr_2015/app/1A_089.pdf)
5. [dlib pretrained models](https://github.com/davisking/dlib-models)
