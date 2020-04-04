# Face-recognition-real
Face recognition with eye and mouth detection

# Installation
1. Imutils: ```python3 pip install --upgrade imutils```
2. OpenCV: ```pip install opencv ``` [For MacOS](https://www.pyimagesearch.com/2016/12/19/install-opencv-3-on-macos-with-homebrew-the-easy-way/)
3. dlib:
On Ubuntu  
```sudo apt-get install build-essential cmake  ```  
```sudo apt-get install libgtk-3-dev ```  
```sudo apt-get install libboost-all-dev```


### Tasks
- [x] Face registration (collecting images)
- [x] Face recognition with blink and mouth detection
- [ ] Speed up the video stream

Dataset: Labelled faces in the wild [Download](http://vis-www.cs.umass.edu/lfw/#download)

### Face registration
For using a custom dataset, run ```python3 face_registration.py``` to collect images of a person in folder ```dataset/[person_name]```

**Added flask server** to make an api call   
```python3 app.py```

### Eye and mouth detection
* Using face alignment [2] to get the facial landmarks and using [1] to compute eye aspect ratio to identify eye blinking and opening of mouth.   
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
