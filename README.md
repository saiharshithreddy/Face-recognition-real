# Face-recognition-real
Face recognition with eye and mouth detection

# Installation
1. Imutils: ```python3 pip install --upgrade imutils```
2. OpenCV: ```pip install opencv ```
3. dlib:
On Ubuntu  
```sudo apt-get install build-essential cmake  ```  
```sudo apt-get install libgtk-3-dev ```  
```sudo apt-get install libboost-all-dev```


### Tasks
- [x] Face registration (collecting images)
- [ ] Face recognition with blink and mouth detection

Dataset: Labelled faces in the wild [Download](http://vis-www.cs.umass.edu/lfw/#download)

### Face registration
For using a custom dataset, run ```python3 face_registration.py``` to collect images of a person in folder ```dataset/[person_name]```


### References
1. [Real time eye detection](http://vision.fe.uni-lj.si/cvww2016/proceedings/papers/05.pdf)
