
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import argparse
import pickle

embeddings = 'output/embeddings.pickle'
recognizer = 'output/recognizer.pickle'
labels = 'output/le.pickle'
# load the face embeddings
print("Loading face embeddings")
data = pickle.loads(open(embeddings, "rb").read())
print("Data: ", list(data.keys()))


print("Encoding labels...")
le = LabelEncoder()
labels = le.fit_transform(data["names"])


print("Training model")
recognizer = SVC(C=1.0, kernel="linear", probability=True)
recognizer.fit(data["embeddings"], labels)


f = open(recognizer, "wb")
f.write(pickle.dumps(recognizer))
f.close()


f = open(labels, "wb")
f.write(pickle.dumps(le))
f.close()
