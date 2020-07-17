import pickle
import sklearn
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder


print("> Loading face embeddings...")
data = pickle.loads(open('./outputs/embeddings/embeddings.pickle','rb').read())


print("> Encoding labels...")
le = LabelEncoder()
labels = le.fit_transform(data['names'])


print("> Training model...")
recognizer = SVC(C=1.0, kernel='linear', probability=True)
recognizer.fit(data['embeddings'],labels)


with open('./outputs/recognizer/recognizer.pickle', 'wb') as f:
    f.write(pickle.dumps(recognizer))


with open('./outputs/label_encodings/le.pickle', 'wb') as f:
    f.write(pickle.dumps(le))
