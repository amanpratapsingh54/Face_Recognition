import cv2
import numpy as np
from imutils.video import FPS
import pickle
import time


modelpath = './face_detection/res10_300x300_ssd_iter_140000.caffemodel'
prototxtpath = './face_detection/deploy.prototxt.txt'
embedderpath = './embedder/openface_nn4.small2.v1.t7'


facedetector = cv2.CascadeClassifier('./cascades/haarcascade_frontalface_default.xml')


print("> Loading face detector...")
detector = cv2.dnn.readNetFromCaffe(prototxtpath, modelpath)


print("> Loading face recognizer...")
embedder = cv2.dnn.readNetFromTorch(embedderpath)


recognizer = pickle.loads(open('./outputs/recognizer/recognizer.pickle','rb').read())
le = pickle.loads(open('./outputs/label_encodings/le.pickle','rb').read())


print("> Starting video stream...")
cam = cv2.VideoCapture(0)
time.sleep(1)

fps = FPS().start()
start = time.time()


while(True):
    
    ret, frame = cam.read()
    frame = cv2.flip(frame, +1)
    r = 600/frame.shape[1]
    dim = (600,int(r*frame.shape[0]))
    frame = cv2.resize(frame,dim,interpolation = cv2.INTER_AREA)
    (h,w) = frame.shape[:2]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetector.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(10, 10)
    )

    for (x,y,w,h) in faces:

        face = frame[max(0,int(y-(h/4))):(int(y+(h*1.25))), max(0,int(x-(w/4))):(int(x+(w*1.25)))]

        
        faceblob = cv2.dnn.blobFromImage(face, 1.0/255.0, (96, 96), (0, 0, 0), swapRB=True, crop=False)

        
        embedder.setInput(faceblob)
        embedding = embedder.forward()

        
        preds = recognizer.predict_proba(embedding)[0]
        j = np.argmax(preds)
        proba = preds[j]
        name = le.classes_[j]

        
        text = "{}".format(name)
        y_ = y - 10 if y - 10 > 10 else y + 10
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255), 2)
        cv2.putText(frame, text, (x,y_), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

    
    fps.update()
    cv2.imshow("cam",frame)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

end = time.time()
fps.stop()
print("> Elasped time: {:.2f} sec".format(end-start))
print("> Approx. FPS: {:.2f}".format(fps.fps()))

cam.release()
cv2.destroyAllWindows()

