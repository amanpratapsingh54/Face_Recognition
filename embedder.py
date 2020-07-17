import cv2
from imutils import paths
import os
import numpy as np
import pickle


modelpath = './face_detection/res10_300x300_ssd_iter_140000.caffemodel'
prototxtpath = './face_detection/deploy.prototxt.txt'
embedderpath = './embedder/openface_nn4.small2.v1.t7'


detector = cv2.dnn.readNetFromCaffe(prototxtpath, modelpath)


print("> Loading face recognizer...")
embedder = cv2.dnn.readNetFromTorch(embedderpath)

print("> Quantifying faces...")
imagepaths = list(paths.list_images('./dataset'))

knownembeddings = []
knownnames = []

total = 0


for (i,imagepath) in enumerate(imagepaths):
    print("> Processing image {}/{}".format(i + 1, len(imagepaths)))
    
    name = imagepath.split(os.path.sep)[-2]

    
    img = cv2.imread(imagepath)
    r = 600/img.shape[1]
    dim = (600,int(r*img.shape[0]))
    img = cv2.resize(img,dim,interpolation = cv2.INTER_AREA)
    (h,w) = img.shape[:2]

    
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300,300)), 1.0, (300,300), (104.0, 177.0, 123.0), swapRB=False, crop=False)

    
    detector.setInput(blob)
    detections = detector.forward()
    if len(detections) > 0:
        i = np.argmax(detections[0, 0, :, 2])
        confidence = detections[0, 0, i, 2]

        
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7]*np.array([w,h,w,h])
            (x,y,ex,ey) = box.astype("int")

            
            face = img[y:ey, x:ex]
            (fh,fw) = face.shape[:2]
            if fh<20 and fw<20:
                continue

            
            faceblob = cv2.dnn.blobFromImage(face, 1.0/255, (96, 96), (0,0,0), swapRB=True, crop=False)
            embedder.setInput(faceblob)
            embeddings = embedder.forward()

            
            knownnames.append(name)
            knownembeddings.append(embeddings.flatten())
            total = total+1


print("> Serializing {} encodings...".format(total))
data = {"embeddings": knownembeddings, "names": knownnames}


with open('./outputs/embeddings/embeddings.pickle', 'wb') as f:
    f.write(pickle.dumps(data))


