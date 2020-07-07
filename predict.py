from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import time
import imutils
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from imutils import paths
from cv2 import cv2
labels = {0: '6', 1: 'NONE', 2: 'l', 3: '5'}
def draw_capture_area(frame):
    cv2.putText(frame,"Place Hand In The Green Box",(320,60),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255))
    cv2.rectangle(frame,(320,60),(600,400),(0,255,0),1)
def mystery_function(frame):
    frame = cv2.resize(frame,(256,256))
    converted2 = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    converted = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    v = np.median(converted2)
    sigma = 0.33
    lower_thresh = int(max(0, (1.0 - sigma) * v))
    upper_thresh = int(min(255, (1.0 + sigma) * v))
    lowerBoundary = np.array([0,40,30],dtype="uint8")
    upperBoundary = np.array([43,255,254],dtype="uint8")
    skinMask = cv2.inRange(converted,lowerBoundary,upperBoundary)
    skinMask = cv2.addWeighted(skinMask,0.5,skinMask,0.5,0.0)

    skinMask = cv2.GaussianBlur(converted2,(5,5),0)
    converted2 = cv2.GaussianBlur(converted2,(5,5),0)
    skin = cv2.bitwise_and(converted2,converted2,mask=skinMask)
    img2 = cv2.Canny(skin,lower_thresh,upper_thresh)
    return img2
def textify(frame,text,prev,i):
    if text == prev:
        return prev,i
    prev = text
    i+=1
    return prev,i
cam = cv2.VideoCapture(0)
model = load_model("Model.h5")
i = 0
prev = None
predict = 0
text = ""
prev = ""
preds=""
while True:
    ret,frame = cam.read()
    frame = cv2.flip(frame,1)
    draw_capture_area(frame)
    image = frame[60:400,320:600]
    image = mystery_function(image)
    cv2.imshow("Capture Area",image)
    if predict == 20:
        image = cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
        image = cv2.resize(image,(28,28))
        image = image.astype("float") /255.0
        image = img_to_array(image)
        image = np.expand_dims(image,axis=0)
        preds = str(labels[model.predict_classes(image)[0]])
        print(preds)
        if prev != preds:
            text = text+" "+preds
            prev = preds
        predict = 0
    board = np.zeros((512,512,3), np.uint8)
    cv2.putText(board,preds,(256,256),cv2.FONT_HERSHEY_PLAIN,5,(255,0,255),2)
    cv2.imshow("Frame",frame)
    cv2.imshow("Board",board)
    if cv2.waitKey(1) == ord('q'):
        cv2.destroyAllWindows()
        cam.release()
        break
    predict+=1