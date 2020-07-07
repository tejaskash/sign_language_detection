from cv2 import cv2
import time
import numpy as np 
import os

cam = cv2.VideoCapture(0)
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

def draw_capture_area(frame):
    cv2.putText(frame,"Place Hand In The Green Box",(320,60),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255))
    cv2.putText(frame,"Press S to start Capture",(20,320),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255))
    cv2.putText(frame,str(COUNTER),(50,350),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,0,0))
    if CAPTURE==255:
        cv2.putText(frame,"Done",(50,380),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,0,0))    
    cv2.rectangle(frame,(280,60),(600,400),(0,255,0),1)
symbol = input("Enter Symbol: ")
if not os.path.isdir(f"dataset/{symbol}"):
    os.mkdir(f"dataset/{symbol}")
os.system(f"dataset/{symbol}/*")
CAPTURE=0
NUM_IMAGES = 2048
COUNTER = 0
while True:
    if cv2.waitKey(1) == ord('q'):
        cv2.destroyAllWindows()
        cam.release()
        break
    ret,frame = cam.read()
    frame = cv2.flip(frame,1)
    draw_capture_area(frame)
    cv2.imshow("Frame",frame)
    cropped = frame[60:400,280:600]
    cropped = mystery_function(cropped)
    cv2.imshow("Capture Area",cropped)
    if CAPTURE:
        if COUNTER<NUM_IMAGES:
            print("Counter: ",COUNTER)
            cv2.imwrite(f"dataset/{symbol}/"+str(COUNTER)+".jpg",cropped)
            COUNTER+=1
        else:
            CAPTURE=0
            cv2.putText(frame,"Done",(20,350),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,0,0))
        
    if cv2.waitKey(1) == ord('s') and CAPTURE == 0:
        print("Pressed S")
        CAPTURE=1
    if cv2.waitKey(1) == ord('s') and CAPTURE == 1:
        print("Pressed S")
        CAPTURE=0
    if cv2.waitKey(1) == ord('q'):
        cv2.destroyAllWindows()
        cam.release()
        break