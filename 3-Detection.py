#captures the billiard table background
#remove everything from the table

#import modules
import numpy as np
import cv2
from itertools import permutations  
import json
import os


#importation settings
with open('camera.json', 'r') as f: #'r' implies read
    data = json.load(f) #store .json data in variable 'data'

for k,v in data.items():
    globals()[k]=v

print("Camera Data:", data)

#screen properties
cv2.namedWindow("Billard", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Billard", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN) 

#cap replaced with 'capture', more readable
capture = cv2.VideoCapture(camera_number)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

if os.path.isfile('debug.mp4'):
    capture = cv2.VideoCapture('debug.mp4')


while True:
    
    ok, frame = capture.read()
    frame2 = frame.copy()
    cv2.putText(frame,
            "Clear the billard table then press any key!",
            (30,30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1/2,
            (0,0,255),
            1,
            cv2.LINE_AA)

    if (cv2.waitKey(30)!=-1):
        break

    cv2.imshow('Billard', frame)

#store background (camera frame pointing at table) in 'background.jpg'
frame2 = cv2.resize(frame2,(1920,1080)) 
cv2.imwrite('background.jpg', frame2)

cv2.destroyAllWindows()
capture.release()

#alert user
input("Detection finished")
exit()

