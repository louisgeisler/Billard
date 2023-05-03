#captures and verifies the camera input/output

#import modules
import numpy as np
import cv2
import json

#Finding the camera
cv2.namedWindow("Camera", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Camera", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

#webcam camera id
camera_number = 0

#capturing the camera video input
while True:
    
    #cap replaced with "capture", more readable
    capture = cv2.VideoCapture(camera_number)
    test, frame = capture.read()

    #error checking for missing camera source
    if not test:
        
        assert camera_number != 0, "No Camera Available!!"
        camera_number = 0
        capture = cv2.VideoCapture(camera_number)
        test, frame = capture.read()
        
    frame = cv2.resize(frame, (1920, 1080))

    while True:

        ok, frame = capture.read()
        assert ok, "Camera disconnected"
        frame = cv2.resize(frame, (1920, 1080))
        
        cv2.putText(frame,
                "Press Enter to validate, or press any other key to switch cameras:",
                (30,30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1/2,
                (0,0,255),
                1,
                cv2.LINE_AA)

        cv2.imshow('Camera', frame)

        #if any other button is pressed
        k = cv2.waitKey(1)
        if (k != -1):
            break

    #if 'enter' is pressed
    if k==13:
        break

    camera_number += 1

cv2.destroyAllWindows()
capture.release()

#convert incoming python data to json in 'camera.json'
data={"camera_number": camera_number}
with open('camera.json', 'w') as f: #'w' implies write
    json.dump(data, f)

#alert user
input("Calibration finished!!")
exit()
