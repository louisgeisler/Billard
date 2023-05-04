#captures and verifies the camera input/output
#ie. lets you choose the source

#import modules
import numpy as np
import cv2
import json

#naming the camera display popup window
cv2.namedWindow("Camera", cv2.WND_PROP_FULLSCREEN)

#set properties of named window, "Camera"
cv2.setWindowProperty("Camera", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

#webcam camera id
camera_number = 0

#while loop runs until user presses Any Key / Enter
while True:
    
    #capture the incoming video from "camera_number" (0)
    capture = cv2.VideoCapture(camera_number) 

    #create capture "frame" and "test"
    test, frame = capture.read()

    #error checking for missing camera source
    if not test:
        
        #returns False if camera_number is not = 0
        assert camera_number != 0, "No Camera Available!!" 

        #force camera_number = 0, create capture "frame" and "test" again
        camera_number = 0
        capture = cv2.VideoCapture(camera_number)
        test, frame = capture.read()
        
    #make the frame capture fit 1920x1080
    frame = cv2.resize(frame, (1920, 1080))

    while True:

        #create capture "ok" and "frame" 
        ok, frame = capture.read()

        #disconnect if "ok" returns False
        assert ok, "Camera disconnected"
        frame = cv2.resize(frame, (1920, 1080))
        
        #user instructions
        cv2.putText(frame,
                "Press Enter to validate, or press any other key to switch cameras:",
                (30,30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1/2,
                (0,0,255),
                1,
                cv2.LINE_AA)

        #show "Camera" window with "frame" capture
        cv2.imshow('Camera', frame)

        #if any other button is pressed
        k = cv2.waitKey(1)
        if (k != -1):
            break

    #if 'enter' is pressed
    if k==13:
        break

    camera_number += 1

#close the window(s)
cv2.destroyAllWindows()
#free memory
capture.release()

#write incoming data from 'camera_number 0' to json in 'camera.json'
data={"camera_number": camera_number}
with open('camera.json', 'w') as f: 
    json.dump(data, f)

#alert user
input("Calibration finished!!")
exit()
