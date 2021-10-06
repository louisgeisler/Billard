import numpy as np
import cv2
import json
from scipy import optimize

############### Trouver Caméra ###############

cv2.namedWindow("Camera", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Camera", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


camera_number = 0

while True:
    
    cap = cv2.VideoCapture(camera_number)
    test, frame = cap.read()

    if not test:
        
        assert camera_number != 0, "No Camera Available !!!"
        camera_number = 0
        cap = cv2.VideoCapture(camera_number)
        test, frame = cap.read()
        
    frame = cv2.resize(frame, (1920, 1080))

    while True:

        ok, frame = cap.read()
        assert ok, "Camera disconnected"
        frame = cv2.resize(frame, (1920, 1080))
        
        cv2.putText(frame,
                "Press enter to valide, or press any other touch to switch of camera :",
                (30,30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1/2,
                (0,0,255),
                1,
                cv2.LINE_AA)

        cv2.imshow('Camera', frame)

        k = cv2.waitKey(1)
        if (k != -1):
            break

    if k==13:
        break

    camera_number += 1

cv2.destroyAllWindows()
cap.release()

data={"camera_number": camera_number}
with open('camera.json', 'w') as f:
    json.dump(data, f)

input("Calibration finished !!")
exit()
