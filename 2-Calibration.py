import numpy as np
import cv2
from itertools import permutations  
import json
import os

margeX=400
margeY=300
l_point_test=[]
for x in [margeX, 1920-margeX]:
    for y in [margeY, 1080-margeY]:
        l_point_test += [[x,y]]

############### Setting Importation ###############

with open('camera.json', 'r') as f:
    data = json.load(f)

for k,v in data.items():
    globals()[k]=v

print("Camera Data:", data)
try:
    with open('data.json', 'r') as f:
        data = json.load(f)

    for k,v in data.items():
        globals()[k]=np.array(v)
    
except:
    l_circle_screen=[[0,0],
                 [0,1080],
                 [1920,0],
                 [1920,1080]]
    l_circle_camera = l_circle_screen
    l_circle_projector = l_point_test
    

print("Calibraton Data:", data)

############### Configuration Windows ###############
cv2.namedWindow("Billard", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Billard", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


############### Configuration Camera ###############


def get_frame():

    global camera_number
    cap = cv2.VideoCapture(camera_number)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    if os.path.isfile('debug.mp4'):
        cap = cv2.VideoCapture('debug.mp4')
    
    ok, frame = cap.read()

    cap.release()
    
    if not ok:
        cv2.destroyAllWindows()
        print("No Camera available")
        input()
        exit()
    
    return frame

############### Events ###############

def draw_circle(event,x,y,flags,param):
    
    global l_circle, background
    
    if event in [cv2.EVENT_RBUTTONDOWN, cv2.EVENT_LBUTTONDOWN]:
        
        if len(l_circle)>=4:
            l_circle[min(enumerate([(xC-x)**2+(yC-y)**2 for xC,yC in l_circle]), key=lambda x: x[1])[0]]=[x,y]
        else:
            l_circle+=[[x,y]]
        
        frame=background.copy()
        
        for i,p1 in enumerate(l_circle):
            p1=tuple(p1)
            for p2 in l_circle[i+1:]:
                p2=tuple(p2)
                cv2.line(frame,p1,p2,(255,0,0),3)
            cv2.circle(frame,p1,20,(0,0,255),-1)
        
        cv2.imshow('Billard',frame)


############### Calibration Camera ###############

background=get_frame()

cv2.putText(background,
            "Click on the four corners of the billard !",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1/2,
            (0,0,255),
            1,
            cv2.LINE_AA)

l_circle=l_circle_camera
cv2.imshow('Billard', background)

cv2.setMouseCallback('Billard', draw_circle)

frame=background.copy()
for i,p1 in enumerate(l_circle):
    p1=tuple(p1)
    for p2 in l_circle[i+1:]:
        p2=tuple(p2)
        cv2.line(frame,p1,p2,(255,0,0),3)
    cv2.circle(frame,p1,20,(0,0,255),-1)

cv2.imshow('Billard',frame)

cv2.waitKey(0)
while len(l_circle)<4:
    cv2.waitKey(0)
    
cv2.setMouseCallback('Billard', lambda *args: None)
    
l_circle_camera=l_circle.copy()

frame_camera=background.copy()


############### Calibration Projecteur ###############


calibration_test=np.zeros((1080,1920,3), np.uint8)

margeX=400
margeY=300
l_point_test=[]
for x in [margeX, 1920-margeX]:
    for y in [margeY, 1080-margeY]:
        l_point_test += [[x,y]]
        #cv2.circle(calibration_test,(x,y),40,(255,255,255),-1)


for i,p1 in enumerate(l_point_test):
    p1=tuple(p1)
    for p2 in l_point_test[i+1:]:
        p2=tuple(p2)
        cv2.line(calibration_test,p1,p2,(255,255,255),3)
    cv2.circle(calibration_test,p1,40,(255,255,255),-1)


cv2.imshow('Billard',calibration_test)
cv2.waitKey(80)

background=get_frame()

cv2.putText(background,
            "Click left on the four circles currently projected on the billard !",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1/2,
            (0,0,255),
            1,
            cv2.LINE_AA)

for x,y in l_circle_camera:
    cv2.circle(background, (x,y), 20, (0,0,255), -1)

l_circle=l_circle_projector
cv2.imshow('Billard',background)

cv2.setMouseCallback('Billard',draw_circle)

frame=background.copy()
for i,p1 in enumerate(l_circle):
    p1=tuple(p1)
    for p2 in l_circle[i+1:]:
        p2=tuple(p2)
        cv2.line(frame,p1,p2,(255,0,0),3)
    cv2.circle(frame,p1,20,(0,0,255),-1)

cv2.imshow('Billard',frame)

cv2.waitKey(0)
while len(l_circle)<4:
    cv2.waitKey(0)

cv2.setMouseCallback('Billard', lambda *args: None)
    
l_circle_projector=l_circle.copy()

frame_projector = background.copy()
for p1 in l_circle_projector:
    cv2.circle(frame_projector, tuple(p1), 40, (0, 0, 255), -1)


############### Screen ###############

l_circle_screen=[[0,0],
                 [0,1080],
                 [1920,0],
                 [1920,1080]]


############### Informations ###############

def ordering(l_point1, l_point2):

    global background
    
    l_distance_ordered=[]
    for l_ordering in permutations(list(range(len(l_point1)))):
        d=0
        l_ordered_point=[]
        for i1,i2 in enumerate(l_ordering):
            d+=np.linalg.norm(np.array(l_point1[i1])
                              - np.array(l_point2[i2]))
            l_ordered_point+=[l_point2[i2]]
        
        l_distance_ordered+=[[d,l_ordered_point]]

    l_ordered_point=min(l_distance_ordered, key=lambda x: x[0])[1]

    frame=background.copy()
    
    for p1,p2 in zip(l_point1, l_ordered_point):
        p1=tuple(p1)
        p2=tuple(p2)
        cv2.line(frame,p1,p2,(255,0,0),3)
        cv2.circle(frame,p1,20,(0,0,255),-1)
        cv2.circle(frame,p2,20,(0,255,0),-1)

    cv2.imshow('Billard',frame)
    cv2.waitKey(2000)
    
    return l_ordered_point


l_circle_camera = ordering(l_circle_projector, l_circle_camera)
l_circle_screen = ordering(l_circle_camera, l_circle_screen)
l_point_test =  ordering(l_circle_camera, l_point_test)


l_circle_projector=np.float32(l_circle_projector)
l_circle_camera=np.float32(l_circle_camera)
l_circle_screen=np.float32(l_circle_screen)
l_circle_test=np.float32(l_point_test)


calibration_test2=np.zeros((1080,1920,3), np.uint8)
for i,p1 in enumerate(l_circle_screen):
    p1=tuple(map(int, p1))
    for p2 in l_circle_screen[i+1:]:
        p2=tuple(map(int, p2))
        cv2.line(calibration_test2,p1,p2,(255,255,255),3)
    cv2.circle(calibration_test2,p1,40,(255,255,255),-1)


cv2.putText(calibration_test2,
            "Image 2D a deformer dans un espace 3D pour fitter le billard, auppuyez sur une touche pour continuer",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1/2,
            (0,0,255),
            1,
            cv2.LINE_AA)
cv2.imshow('Billard', calibration_test2)
cv2.waitKey(0)

tMat1_0 = cv2.getPerspectiveTransform(l_circle_screen, l_circle_test)
tMat1_1 = cv2.getPerspectiveTransform(l_circle_projector, l_circle_test)
tMat1_2 = cv2.getPerspectiveTransform(l_circle_test, l_circle_camera)

tMat1 = tMat1_1.dot(tMat1_2).dot(tMat1_0)

test_1 = cv2.warpPerspective(calibration_test2, tMat1, (1920,1080), flags=cv2.INTER_LINEAR)
cv2.putText(test_1,
            "Image 2D deforme dans un espace 3D, auppuyez sur une touche pour continuer",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1/2,
            (0,0,255),
            1,
            cv2.LINE_AA)
cv2.imshow('Billard',test_1)
cv2.waitKey(0)



tMat2 = cv2.getPerspectiveTransform(l_circle_camera, l_circle_screen)
test_2 = cv2.warpPerspective(frame_camera, tMat2, (1920,1080), flags=cv2.INTER_LINEAR)
cv2.putText(test_2,
            "Image 3D du billard déformer pour passer dans l'espace 2D de l'écran, auppuyez sur une touche pour continuer",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1/2,
            (0,0,255),
            1,
            cv2.LINE_AA)
cv2.imshow('Billard',test_2)
cv2.waitKey(0)

d_information={"m_projector2camera": tMat1,
               "m_camera2screen": tMat2,
               "l_circle_projector": l_circle_projector.astype('int'),
               "l_circle_camera": l_circle_camera.astype('int'),
               "l_circle_screen": l_circle_screen.astype('int'),
        }
"""
with open('data.pkl', 'wb') as f:
    pickle.dump(d_information, f)
"""

d_information={k:v.tolist() for k,v in d_information.items()}

with open('data.json', 'w') as f:
    json.dump(d_information, f)

cv2.destroyAllWindows()
input("Calibration finished !!")
exit()
