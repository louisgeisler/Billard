import numpy as np
import cv2
import json
from math import *
from time import *
import time
import matplotlib.pyplot as plt
import pymunk
#import pymunk.pygame_util
import numpy.polynomial.polynomial as poly
from random import random
import os.path

############### Setting Importation ###############

with open('camera.json', 'r') as f:
    data = json.load(f)

for k,v in data.items():
    globals()[k]=v

print("Camera Data:", data)

with open('data.json', 'r') as f:
    data = json.load(f)

for k,v in data.items():
    globals()[k]=np.array(v)

print("Calibraton Data:", data)

############### Configuration Fenetre ###############
cv2.namedWindow("Billard", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Billard", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN) #, cv2.WINDOW_NORMAL, cv2.WINDOW_NORMAL) #

if os.path.isfile('debug.mp4'):
    cap = cv2.VideoCapture('debug.mp4')
else:
    cap = cv2.VideoCapture(camera_number)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 60)
    fps = int(cap.get(5))
    print("fps:", fps)
#cap = cv2.VideoCapture('output.avi')

############### Billard ###############
"""
cap = cv2.VideoCapture(camera_number, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv2.CAP_PROP_FPS, 25)
fps = int(cap.get(5))
print("fps:", fps)


ret, frame = cap.read()

matrix=m_projector2camera.dot(m_camera2screen)

while True:
    ret, frame = cap.read()
    frame = cv2.warpPerspective(frame, matrix, (1920,1080), flags=cv2.INTER_LINEAR)
    cv2.imshow('Billard', frame)
    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break

        
cv2.destroyAllWindows()
cap.release()"""

ret, frame = cap.read()
l, h ,c = frame.shape
cadre = (1920, 1080)
print(l,h)



"""screen = pygame.display.set_mode(cadre)
#screen = pygame.display.set_mode(cadre, pygame.FULLSCREEN)
draw_options = pymunk.pygame_util.DrawOptions(screen)
"""


##### Moteur Physique #####
space = pymunk.Space()
space.gravity = (0, 0)

# Edge of billard:
body = space.static_body
body.position = (0, 0)
lEdges = [pymunk.Segment(body, (0, 0), (cadre[0], 0), 10),
          pymunk.Segment(body, (cadre[0], 0), (cadre[0], cadre[1]), 1),
          pymunk.Segment(body, (cadre[0], cadre[1]), (0, cadre[1]), 1),
          pymunk.Segment(body, (0, cadre[1]), (0, 0), 1)
]
for edge in lEdges:
    edge.elasticity = 0.95
    edge.friction = 0 ## 0.1

space.add(*lEdges)



class Ball:

    scoreLimit = 0.90
    pointLimit = 15 # points
    lBall=[]
    id_count=0
    radius = 27
    mass = 111
    deceleration = 26
    memory = 10 # Le nombre de point utiliser pour les calcules
    max_ball = 20
    sId_available = set(range(max_ball))

    def __init__(self, t, x, y):


        global space, t_physic_engine
        Ball.lBall+=[self]
        self.lPos = [[t, x, y]]
        self.lPos_prediction = [[t_physic_engine, x, y]]
        self.lVitesse = [[t, 0, 0, 0]]  # [[t, dx, dy, v]]
        self.lVitesse_prediction = [[t_physic_engine, 0, 0, 0]]
        self.polynome_distance = poly.Polynomial([0])

        self.debut_simulation = 0

        self.lChemin = []
        self.lVt1 = []
        self.lVt2 = []

        self.lTest = []

        self.lAngle = []
        
        self.lAcc = [[t, 0]]

        self.distance=0

        self.std = []
        
        self.id = min(Ball.sId_available)
        Ball.sId_available.remove(self.id)
        
        self.explosion_frame = 0
        self.debut_simulation = time.time()

        self.movement = 4
        self.prediction_run = False

        
        inertia = pymunk.moment_for_circle(Ball.mass, 0, Ball.radius, (0, 0))
        body = pymunk.Body(Ball.mass, inertia)
        body.position = (x, y)

        shape = pymunk.Circle(body, Ball.radius, (0, 0))
        shape.elasticity = 1 #0.95
        shape.friction = 0.1 #0.9
        space.add(body, shape)
                 

        body.velocity_func = self.static_friction

        self.shape = shape
        self.body = body
        self.creation_time = time.time()



    def static_friction(self, body, gravity, damping, dt):
    
        pymunk.Body.update_velocity(body, gravity, damping, dt)
        
        if self.prediction_run:
            if round(body.velocity.length,3) < 0.01:
                body.velocity = body.velocity*0
            else:
                if len(ball.lVitesse)>1:
                    v=float(self.lVitesse[-1][3])
                else:
                    v=0.0

                body.velocity = body.velocity/body.velocity.length * v * 2 * float(np.exp(-self.debut_simulation/4))
                #body.velocity = body.velocity/body.velocity.length*max(ball.lAcc[-1][1]*dt, v)**1.1
        else:
            #body.velocity = body.velocity - body.velocity/body.velocity.length * 30 #* float(np.exp(-self.debut_simulation/4))
            pass
        
        self.debut_simulation += dt


    def __del__(self):

        global space
        Ball.sId_available.add(self.id)
        space.remove(self.body, self.shape)
        Ball.lBall.remove(self)
        del self


    def interpolation(self, dt):
        tv, dx, dy, v = self.lVitesse[-1]
        t, x, y = self.lPos[-1]
        dp = self.polynome_distance(tv) - self.polynome_distance(tv+dt)
        px = x + dx*v*dt
        py = y + dy*v*dt
        px = min(max(px, 1), 1919)
        py = min(max(py, 1), 1079)
        #x += dx*dp
        #y += dy*dp
        
        if v<15:
            v=0
            dx=0
            dy=0
        
        vx = dx*v
        vy = dy*v
        return px, py, vx, vy, v


    def add_pos(self, t, x, y):
        
        global space, t_physic_engine
        # Le déplacement est-il suffisant ?
        if len(self.lPos)>=1:
            _, x1, y1 = self.lPos[-1]
            d = ((x1 - x)**2 + (y1 - y)**2)**0.5
            if d<10:
                _, x, y = self.lPos[-1]



        # Limite le nombre de donné pour notre prédiction
        aAverage = np.array(self.lPos[-Ball.memory+1:] + [[t, x, y]])
        
        self.lPos += [list(np.average(aAverage,
                                      weights=np.exp((aAverage[:,0]-t)/0.1),
                                      axis=0))] # [[t, x, y], ...]
        
        if self.lPos_prediction[-1][0]!=t_physic_engine:
            self.lPos_prediction+=[[t_physic_engine] + list(self.body.position)]
            self.lVitesse_prediction+=[[t_physic_engine] + list(self.body.velocity) + [self.body.velocity.length]]


        if len(self.lPos) < ball.memory:
            return

        a = np.array(self.lPos[-2:])
        dt, dx, dy = a[1] - a[0]
        vx, vy = dx/dt, dy/dt
        self.lTest += [[t, (vx**2 + vy**2)**0.5]]
        
        lFitting_pos = self.lPos[-Ball.memory:]
        aFitting_pos = np.array(lFitting_pos)

        [[dx],[dy],[cx],[cy]] = cv2.fitLine(np.array([[[x,y]] for t,x,y in lFitting_pos]), cv2.DIST_L2, 0, 0.01, 0.01)

        # Déduit le sens:
        p0 = np.array(lFitting_pos[-2][1:3])
        p1 = np.array([cx+10*dx, cy+10*dy])
        p2 = np.array([cx-10*dx, cy-10*dy])

        d1 = np.linalg.norm(p0-p1)
        d2 = np.linalg.norm(p0-p2)

        if d2<d1:
            dx *= -1
            dy *= -1

        # Projection sur l'axe de la direction:
        aFitting_pos = np.array(lFitting_pos)
        lT = aFitting_pos[:,0]
        lXY = aFitting_pos[:,1:3]
        rotation_matrix = np.array([[dx , -dy], [dy, dx]]).T
        lDistance = rotation_matrix.dot(lXY.T)[0,:]

        t_mean = (lT[-1] + lT[0])/2

        # Interpolation polynomiale de l'avancement:
        coefs = poly.polyfit(lT, lDistance, 2)
        
        pd = poly.Polynomial(coefs)
        pv = pd.deriv()
        pa = pv.deriv()

        self.polynome_distance = pd
        v = pv(t_mean)
        
        acc = pa(t_mean)
        
        # Limiter les vitesses trop petite:
        if v<15:
            vx, vy, v = 0, 0, 0
        
        # Limiter les vitesses trop petite:
        if abs(acc)<1:
            acc = 0

        self.lAngle += [[t, np.arctan2(dx, dy)]]
        
        aAverage = np.array(self.lVitesse[-Ball.memory+1:] + [[t_mean, dx, dy, v]])
        
        self.lVitesse += [list(np.average(aAverage,
                                          weights=np.exp((aAverage[:,0]-t)/0.03),
                                          axis=0))] # [[t, vx, vy, v], ...]
        
        
        aAverage = np.array(self.lAcc[-Ball.memory+1:] + [[t_mean, acc]])
        
        self.lAcc += [list(np.average(aAverage,
                                      weights=np.exp((aAverage[:,0]-t)/1000),
                                      axis=0))] # [[t, acc], ...]
        

        _, px, py = self.lPos[-(Ball.memory+1)//2]
        self.lChemin += [[px,py]]
        pv = self.lVitesse[-1][3]
        sV=11
        self.lVt1 += [[px-dy*pv/sV, py+dx*pv/sV]]
        self.lVt2 += [[px+dy*pv/sV, py-dx*pv/sV]]
        
        if v>20:
            if not self.prediction_run:
                print("Launch",t,self.prediction_run,"id", self.id)
                dt = (time.time() - t_prediction)*7
                px = min(max(x + dx*v*dt, 1), 1919)
                py = min(max(y + dy*v*dt, 1), 1079)
                self.body.position = pymunk.Vec2d(px, py)
                self.body.velocity = pymunk.Vec2d(dx, dy) * 1000
                scale = 100
                #self.body.apply_impulse_at_local_point((dx*scale, dy*scale))
                self.prediction_run = True
                self.debut_simulation = 0
        
        if v==0:
            self.body.position = pymunk.Vec2d(x, y)
            self.body.velocity = pymunk.Vec2d(0, 0)
            self.prediction_run = False

            self.lPos_prediction = [[t_physic_engine, x, y]]

            self.lChemin = []
            self.lVt1 = []
            self.lVt2 = []
    

    @classmethod
    def mapping_detecting_balls(cls, t, lDetected_ball):

        if not lDetected_ball:
            return

        for ball in Ball.lBall:
            if t - ball.lPos[-1][0] > 1.5:
                ball.__del__()

        lDistance = []
        lMapped_ball = set()
        lMapped_detected_ball = set()
       
        for ball in Ball.lBall:
            _, x1, y1 = ball.lPos[-1]
            for (x2, y2) in lDetected_ball:
                #assert not ((x2-x1)**2 + (y2-y1)**2)**0.5 in dDistance, "BUG !!!!!!!"
                lDistance += [[((x2-x1)**2 + (y2-y1)**2)**0.5, (x2, y2), ball]]
        
        lDistance = sorted(lDistance, key = lambda x: x[0])
        
        i = 0
        for d, detected, ball in lDistance:

            if i >= len(Ball.lBall):
                break
            
            if (not detected in lMapped_detected_ball) and (not ball in lMapped_ball):
                
                lMapped_detected_ball.add(detected)
                lMapped_ball.add(ball)
                lDetected_ball.remove(detected)
                ball.add_pos(t, *detected)
                i += 1
                
        for detected in lDetected_ball:
            Ball(t, *detected)
        

        

"""
Vitesse maximale boule: 12,5 m/s
FPS: 30 image/s
Largeur billard: 3,6 m
Largeur image: 1920 px

RayonMinimal = VitesseMaximaleBoule / FPS / LargeurBillard * LargeurImage
 = 12,5 / 30 / 3,6 * 1920
 = 
"""

_, frame = cap.read()

prevframe = frame[:,:,:] #frame[:,:,2]    #First frame
prevframe = cv2.warpPerspective(prevframe, m_camera2screen, (1920,1080), flags=cv2.INTER_LINEAR)
cv2.imshow('Billard', prevframe)

background = cv2.imread("background.jpg")[:,:,2]
background = cv2.warpPerspective(background, m_camera2screen, (1920,1080), flags=cv2.INTER_LINEAR)

fond = cv2.imread("FondDVIC.png")

debut_time = time.time()
n_frame = 0
first = True
t_prediction = time.time()
t_physic_engine = 0

#for i in range(250):
#    ret, frame = cap.read()

while True:

    t_frame = time.time()
    ret, frame = cap.read()
    nextframe = frame[:,:,2].copy()
    #frame = cv2.warpPerspective(frame, m_camera2screen, (1920,1080), flags=cv2.INTER_LINEAR)
    
    nextframe = cv2.warpPerspective(nextframe, m_camera2screen, (1920,1080), flags=cv2.INTER_LINEAR)
    nextframe = cv2.absdiff(background, nextframe)

    nextframe = cv2.GaussianBlur(nextframe,(5,5),0)
    
    _, nextframe = cv2.threshold(nextframe, 100, 255, cv2.THRESH_BINARY)
    
    contours, hierarchy = cv2.findContours(nextframe, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    newframe = fond.copy() # np.zeros(frame.shape)

    l=[]
    
    for c in contours:
        
        M = cv2.moments(c)

        # Surface trop petite ?
        if M["m00"]<np.pi*25**2:
            continue

        lX=[x for [[x, _]] in c]
        lY=[y for [[_, y]] in c]
        
        if np.corrcoef(lX, lY)[0, 1]**2 > 0.75:
            [vx,vy,x,y] = cv2.fitLine(c, cv2.DIST_L2, 0, 0.01, 0.01)
            cv2.line(
                newframe,
                tuple(map(int, (x+vx*-1920, y+vy*-1920))),
                tuple(map(int, (x+vx*1920, y+vy*1920))),
                (255, 255, 255),
                15
            )
            #p = np.polyfit(lX, lY, 1)
            #cv2.line(nextframe, (0, int(np.polyval(p, 0))), (1920, int(np.polyval(p, 1920))), 150, 20)
            continue
        
        x = int(M["m10"] / M["m00"])
        y = int(M["m01"] / M["m00"])
        ecartype = np.std([((x-ix)**2 + (y-iy)**2)**0.5 for ix, iy in zip(lX, lY)])

        # ça ressemble à un cercle ?
        if ecartype < 10:
            #Ball.add_ball(t_frame, x, y)
            #cv2.circle(newframe, (x, y), 50, (255, 255, 255), 10)
            l+=[(x, y)]


    Ball.mapping_detecting_balls(t_frame - debut_time, l)


    for ball in Ball.lBall:

        x, y, vx, vy, v = ball.interpolation((time.time() - t_prediction) + 0.4)
        x, y = int(x), int(y)
        zoom = 0.7
        
        """x_predicted, y_predicted = x + vx*dt*latence, y + vy*dt*latence
        if v > 0:
            print(x,y, x_predicted, y_predicted, v )
        x, y = x_predicted, y_predicted """

        """cv2.line(newframe,
                 (int(x), int(y)),
                 (int(x+vx*2000), int(y+vy*2000)),
                 200,
                 5)"""

        cv2.circle(newframe, (x, y), 50, (255, 255, 255), 10)

        if v > 0:
            cv2.arrowedLine(newframe,
                            (int(x), int(y)),
                            (int(x+vx*zoom), int(y+vy*zoom)),
                            (255, 255, 255),
                            10)

            if len(ball.lChemin)>=2:
                pts = np.array(ball.lChemin + ball.lVt1[::-1],
                          np.int32).reshape((-1, 1, 2))
                #cv2.fillPoly(newframe, [pts], (110, 0, 0))
                chemin = np.array(ball.lChemin + [[x,y]], np.int32).reshape((-1, 1, 2))
                vdx, vdy = np.array([x,y]) - np.array(ball.lChemin[-1])
                vitesse = np.array(ball.lVt1 +
                                    [[ball.lVt1[-1][0]+vdx, ball.lVt1[-1][1]+vdy]] +
                                    [[ball.lVt2[-1][0]+vdx, ball.lVt2[-1][1]+vdy]] +
                                     ball.lVt2[::-1], np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(newframe, [vitesse], (255, 0, 0))
                cv2.polylines(newframe, [vitesse], True, (0, 0, 255), 2)
                
                #acc = np.array(ball.lAcceleration_tangencielle, np.int32).reshape((-1, 1, 2))
                #cv2.polylines(newframe, [vitesse], False, (0, 110, 0), 5)
                #cv2.polylines(newframe, [acc], False, (0, 0, 110), 5)
                cv2.polylines(newframe, [chemin], False, (255, 255, 255), 2)
                vu = np.array(ball.lVt1[0]) - np.array(ball.lChemin[0])
                vu /= np.linalg.norm(vu)
                s=55
                cv2.arrowedLine(newframe,
                                tuple(map(int, ball.lChemin[0])),
                                tuple(map(int, np.array(ball.lChemin[0]) + vu*s)),
                                (255, 255, 255),
                                2)
                cv2.arrowedLine(newframe,
                                tuple(map(int, ball.lChemin[0])),
                                tuple(map(int, np.array(ball.lChemin[0]) - vu*s)),
                                (255, 255, 255),
                                2)

                cv2.putText(newframe,
                            "Speed(px/s)",
                            tuple(map(int, ball.lChemin[1])),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (255, 255, 255),
                            2,
                            cv2.LINE_AA)

                prediction = np.array([[[x,y]] for t,x,y in ball.lPos_prediction], np.int32)
                cv2.polylines(newframe, [prediction], False, (0, 255, 0), 4)

        cv2.circle(newframe,
                   tuple(map(int, ball.body.position)),
                   int(Ball.radius),
                   (0, 255, 0),
                   -1)
        
        font_scale = cv2.getFontScaleFromHeight(cv2.FONT_HERSHEY_SIMPLEX, 30, 3) 
        cv2.putText(newframe,
                    "ID:" + chr(ord('A') + ball.id),
                    (int(x + Ball.radius*2.2), y + 33),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA)
        cv2.putText(newframe,
                    "V=" + str(round(v)),
                    (int(x + Ball.radius*2.2), y-3),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA)
            

        """if explosion:
            f = explosion_gif[ball.explosion_frame]
            ball.explosion_frame+=1

            n_frame, l, h, c = explosion_gif.shape
            nextframe"""

    

    
    cv2.putText(newframe,
                "FPS: " + str(int(1/(time.time() - t_frame))),
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (255, 255, 255),
                3,
                cv2.LINE_AA)

    """cv2.putText(newframe,
                "Frame: " + str(n_frame),
                (50, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (255, 255, 255),
                3,
                cv2.LINE_AA)"""
    n_frame+=1
    """dt = (time.time() - t_prediction)
    for i in range(10):
        space.step(dt/10)"""


    step = 1/120
    n = int((time.time() - debut_time - t_physic_engine) / step)
    for i in range(n):
        space.step(step)

    t_physic_engine += step*n
        
    
    #print(space.current_time_step)
    t_prediction = time.time()

    newframe = cv2.warpPerspective(newframe, m_projector2camera, (1920,1080), flags=cv2.INTER_LINEAR)
        
    cv2.imshow('Billard', newframe)
    
    #k = cv2.waitKey(int((time.time() - t_frame)*1000)) & 0xff
    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break

    
    idBall = k - ord('a')
    if idBall in [b.id for b in Ball.lBall]:
        
        ball = Ball.lBall[idBall]

        fig, axs = plt.subplots(2, 2)
        lVt = [t for t,vx,vy,v in ball.lVitesse]
        lVv = [v for t,vx,vy,v in ball.lVitesse]
        axs[0, 0].plot(lVt, lVv, 'o-')
        axs[0, 0].set(xlabel="Time (s)",
                      ylabel="Speed (px/s)",
                      title="Speed of the ball")
        axs[0, 0].legend(loc='best')
        #axs[0, 0].set_title("Vitesse de la balle")
        
        a = np.array(ball.lAcc)
        axs[0, 1].plot(a[:,0], a[:,1], 'o-')
        axs[0, 1].set(xlabel="Time (s)",
                      ylabel="Acceleration (px/s²)",
                      title="Acceleration of the ball")
        #plt.xlabel("Temp")
        #plt.ylabel("Accélération")
        #plt.figure()
        
        #plt.draw()
        #plt.pause(0.001)
        
        lT = [t for t,x,y in ball.lPos]
        lX = [x for t,x,y in ball.lPos]
        lY = [y for t,x,y in ball.lPos]
        lT_prediction = [t for t,x,y in ball.lPos_prediction]
        lX_prediction = [x for t,x,y in ball.lPos_prediction]
        lY_prediction = [y for t,x,y in ball.lPos_prediction]
        axs[1, 0].plot(lX, lY, 'o-', label='Ball')
        axs[1, 0].plot(lX_prediction, lY_prediction, 'x-', label='Prediction')
        axs[1, 0].set(xlabel="X",
                      ylabel="Y",
                      title="Position of the ball")
        axs[1, 0].legend(loc='best')

        lT = [t for t, a in ball.lAngle]
        lA = [a for t, a in ball.lAngle]
        axs[1, 1].plot(lT, lA, 'o-')
        axs[1, 1].set(xlabel="Time(s)",
                      ylabel="Direction of the ball (radians)",
                      title="Direction of the ball")
        
        """for t, x, y in zip(lT, lX, lY):
            plt.annotate(int(t), (x, y))
        #plt.plot(lX_prediction, lY_prediction, 'o-')
        for t, x, y in zip(lT_prediction, lX_prediction, lY_prediction):
            plt.annotate(int(t), (x, y))"""
        #plt.xlabel("X")
        #plt.ylabel("Y")
        
        #plt.draw()
        #plt.pause(0.001)
        
        #plt.ion()
        plt.show()
        #plt.pause(0.001)
        
        #plt.ion()
        #plt.show(block = False)
        """
        fig = plt.figure()
        fig_ax = fig.add_subplot(111)
        fig_ax.plot(lVtp, lVvp, 'o-')
        fig_ax.set_title("test")
        fig_ax.grid(True)
        fig.show()"""
        
    """
    screen.fill((0,0,0))
    space.debug_draw(draw_options)
    pygame.display.update()"""

cv2.destroyAllWindows()
cap.release()
exit()
