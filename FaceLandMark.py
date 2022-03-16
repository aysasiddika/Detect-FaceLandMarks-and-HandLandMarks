from contextlib import redirect_stderr
from sre_constants import SUCCESS
from tkinter import font
from turtle import color
import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)   #"video.mp4")
pTime = 0

mpDraw = mp.solutions.drawing_utils

mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=2)
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=2,color= (0,255,0))

mpHands =mp.solutions.hands
hands = mpHands.Hands()

while True:
    SUCCESS, img = cap.read()
    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    #try:
    #    img = cv2.resize(img,(0,0),None,0.25,0.25)
    #except:
    #    print('Invalid Frame')

    results = faceMesh.process(imgRGB)
    resultH = hands.process(imgRGB)
    if resultH.multi_hand_landmarks:
        for handlms in resultH.multi_hand_landmarks:
            mpDraw.draw_landmarks(img,handlms,mpHands.HAND_CONNECTIONS)
    if results.multi_face_landmarks:
        for facelnms in results.multi_face_landmarks:
            mpDraw.draw_landmarks(img,facelnms,mpFaceMesh.FACEMESH_CONTOURS,drawSpec,drawSpec)
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime= cTime
    cv2.putText(img,f'FPS: {int(fps)}',(20,70),cv2.FONT_HERSHEY_PLAIN,3,(0,255,0),3)
    cv2.imshow("image",img)
    cv2.waitKey(1)