# import the necessary packages
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import playsound
import argparse
import imutils
import dlib
import cv2
import sys
from math import pow,sqrt
from tkinter import *
from time import sleep

#constants    
shape_path = r'shape_predictor_68_face_landmarks.dat'
pilotsound = r'pilot.mp3'
copilotsound = r'copilot.mp3'
earcons = 0.3
eyeframecons = 48

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_path)
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
title1="SLEEP DETECTOR"
l=[]
sb=[]
eb=[]
pilot=""
cpilot=""
mframe="" 

def PStart ():
    global pilot
    if pilot=="":
        p=Detector(0,"PILOT",pilotsound)
        pilot=p
        p.InitiateVideoStream()
        p.InfiniteLoop() 
def CStart ():
    global cpilot
    if cpilot=="":
        c=Detector(1,"CO-PILOT",copilotsound)
        cpilot=c
        c.InitiateVideoStream()
        c.InfiniteLoop()  
def PoThread (): 
    pilo=Thread(target=PStart)
    pilo.deamon=True
    pilo.start()   
def CpoThread (): 
    cpilo=Thread(target=CStart)
    cpilo.deamon=True
    cpilo.start()  
def PEnd ():
    global pilot
    if pilot !="":
        pilot.end = True
        del pilot
        pilot =""    
def CEnd ():     
    global cpilot
    if cpilot !="":
        cpilot.end = True
        del cpilot
        cpilot ="" 
       
def BStart ():
    if pilot=="":
        PoThread()
    if cpilot=="":
        CpoThread () 
    
def BoThread (): 
    both=Thread(target=BStart)
    both.deamon=True
    both.start()    

def BEnd ():     
    if pilot !="":
        PEnd()
    if cpilot !="":
        CEnd () 

def LocExit():
    mframe.destroy()
    BEnd ()
    

lnames={"PILOT CAMERA":[0,0,0,1,0,2,PoThread,PEnd],"COPILOT CAMERA":[1,0,1,1,1,2,CpoThread,CEnd],"BOTH CAMERA'S":[2,0,2,1,2,2,BoThread,BEnd]}

def SleepGui():
    global l,mframe,sb,eb
    mframe=Tk()
    mframe.title(title1)
    cou=0
    for v in lnames:
        pos=lnames[v]
        l.append(Label(mframe,anchor="center", bd=5,text=v))
        l[cou].grid(row=pos[0],column=pos[1],sticky="sw")
        sb.append(Button(mframe,anchor="center", text='START', command=pos[6],width=7,bg="green"))
        eb.append(Button(mframe,anchor="center", text='STOP', command=pos[7],width=6,bg="red"))
        sb[cou].grid(row=pos[2],column=pos[3],sticky="s")
        eb[cou].grid(row=pos[4],column=pos[5],sticky="s")
        cou=cou+1;     
    l.append(Label(mframe,anchor="center", bd=5,text=""))
    l[cou].grid(row=3,column=1,sticky="sw")      
    sb.append(Button(mframe,anchor="center", text='EXIT', command=LocExit,width=6,bg="red"))
    sb[cou].grid(row=4,column=1,sticky="s")
    mframe.mainloop()


class Detector:
    def __init__(self,videostreamport,framename,soundpath):
        self.counter = 0
        self.sound= False
        self.vstream=videostreamport
        self.fname=framename
        self.vs=0
        self.spath=soundpath
        self.end=False
        
    def DistanceBwTwoPoints(self,a,b):
        #formula square root of (x1-x2)2+(y1-y2)2
        return sqrt(pow(a[0]-b[0],2)+pow(a[1]-b[1],2))
       
    def Ear(self,eyedata):
        #formula EAR = (p2 + p3)/(2p1)
        p1 = self.DistanceBwTwoPoints(eyedata[0],eyedata[3])
        p2 = self.DistanceBwTwoPoints(eyedata[1],eyedata[5])
        p3 = self.DistanceBwTwoPoints(eyedata[2],eyedata[4])
        return (p2+p3)/(2.0*p1)
    
    def InitiateVideoStream(self):
        self.vs = VideoStream(src=self.vstream).start()
        sleep(0.2)
        
    def PlaySoundFuc(self,path):
        playsound.playsound(path)    
    def InfiniteLoop (self):
        while True:
            frame = self.vs.read()
            frame = imutils.resize(frame, width=600)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rects = detector(gray, 0)
            for rect in rects:
                shape = predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)
                le = shape[lStart:lEnd]
                re = shape[rStart:rEnd]
                lear = self.Ear(le)
                rear = self.Ear(re)
                ear = (lear + rear) / 2.0
        
                if ear < earcons:
                    self.counter += 1
                   
                    if self.counter >= eyeframecons:
                        if not self.sound:
                            self.sound = True
                            t = Thread(target=self.PlaySoundFuc,args=(self.spath,))
                            t.deamon = True
                            t.start()
                            # draw an alarm 
                        cv2.putText(frame, self.fname+" WAKEUP!", (10, 30),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
                else:
                    self.counter = 0
                    self.sound = False
            
            # show the frame
            cv2.imshow(self.fname, frame)
            cv2.waitKey(1)

            # Break the loop
            if self.end:
                break
            #fun()
            
    def __del__ (self):
         cv2.destroyAllWindows()
         self.vs.stop()
         del self.counter
         del self.sound
         del self.vstream
         del self.fname
         del self.vs
         del self.spath

            
if __name__ == '__main__':
    SleepGui()

  

