#!/usr/bin/python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
from matplotlib import pyplot as plt
import time
from math import *

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2040)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

lower = 0
upper = 1

def auto_canny(image, sigma=0.33): # sigma is how far from the median we are setting the thresholds

    v = np.median(image) # compute the median of the single channel pixel intensities
    
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper) # apply automatic Canny edge detection using the computed median
    
    return edged # return the edged image

while(True):
    ret, frame = cap.read(0)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    bordas = auto_canny(blur)

    circles = []
    bordas_color = cv2.cvtColor(bordas, cv2.COLOR_GRAY2BGR)

    circles = None
    circles=cv2.HoughCircles(bordas,cv2.HOUGH_GRADIENT,2,40,param1=50,param2=100,minRadius=5,maxRadius=60)
    if circles is not None:        
        circles = np.uint16(np.around(circles))
        lista_x = []
        lista_y = []
        for i in circles[0,:]:
            #cv2.circle(img, center, radius, color[, thickness[, lineType[, shift]]])
            cv2.circle(bordas_color,(i[0],i[1]),i[2],(0,255,0),2)
            cv2.circle(bordas_color,(i[0],i[1]),2,(0,0,255),3)
            lista_x.append(i[0])          
            lista_y.append(i[1])
            
        for i in range(0, len(lista_x)-1):
            x0 = lista_x[i]
            x = lista_x[i+1]
            y0 = lista_y[i]
            y = lista_y[i+1]
            x_x0= fabs((x-x0)*(x-x0))
            y_y0= fabs((y-y0)*(y-y0))
            h = sqrt(x_x0 + y_y0)
            f = 876.4285714285714
            H = 14
            D = (f/h)*H
            cv2.line(bordas_color,(x0,y0),(x,y),(255,0,0),5)
            termoy = abs(y0 - y)
            termox = abs(x0 - x)
            angulo = atan2(termoy,termox)
            angulo = degrees(angulo)

                
    # cv2.putText(img, text, org, fontFace, fontScale, color[, thickness[, lineType[, bottomLeftOrigin]]])
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(bordas_color,'Press q to quit',(10,50), font, (0.75),(255,255,255),2,cv2.LINE_AA)
    cv2.putText(bordas_color,"Distancia: {}".format(D),(10,100), font, (0.75),(255,255,255),2,cv2.LINE_AA)
    cv2.putText(bordas_color,"Angulo: {}".format(angulo),(10,150), font, (0.75),(255,255,255),2,cv2.LINE_AA)
    cv2.imshow('Detector de circulos',bordas_color)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
