#!/usr/bin/python
# -*- coding: utf-8 -*-

import cv2
import auxiliar as aux
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2040)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    
# Returns an image containing the borders of the image
# sigma is how far from the median we are setting the thresholds
def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    # return the edged image
    return edged

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if ret == False:
        print("Codigo de retorno FALSO - problema para capturar o frame")

    # Our operations on the frame come here
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    cor_ciano = "#056bab"
    cor1_ciano, cor2_ciano = aux.ranges(cor_ciano)
    img_rgb_ciano = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mascara_ciano = cv2.inRange(img_hsv, cor1_ciano, cor2_ciano)

    
    cor_magenta= "#90183c"
    cor1_magenta, cor2_magenta = aux.ranges(cor_magenta)
    img_rgb_magenta = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_hsv_magenta = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mascara_magenta = cv2.inRange(img_hsv_magenta, cor1_magenta, cor2_magenta)    


    mascara = mascara_ciano+mascara_magenta
    # Display the resulting frame
    # cv2.imshow('frame',frame)
    cv2.imshow('mascara', mascara)
    
    #Blur para fechar pontos falhos
    mascara_blur = cv2.blur(mascara, (3,3))
    mask= mascara_blur
    #cv2.imshow('mascara', mask)
    
    bordas = auto_canny(mask)
    circles=cv2.HoughCircles(image=bordas,method=cv2.HOUGH_GRADIENT,dp=1.5,minDist=40,param1=200,param2=100,minRadius=5,maxRadius=200)
    bordas_rgb = cv2.cvtColor(bordas, cv2.COLOR_GRAY2RGB)
    output = bordas_rgb

    if circles is not None:        
        circles = np.uint16(np.around(circles))
        for i in circles[0,:]:
            # draw the outer circle
            cv2.circle(output,(i[0],i[1]),i[2],(0,255,0),2)
            # draw the center of the circle
            cv2.circle(output,(i[0],i[1]),2,(0,0,255),3)
            
    #cv2.imshow('mascara', output)
    
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

