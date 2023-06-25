# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 17:34:37 2022

@author: aoosh
"""

import numpy as np
import cv2

def detect(frame):
    
    gray = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY)
    
   
    img_edges = cv2.Canny(gray , 50,190,3)
    
    
    
    _ , img_thresh = cv2.threshold(img_edges,
                                     254,255,
                                     cv2.THRESH_BINARY)
        
        
    contours , _ = cv2.findContours(img_thresh,
                                    cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    
    min_radius_thresh = 3
    max_radius_thresh = 30
    
    centers = []
    radiuses = []
    for contour in contours:
        
        (x,y),radius = cv2.minEnclosingCircle(contour)
        radius = int(radius)
        
        if radius > min_radius_thresh and radius < max_radius_thresh:
            centers.append(np.array([x,y],dtype=np.int16))
            radiuses.append(radius)
    
    
    return centers , radiuses