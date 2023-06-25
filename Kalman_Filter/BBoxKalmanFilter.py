# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 19:15:34 2023

@author: aoosh
"""

import numpy as np


class KalmanFilterBBox():
    # x(t) = A(t) * x(t-1) + e(t) || e(t) ~ N(0,Q(t))
    # z(t) = H(t) * x(t) + r(t)   || r(t) ~ N(0,R(t))
    
    # x(t) = [x1 y1 x2 y2 vx vy]
    
    def __init__(self , dt = 0.5):
        R = np.random.randint(0,255)
        G = np.random.randint(0,255)
        B = np.random.randint(0,255)
        self.color = [R,G,B]
        
        self.dt = dt
        self.A = np.array([[1,0,0,0,dt,0],
                           [0,1,0,0,0,dt],
                           [0,0,1,0,dt,0],
                           [0,0,0,1,0,dt],
                           [0,0,0,0,1,0],
                           [0,0,0,0,0,1]
                          ])
        self.H = np.array([[1,0,0,0,0,0],
                           [0,1,0,0,0,0],
                           [0,0,1,0,0,0],
                           [0,0,0,1,0,0]
                          ])
        self.X = np.zeros(self.A.shape[1])
        self.P = np.eye(self.A.shape[1])
        self.Q = np.array([[25 , 0 , 0 , 0 , 0 , 0],
                           [0 , 25 , 0 , 0 , 0 , 0],
                           [0 , 0 , 25 , 0 , 0 , 0],
                           [0 , 0 , 0 , 25 , 0 , 0],
                           [0 , 0 , 0 , 0 , 49 , 0],
                           [0 , 0 , 0 , 0 , 0 , 49]
                          ])
        self.R = dt * np.eye(4)
        
        self.counterSame = 0
        self.counterEscaped = 0

        self.apear = 0

        self.chi = 9.488
    
    def _isValid(self , z):

        if self.apear < 8:
            return True

        '''
            Innovation test.
        '''
        
        epsilon = np.dot(z.T , np.dot(np.linalg.inv(self.S),z))
        return epsilon < self.chi
    
    def showDetection(self):
        if self.apear > 10: self.apear = 10
        return self.apear > 2
        
    def Predict(self , dt = 1):
        
        self.counterEscaped += 1
        self.X = np.dot(self.A , self.X)
        self.P = np.dot(np.dot(self.A , self.P),self.A.T) + self.Q
        
    def Update(self , z):
        
        self.S = np.dot(self.H , np.dot(self.P,self.H.T)) + self.R
        innovation = z - np.dot(self.H , self.X)
        if not self._isValid(innovation): return
        self.counterEscaped = 0
        self.apear += 1

        
        self.Kalman_Gain = np.dot(np.dot(self.P,self.H.T),np.linalg.inv(self.S))
        
        self.X += np.dot(self.Kalman_Gain,innovation)
        
        self.P = np.dot((np.eye(self.H.shape[1])-np.dot(self.Kalman_Gain,self.H)),self.P)
    
    def getNextPosition(self):
        
        state = self.X[:4]
        state = state.astype(np.int32)
        
        return np.array(state)
    
    def getVelocity(self):
        
        state = self.X[4:]
        state = state.astype(np.int32)
        
        return state
    
    def getCenter(self):
        
        state = self.getNextPosition()
        center = np.zeros(2)
        center[0] = 0.5*(state[0] + state[2])
        center[1] = 0.5*(state[1] + state[3])
        
        return center.astype(np.int32)
    
    def removeMe(self):
        self.counterEscaped += 1
        if self.counterEscaped > 40:
            return True
        return False
    
    def removeDouble(self):
        self.counterSame += 1
        if self.counterSame > 3:
            return True
        return False
    
    def __eq__(self , other):
        
        x1 = self.getNextPosition()
        x2 = other.getNextPosition()
        
        if np.linalg.norm(x1-x2) < 10: return True


class BBoxKalmanFilterDictionary():
    
    def __init__(self):
        
        self.MultiKF = {}
    
    def AddPredictor(self , key = 0 , dt = 0.5):
        print(f"following new data with key: {key}")
        self.MultiKF[key] = KalmanFilterBBox(dt)
    
    def RemovePredictor(self , key):
        print(f"deleted existing data with key: {key}")
        try: self.MultiKF.pop(key)
        except: print("deleted this key before")
    