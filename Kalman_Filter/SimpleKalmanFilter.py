

import numpy as np


class KalmanFilter():
    # x(t) = A(t) * x(t-1) + e(t) || e(t) ~ N(0,Q(t))
    # z(t) = H(t) * x(t) + r(t)   || r(t) ~ N(0,R(t))
    
    def __init__(self , dt = 0.5):
        R = np.random.randint(0,255)
        G = np.random.randint(0,255)
        B = np.random.randint(0,255)
        self.color = [R,G,B]
        
        self.dt = dt
        self.A = np.array([[1,0,1,0],
                           [0,1,0,1],
                           [0,0,1,0],
                           [0,0,0,1]
                          ])
        self.H = np.array([[1,0,0,0],
                           [0,1,0,0]
                          ])
        self.X = np.zeros(self.A.shape[1])
        self.P = 16*np.eye(self.A.shape[1])
        self.Q = np.array([[1 , 0 , 0 , 0],
                           [0 , 1 , 0 , 0],
                           [0 , 0 , 4 , 0],
                           [0 , 0 , 0 , 4]
                          ])
        self.R = np.diag([16,16])
        
        self.counter = 0
        
        self.apear = 0
        
        
        
    def Predict(self):
        
        self.X = np.dot(self.A , self.X)
        self.P = np.dot(np.dot(self.A , self.P),self.A.T) + self.Q
        self.counter +=1


    def _isValid(self , z):

        if self.apear < 8:
            return True

        '''
            Innovation test.
        '''
        chi = 5.991
        epsilon = np.dot(z.T , np.dot(np.linalg.inv(self.S),z))
        return epsilon < chi
    
    def showDetection(self):
        if self.apear > 10: self.apear = 10
        return self.apear > 2

    def Update(self , z):
        

        innovation = z - np.dot(self.H , self.X)
        if not self._isValid(innovation): return
        
        self.apear += 1
        self.counter -= 1
        self.S = np.dot(self.H , np.dot(self.P,self.H.T)) + self.R
        self.Kalman_Gain = np.dot(np.dot(self.P,self.H.T),np.linalg.inv(self.S))
        
        self.X += np.dot(self.Kalman_Gain,innovation)
        
        self.P = np.dot((np.eye(self.H.shape[1])-np.dot(self.Kalman_Gain,self.H)),self.P)
    
    def getNextPosition(self):
        
        state = self.X[:2]
        state = state.astype(np.int32)
        
        return state
    
    def get_velocity(self):
        
        state = self.X[2:]
        state = state.astype(np.int32)
        
        return state
    
    def removeMe(self):
        self.counter += 1
        if self.counter > 10:
            return True
        return False
    
    def removeDouble(self):
        self.counter += 1
        if self.counter > 5:
            return True
        return False
    
    def __eq__(self , other):
        
        x1 = self.getNextPosition()
        x2 = other.getNextPosition()
        
        if np.linalg.norm(x1-x2) < 10: return True


class KalmanFilterDictionary():
    
    def __init__(self):
        
        self.MultiKF = {}
    
    def AddPredictor(self , key = 0 , dt = 0.5):
        print(f"following new data with key: {key}")
        self.MultiKF[key] = KalmanFilter(dt)
    
    def RemovePredictor(self , key):
        print(f"deleted existing data with key: {key}")
        self.MultiKF.pop(key)
    
