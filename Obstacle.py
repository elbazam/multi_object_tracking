
import numpy as np
from ConstantValues import *


class Obstacle():
    
    def __init__(self , key =0  , center = np.zeros(2) , velocity = np.zeros(2) , randomObstacle = True):
        
        print("Obstacle ", key ," has been created!")
        self.center = center
        self.velocity = velocity
        self.key = key
        R = np.random.randint(0,255)
        G = np.random.randint(0,255)
        B = np.random.randint(0,255)
        self.color = [R,G,B]
        
        self.boundary = RECTANGLULAR
        if randomObstacle: self.ObstacleInit()
    
    def ObstacleInit(self):
        self.CreateRandomCenter()
        self.CreateRandomVelocity()
    
    def CreateRandomCenter(self):
        
        self.center = np.random.randint(AWAY_FROM_BOUNDARY,
                                        self.boundary - AWAY_FROM_BOUNDARY ,
                                        size = 2)
    
    def CreateRandomVelocity(self):
        self.velocity = np.random.randint(-UPPER_VELOCITY,
                                        UPPER_VELOCITY ,
                                        size = 2)
        
        
    def _InImage(self):
        
        condition_x = self.center[0] > -1 and self.center[0] < self.boundary
        if not condition_x: return False
        condition_y = self.center[1] > -1 and self.center[1] < self.boundary
        
        return condition_x and condition_y
        
    def UpdateCenter(self):
        
        self.center += self.velocity
    
    def GetCenter(self):
        return self.center
    
    def Getkey(self):
        return self.key


class ObstacleList():
    
    def __init__(self):
        
        self.Obstacles = {}
    
    def AddObstacle(self ,key = 0,
                    center = np.zeros(2),
                    velocity = np.zeros(2),
                    randomObstacle = True):
        NewObstacle = Obstacle(key = key,center = center,
                                       velocity = velocity,
                                       randomObstacle=randomObstacle)
        self.Obstacles[key] = NewObstacle
    
    
    def UpdateCenters(self):
        for key in self.Obstacles.keys():
            self.Obstacles[key].UpdateCenter()
    
    def RemoveInvlidObstacles(self):
        keys_list = []
        for key , value in self.Obstacles.items():
            if not value._InImage(): keys_list.append(key)
        for key in keys_list:
            self.RemoveObstacle(key)
    
    def RemoveObstacle(self , key):
        print("deleting obstacle number " , key)
        self.Obstacles.pop(key)
    
    def PrintCenters(self):
        for key in self.Obstacles.keys():
            print("Obstacle number " , key, " in location",self.Obstacles[key].GetCenter())
        


    
    







