# -*- coding: utf-8 -*-
"""
Created on Sun Jun 18 15:57:30 2023

@author: aoosh
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors


def measurementsHandler(distances , z):
    
    '''
    inputs:
        d -> list of distances: float.
        z -> list of observations: float.
    return:
        z' -> weighted observation.
    '''
    if len(z) == 1:
        return z[0]
    
    npDistances = np.array(distances)
    npDistances = npDistances * npDistances
    weights = 1 -  npDistances / np.sum(npDistances)
    
    return np.average(z , weights=weights , axis = 0)
    
    
def trackerMaker(trackers,zList,cov , dt = 0.5):
    
    RestOfZ = np.copy(zList)
    check = NearestNeighbors(n_neighbors=1)
    check.fit(zList)
    AssociatedZ = []
    delist = []
    
    for kf in trackers.MultiKF.values():
        kfPose = [kf.getNextPosition()]
        _ , idx = check.kneighbors(kfPose)
        
        if idx[0][0] not in delist:
            AssociatedZ.append(zList[idx[0][0]])
            delist.append(idx[0][0])
    AssociatedZ = np.array(AssociatedZ)
    RestOfZ = np.delete(RestOfZ , delist , 0)
    if list(trackers.MultiKF.keys()):
        lastKey = list(trackers.MultiKF.keys())[-1] + 1
        while lastKey in list(trackers.MultiKF.keys()):
            lastKey += 1
    else:
        lastKey = 0
    RestOfZSize = AssociatedZ.shape[0]
    if RestOfZSize > 0 and not lastKey == 0:
        check.fit(AssociatedZ)
    UpdateTracker = True
    n = zList.shape[1]
    for z in RestOfZ:
        if not lastKey == 0 and RestOfZSize > 0:
            UpdateTracker = False
            _ , idx = check.kneighbors([z])
            v = z - AssociatedZ[idx[0][0]]
            epsilon = np.dot(v.T,np.dot(np.linalg.inv(cov) , v))
            print(epsilon)
            if n == 2 and not epsilon < 5.991: UpdateTracker = True
            elif n == 4 and  not epsilon < 9.448: UpdateTracker = True
        
        if UpdateTracker:
            trackers.AddPredictor(lastKey , dt = dt)
            trackers.MultiKF[lastKey].X[:n] = z
            
            lastKey += 1
        UpdateTracker = False
    
    return trackers


def RemoveDoubles(trackers , threshold = 10):
    
    keys = list(trackers.MultiKF.keys())
    KeysVisited = []
    KeysRemove = []
    
    for key in keys:
        for inKey in keys:
            if key == inKey or inKey in KeysVisited:
                continue
            if trackers.MultiKF[key] == trackers.MultiKF[inKey]:
                if(trackers.MultiKF[inKey].removeDouble()):
                    KeysRemove.append(inKey)
        KeysVisited.append(key)
    for inKey in KeysRemove:
        print("Removing doubles")
        trackers.RemovePredictor(inKey)
    return trackers
    
                
            
        
    
    



    