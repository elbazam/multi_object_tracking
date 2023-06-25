
import numpy as np
import cv2
from yoloDetector import yoloDetector
from Kalman_Filter.BBoxKalmanFilter import BBoxKalmanFilterDictionary
from sklearn.neighbors import NearestNeighbors

from functions import measurementsHandler , trackerMaker , RemoveDoubles


import argparse


class YoloTracker():
    
    def __init__(self , name = 'video/4.mp4' , saveSolution = False):
        if saveSolution: saveName = 'saved' + name
        if name == '0': name = 0
        
        self.video = cv2.VideoCapture(name)
        self._initVideoParameters()
        self._initDetector()
        self._initTracker()
        self.cov = 25*np.eye(4)
        if saveSolution:
            self.result = cv2.VideoWriter(saveName, 
                                      cv2.VideoWriter_fourcc(*'MJPG'),
                                      int(self.video.get(cv2.CAP_PROP_FPS)),
                                      [self.width,self.height])
        self.neigh = NearestNeighbors(n_neighbors=1)
        
        self.removeList = []
        self.thresh = 100
        
        while self.video.isOpened():
            ret , self.frame = self.video.read()
            if not ret: break
        
            self._update()
            
            
            cv2.imshow('image' , self.frame)
            if saveSolution: self.result.write(self.frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                
                break
            
        
        cv2.destroyAllWindows()
        if saveSolution: self.result.release()
    
    def _initVideoParameters(self):
        self.dt = 1 / self.video.get(cv2.CAP_PROP_FPS)
        _ , self.frame = self.video.read()
        self.height , self.width , _ = self.frame.shape
        
    def _initDetector(self):
        self.yolo = yoloDetector(height=self.height , width=self.width)
    
    def _initTracker(self):
        self.trackers = BBoxKalmanFilterDictionary()
        
    def _getMeasurements(self):
        self.zList = self.yolo.getMeasurements(self.frame)
    
    def _updateTrackersInformation(self):
        self.kfNumber = len(self.trackers.MultiKF.keys())
        self.zNumber = self.zList.shape[0]
        
        

        if self.kfNumber < self.zNumber:
            self.trackers = trackerMaker(self.trackers,self.zList,self.cov,self.dt)
        elif self.kfNumber > self.zNumber and self.zNumber > 0:
            self.trackers = RemoveDoubles(self.trackers)
    
    def _fitNearestNeighbors(self):
        if self.zList.shape[0] >= 2:
            self.neigh.fit(self.zList)
    
    def _updateTrackedObjectInformation(self):
        for key,kf in self.trackers.MultiKF.items():
            self.trackers.MultiKF[key].Predict()
            remove = self.trackers.MultiKF[key].removeMe()
            if remove:
                self.removeList.append(key)
            kfPose = np.expand_dims(self.trackers.MultiKF[key].getNextPosition(),axis=0)
            
            
            if self.zList.shape[0] >= 2:
                d , relativeIndex = self.neigh.kneighbors(kfPose)
                
                if d[0][0] < self.thresh:
                    measurement = measurementsHandler(d[0] ,np.array(self.zList)[relativeIndex[0]] )
                    
                    self.trackers.MultiKF[key].Update(measurement)
                else:
                    remove = self.trackers.MultiKF[key].removeMe()
                    if remove:
                        self.removeList.append(key)
            elif self.zList.shape[0] == 1:
                measurement = self.zList[0]
                
                self.trackers.MultiKF[key].Update(measurement)
            
            pose = self.trackers.MultiKF[key].getNextPosition()
            vel = self.trackers.MultiKF[key].getVelocity() * self.dt
            center = self.trackers.MultiKF[key].getCenter()
            color = kf.color

            if self.trackers.MultiKF[key].showDetection():
                
                cv2.rectangle(self.frame , (pose[0],pose[1]),
                                    (pose[2],pose[3]),
                                        color,2)
                cv2.putText(self.frame, "id: " + str(key),(center + [0,-10]), cv2.FONT_HERSHEY_PLAIN,
                1, color, 2)
                cv2.arrowedLine(self.frame, center, center+vel.astype(np.int32),color,4)
    
    def _removeVanishedTrackers(self):
        for key in self.removeList:
            self.trackers.RemovePredictor(key)
        self.removeList = []
    
    def _update(self):
        self._getMeasurements()
        self._updateTrackersInformation()
        self._fitNearestNeighbors()
        self._updateTrackedObjectInformation()
        self._removeVanishedTrackers()
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--VideoName" ,type=str, default='video/1.mp4' , help='Name of the video')
    parser.add_argument("--Save" , type = bool , default=False , help='Name of the video')
    args = parser.parse_args()
    name = args.VideoName
    save = args.Save

    YoloTracker(name = name , saveSolution = save)
