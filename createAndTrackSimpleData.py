
import numpy as np
import cv2
from detector.circleDetector import detect
from ConstantValues import *
from Obstacle import ObstacleList
from Kalman_Filter.SimpleKalmanFilter import KalmanFilterDictionary
from sklearn.neighbors import NearestNeighbors

from functions import measurementsHandler , trackerMaker , RemoveDoubles
import argparse



def main(balls):

    IMG_SIZE = (RECTANGLULAR,RECTANGLULAR,3)
    thresh = 30

    cov = 16*np.eye(2)
    img = np.ones(IMG_SIZE,dtype = np.uint8)
    img.fill(255)

    trackers = KalmanFilterDictionary()

    circles = ObstacleList()
    for index in range(balls):
        circles.AddObstacle(index)

    frame = np.copy(img)
    for key , obsticle in circles.Obstacles.items():
        color = [int(obsticle.color[0]),int(obsticle.color[1]),int(obsticle.color[2])]
        cv2.circle(frame,obsticle.center,15 ,color,-1)
    zList , _ = detect(frame)

    neigh = NearestNeighbors(n_neighbors=1)

    removeList = []

    while circles.Obstacles:
        
        
        frame = np.copy(img)
        
        for key , obsticle in circles.Obstacles.items():
            
            cv2.circle(frame,obsticle.center,15 ,obsticle.color,-1)
            
        zList , _ = detect(frame)
        
        kfNumber = len(trackers.MultiKF.keys())
        zNumber = len(zList)
        
        if kfNumber < zNumber:
            trackerMaker(trackers,np.array(zList),cov)
        elif zNumber > kfNumber:
            RemoveDoubles(trackers)

        if len(zList) >= 2:
            neigh.fit(zList)
        
        for key,kf in trackers.MultiKF.items():
            trackers.MultiKF[key].Predict()
            kfPose = [trackers.MultiKF[key].getNextPosition()]
            if len(zList) >= 2:
                d , relativeIndex = neigh.kneighbors(kfPose)
            
                if d[0][0] < thresh:
                    measurement = measurementsHandler(d[0] ,np.array(zList)[relativeIndex[0]] )
                    
                    trackers.MultiKF[key].Update(measurement)
                else:
                    remove = trackers.MultiKF[key].removeMe()
                    if remove:
                        removeList.append(key)
            elif len(zList) == 1:
                measurement = zList[0]
                
                trackers.MultiKF[key].Update(measurement)
            pose = trackers.MultiKF[key].getNextPosition()
            vel = trackers.MultiKF[key].get_velocity()
            color = kf.color
            if trackers.MultiKF[key].showDetection():
                cv2.rectangle(frame , (pose[0]-15,pose[1]-15),
                                    (pose[0]+15,pose[1]+15),
                                    color,2)
                cv2.arrowedLine(frame, pose, pose+vel,color,4)
                
        
        for key in removeList:
            trackers.RemovePredictor(key)
        removeList = []
        
        circles.UpdateCenters()
        circles.RemoveInvlidObstacles()
        cv2.imshow('image' , frame)
        key = cv2.waitKey(100) & 0xFF
        if key == ord('q'):
            break
        

    cv2.destroyAllWindows()



if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument("--ball" , type = int , default = 5 , help = 'True for saving solution\nFalse for not')
    args = parser.parse_args()
    
    balls = args.ball


    main( balls)




