
import numpy as np
import cv2
from detector.circleDetector import detect
from Kalman_Filter.SimpleKalmanFilter import KalmanFilterDictionary
from sklearn.neighbors import NearestNeighbors

from functions import measurementsHandler , trackerMaker , RemoveDoubles


import argparse



def main(name , save):
    
    videoName = 'video/' + name
    thresh = 30
    cov = 16*np.eye(2)
    video = cv2.VideoCapture(videoName)
    if save:
        saveVideoName = 'savedVideo/' + name
        result = cv2.VideoWriter(saveVideoName, 
                            cv2.VideoWriter_fourcc(*'MJPG'),
                            10, [700,700])
    trackers = KalmanFilterDictionary()
    
    neigh = NearestNeighbors(n_neighbors=1)
    
    removeList = []

    while video.isOpened():
        ret , frame = video.read()
        if not ret: break
        
        
        zList , _ = detect(frame)
        
        kfNumber = len(trackers.MultiKF.keys())
        zNumber = len(zList)
        
        if kfNumber < zNumber:
            trackers = trackerMaker(trackers,np.array(zList),cov)
        elif kfNumber > zNumber:
            trackers = RemoveDoubles(trackers) 

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
        
        if save: result.write(frame)
        cv2.imshow('image' , frame)
        key = cv2.waitKey(100) & 0xFF
        if key == ord('q'):
            break
        
    if save: result.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument("--name" ,type = str ,  default='ex5.avi' , help='Name of the video')
    parser.add_argument("--save" , type = bool , default = False , help = 'True for saving solution\nFalse for not')
    args = parser.parse_args()
    name = args.name
    save = args.save


    main(name , save)
