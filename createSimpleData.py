
import numpy as np
import cv2
from ConstantValues import *
from Obstacle import ObstacleList
import argparse


IMG_SIZE = (700,700,3)

def main(name , balls):

    videoName = 'video/' + name + '.avi'
    img = np.ones(IMG_SIZE,dtype = np.uint8)
    img.fill(255)

    result = cv2.VideoWriter(videoName, 
                          cv2.VideoWriter_fourcc(*'MJPG'),
                          10, [700,700])
    
    circles = ObstacleList()
    for index in range(balls):
        circles.AddObstacle(index)

    frame = np.copy(img)
    for key , obsticle in circles.Obstacles.items():
        color = [int(obsticle.color[0]),int(obsticle.color[1]),int(obsticle.color[2])]
        cv2.circle(frame,obsticle.center,15 ,color,-1)

    while circles.Obstacles:
        
        
        frame = np.copy(img)
        
        for key , obsticle in circles.Obstacles.items():
            
            cv2.circle(frame,obsticle.center,15 ,obsticle.color,-1)
            
        cv2.imshow('image' , frame)
        key = cv2.waitKey(100) & 0xFF
        circles.UpdateCenters()
        circles.RemoveInvlidObstacles()
        result.write(frame)
        if key == ord('q'):
            break
        
    result.release()

    cv2.destroyAllWindows()

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--name" ,type = str ,  default='ex5' , help='Name of the video')
    parser.add_argument("--ball" , type = int , default = 10 , help = 'True for saving solution\nFalse for not')
    args = parser.parse_args()
    name = args.name
    balls = args.ball


    main(name , balls)