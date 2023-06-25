
import tensorflow as tf
from utils import load_class_names, output_boxes, draw_outputs, resize_image
import numpy as np
import cv2

MODEL_SIZE = (416, 416,3)
NUM_OF_CLASSES = 80
CLASS_NAME = './data/coco.names'
MAX_OUTPUT_SIZE = 40
MAX_OUTPUT_SIZE_PER_CLASS= 20
IOU_THRESHOLD = 0.5
CONFIDENCE_THRESHOLD = 0.5

class yoloDetector():
    
    def __init__(self , height = 0 , width = 0):
        
        self.model = tf.keras.models.load_model('model/yolo_darknet',compile=False)
        self.class_names = load_class_names(CLASS_NAME)
        self.width = width
        self.height = height
        
    def getMeasurements(self , frame):
        
        self._detect(frame)
        self.MeasurementsNumber = self.nums[0].numpy()
        # [x0 , y0 , x1 , y1]
        self.numpyBox = self.boxes[0][:self.MeasurementsNumber].numpy()
        
        self.numpyBox[:,0] = self.numpyBox[:,0] * self.width
        self.numpyBox[:,2] = self.numpyBox[:,2] * self.width
        self.numpyBox[:,1] = self.numpyBox[:,1] * self.height
        self.numpyBox[:,3] = self.numpyBox[:,3] * self.height
        
        return self.numpyBox
        
        
    def insertDetectionInImg(self):
        
        
        img = draw_outputs(self.frame,
                           self.boxes,
                           self.scores,
                           self.classes,
                           self.nums,
                           self.class_names)
        return img
    
    def _detect(self , frame):
        self.frame = frame
        resized_frame = tf.expand_dims(frame, 0)
        resized_frame = resize_image(resized_frame,
                                     (MODEL_SIZE[0],
                                      MODEL_SIZE[1]))
         
        self.pred = self.model.predict(resized_frame , verbose = 0)
        self.boxes, self.scores, self.classes, self.nums = output_boxes( \
            self.pred, MODEL_SIZE,
            max_output_size=MAX_OUTPUT_SIZE,
            max_output_size_per_class=MAX_OUTPUT_SIZE_PER_CLASS,
            iou_threshold=IOU_THRESHOLD,
            confidence_threshold=CONFIDENCE_THRESHOLD)
    
    def drawDetection(self):
        num = self.numpyBox.shape[0]
        for i in range(num):
            left_coordinates = ((self.numpyBox[i,0:2]).astype(np.int32))
            right_coordinates = ((self.numpyBox[i,2:4]).astype(np.int32))
            self.frame = cv2.rectangle(self.frame, (left_coordinates), (right_coordinates), (255,0,0), 2)
        return self.frame
            
        
    
    

 