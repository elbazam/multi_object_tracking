


# Multiple Object Tracking Algorithm Using Kalman Filter.


A Python package for multi-object detection.


Press to watch the video:

[<img src="http://i3.ytimg.com/vi/4F954rXDHiA/hqdefault.jpg" width="50%">](https://www.youtube.com/watch?v=4F954rXDHiA)



## Dependencies

### Circles tracking

- python 3.x
- NumPy
- sklearn
- opencv

### Object tracking

- python3.x
- NumPy
- sklearn
- OpenCV
- tensorflow2


## Execution


### Circles tracking


Press to watch the video:

[<img src="http://i3.ytimg.com/vi/f1iMJtOTRtE/hqdefault.jpg" width="50%">](https://www.youtube.com/watch?v=f1iMJtOTRtE)




#### Create and track data simultaneously

This option creates random sample data while tracking the data simultaneously.

To run:

```bash
python createAndTrackSimpleData.py --ball {Number of circles to track}
```

- ball: int. Number of balls to track. The default value is set to 5.

#### Create data

This option creates a short video of multiple circles randomly moving.

To run:

```bash
python createSimpleData.py --name {Name of the file} --ball {Number of circles to track}
```

- name: str. The name of the video this code creates. Defalut name is "ex5". The file will be saved
in the video folder in "avi" format.
- ball: int. Number of balls to track. The default value is set to 5.

#### Track data

This option opens a short video of multiple circles randomly moving and tracking them.

To run:

```bash
python simpleMultiTracker.py --name {Name of the file} --save {True or False} 
```

- name: str. The name of the video this code reads. Defalut name is "ex5.avi". 
- save: bool. True for saving the file and False for not. The default value is set to False. 
if set to True, saves the file in the "savedVideo" folder.

### Object tracking

#### Pre-Execution step

- This package uses a yolov3 model. The yolov3 model was generated following this tutorial:

[implementation-of-yolov3-simplified](https://www.analyticsvidhya.com/blog/2021/06/implementation-of-yolov3-simplified/)

- The model's name is "yolo_darknet" and is saved in the "model" file.
- The classes' names are located in the "data" folder.

In the case of using another model, change MODEL_SIZE (image input size) values in "yoloDetector.py" to the relevant ones.

#### Track data

This option opens a short video of multiple moving objects and tracks them.

To run:

```bash
python yoloMultiTracker.py --VideoName {Name of the file} --save {True or False} 
```

- name: str. The name of the video this code reads. Defalut name is "video/1.mp4". 
- save: bool. "True" for saving the file and "False" for not. The default value is set to False. 
if set to True, saves the file in the "savedVideo" folder.






