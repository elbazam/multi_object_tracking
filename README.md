


# Multi Tracking Using Kalman Filter.


A python package for multi object detection.



[![Tracking traffic]          // Title
(https://i.ytimg.com/vi/4F954rXDHiA/maxresdefault.jpg)] // Thumbnail
(https://youtu.be/4F954rXDHiA "Tracking traffic")    // Video Link





## Dependencies

### Circles tracking

- python 3.x
- numpy
- sklearn
- opencv

### Object tracking

- python3.x
- numpy
- sklearn
- opencv
- tensorflow2


## Execution


### Circles tracking


[![Tracking circles]          // Title
(https://i.ytimg.com/vi/7TtYKalIaoE/maxresdefault.jpg)] // Thumbnail
(https://youtube.com/shorts/7TtYKalIaoE?feature=share "Tracking circles")    // Video Link



#### Create data and track it online

This option allows creating a random sample data while online tracking the data.

To run:

```bash
python createAndTrackSimpleData.py --ball {Number of circles to track}
```

- ball: int. Number of balls to track. Default value set to 5.

#### Create data

This option creates a short video of multiple circles randomly moving.

To run:

```bash
python createSimpleData.py --name {Name of the file} --ball {Number of circles to track}
```

- name: str. The name of the video this code creates. Defalut name is "ex5". The file will be saved
in the video folder in avi format.
- ball: int. Number of balls to track. Default value set to 5.

#### Track data

This option opens a short video of multiple circles randomly moving and track them.

To run:

```bash
python simpleMultiTracker.py --name {Name of the file} --save {True or False} 
```

- name: str. The name of the video this code reads. Defalut name is "ex5.avi". 
- save: bool. True for saving the file and False for not. Default value set to False. 
if sets to True, saves the file in "savedVideo" folder.

### Object tracking

#### Pre-Execution step

- This package uses a yolov3 model. The yolov3 model was generated following this tutorial:

[implementation-of-yolov3-simplified](https://www.analyticsvidhya.com/blog/2021/06/implementation-of-yolov3-simplified/)

- The model's name is "yolo_darknet" and is saved in "model" file.
- The classes' names are located in the "data" folder.

In case of using other model, change MODEL_SIZE (image input size) values in "yoloDetector.py" to the relevant ones.

#### Track data

This option opens a short video of multiple moving objects and track them.

To run:

```bash
python yoloMultiTracker.py --VideoName {Name of the file} --save {True or False} 
```

- name: str. The name of the video this code reads. Defalut name is "video/1.mp4". 
- save: bool. True for saving the file and False for not. Default value set to False. 
if sets to True, saves the file in "savedVideo" folder.






