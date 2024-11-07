

from ultralytics import YOLO
import numpy as np

class keyPointDetector(object):
    def __init__(self, checkpoint ):
        
        # Load a model
        self.model = YOLO(checkpoint)   # load a custom model
        print('keypoint detection model was loaded')
        
        
    def detect(self, imgPath):

        # collect the results
        results = self.model(imgPath, imgsz=640)[0]

        # collect the detected keypoints
        keypoints = results.keypoints.numpy()
        keypoints_in_pixel = keypoints.xy
        
        py_array = np.array(keypoints_in_pixel)

        return py_array


            