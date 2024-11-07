# -*- coding: utf-8 -*-
"""
Created on Fri May 17 11:07:32 2024

@author: obus
"""

from ultralytics import YOLO
import cv2

class dolphinBodyDetector(object):
    def __init__(self, fastsam_checkpoint ):
        
        # Load a model
        self.model = YOLO(fastsam_checkpoint)   # load a custom model
        print('model was loaded')
        
        
    def detect(self, imgPath, trainedImgSize):
        # results = model( os.path.join( imagePath, filenames[0]))
        img = cv2.imread(imgPath)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img_rgb, ( trainedImgSize[0], trainedImgSize[1]))
        results = self.model.predict(img)  # predict the dolphin
        
        boxCoordi = []
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                
                b = box.xyxy[0]  # get box coordinates in (left, top, right, bottom) format
                b_array = b.numpy()
                boxCoordi.append([b_array[0], b_array[1], b_array[2], b_array[3]])   
        
        return boxCoordi
                
        
