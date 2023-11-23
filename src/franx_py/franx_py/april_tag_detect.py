#!/usr/bin/env python
import cv2 
import numpy as np
import apriltag
from argparse import ArgumentParser

def generateCoordinate(pose):
    x = np.array([[0],
              [0],
              [0],
              [1]
    ])
    return np.dot(pose, x)
    
  
vid = cv2.VideoCapture(4) # 2 
options = apriltag.DetectorOptions(families='tag36h11')
detector = apriltag.Detector(options)

while(True): 
      
    ret, frame = vid.read() 
    result, overlay = apriltag.detect_tags(frame,
                                            detector,
                                            camera_params=(322.282410, 322.282410, 320.818268, 178.779297),
                                            tag_size= 0.06,
                                            vizualization=3,
                                            verbose=3,
                                            annotation=True
                                            )
    if len(result) > 1:
        # print("pose: {} ".format(result[1]))
        x = generateCoordinate(result[1])
        print("x: {}, y: {}, z: {}".format(x[0][0], x[1][0], x[2][0]))
    cv2.imshow('frame', overlay) 
    
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

vid.release() 
cv2.destroyAllWindows() 
