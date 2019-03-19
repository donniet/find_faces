#!/bin/bash


./find_faces \
    --classifierDescription /home/pi/src/detect_faces/facenet-model/FP16/20180402-114759.xml \
    --classifierWeights /home/pi/src/detect_faces/facenet-model/FP16/20180402-114759.bin \
    --detectorDescription /home/pi/src/detect_faces/face-detection-model/FP16/face-detection-adas-0001.xml \
    --detectorWeights /home/pi/src/detect_faces/face-detection-model/FP16/face-detection-adas-0001.bin \
    --device MYRIAD \
    
