#!/bin/bash


# ./find_faces \
#     --classifierDescription /home/pi/src/detect_faces/facenet-model/FP16/20180402-114759.xml \
#     --classifierWeights /home/pi/src/detect_faces/facenet-model/FP16/20180402-114759.bin \
#     --detectorDescription /home/pi/src/detect_faces/face-detection-model/FP16/face-detection-adas-0001.xml \
#     --detectorWeights /home/pi/src/detect_faces/face-detection-model/FP16/face-detection-adas-0001.bin \
#     --device MYRIAD \
#     --distance 0.1 \
#     --height 1088 \
#     --width 1920 \
#     --notificationURL http://mirror.local:8080/api/faces/detections \
#     --motionNotificationURL http://mirror.local:9080/ \
#     --mbx 120 \
#     --mby 68 \
#     --magnitude 60 \
#     --total 10 \
#     --motion ~/motion \
#     --video ~/video

./find_faces \
    --classifierDescription /home/donniet/src/detect_faces/facenet-model/FP32/20180402-114759.xml \
    --classifierWeights /home/donniet/src/detect_faces/facenet-model/FP32/20180402-114759.bin \
    --detectorDescription /home/donniet/src/detect_faces/face-detection-model/FP32/face-detection-adas-0001.xml \
    --detectorWeights /home/donniet/src/detect_faces/face-detection-model/FP32/face-detection-adas-0001.bin \
    --device CPU \
    --distance 0.1 \
    --height 1080 \
    --width 1920 \
    --notificationURL http://mirror.local:8080/api/faces/detections \
    --motionNotificationURL http://mirror.local:9080/ \
    --mbx 120 \
    --mby 68 \
    --magnitude 60 \
    --totalMotion 10 \
    --video ~/test.rgb24
