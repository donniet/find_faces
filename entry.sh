#!/bin/bash
/src/go/bin/find_faces \
    --classifierDescription /go/src/detect_faces/resnet50_128_caffe/FP16/resnet50_128.xml \
    --classifierWeights ~/src/detect_faces/resnet50_128_caffe/FP16/resnet50_128.bin \
    --detectorDescription ~/src/detect_faces/face-detection-model/FP16/face-detection-adas-0001.xml \
    --detectorWeights ~/src/detect_faces/face-detection-model/FP16/face-detection-adas-0001.bin \
    --video tcp://0.0.0.0:9990 \
    --device MYRIAD \
    --width 1920 \
    --height 1080