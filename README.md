# Object Detection and Alerting

## Overview

This project focuses on real-time activity recognition using computer vision and deep learning techniques. It utilizes the YOLO (You Only Look Once) object detection model to identify and classify objects of interest in a video stream. The primary goal is to detect suspicious activities by recognizing specific objects within the video frames.

## Object Classes

The project uses YOLO object detection for real-time video analysis. It has predefined classes for alerting based on the objects detected in the video frames:

- People
- Car
- Truck
- Backpack
- Suspicious handheld device
- Handbag
- Suitcase

## Project Workflow

The code performs the following steps:

1. Initializes the YOLO model with the provided weights.
2. Opens the video file.
3. Processes each frame of the video.
4. Detects and classifies objects in the frames.
5. Alerts if any alerting classes are detected in the frame, displaying the reasons for the alert in the image.
6. Continues processing frames until the video ends.

The code displays an "ALERT!" message on the video frame if any alerting objects are detected. It also lists the reasons for the alert, indicating which objects triggered the alert.
