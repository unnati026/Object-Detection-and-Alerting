# Object-Detection-and-Alerting

This project focuses on real-time activity recognition using computer vision and deep learning techniques. It utilizes the YOLO (You Only Look Once) object detection model to identify and classify objects of interest in a video stream. The primary goal is to detect suspicious activities by recognising specific objects within the video frames.

The project uses YOLO object detection for real-time video analysis. It has predefined classes for alerting based on the objects detected in the video frames:

People
Car
Truck
Backpack
Suspicious handheld device
Handbag
Suitcase
The code performs the following steps:

Initializes the YOLO model with the provided weights.
Opens the video file.
Processes each frame of the video.
Detects and classifies objects in the frames.
Alerts if any alerting classes are detected in the frame, displaying the reasons for the alert in the image.
Continues processing frames until the video ends.
The code displays an "ALERT!" message on the video frame if any alerting objects are detected. It also lists the reasons for the alert, indicating which objects triggered the alert.
