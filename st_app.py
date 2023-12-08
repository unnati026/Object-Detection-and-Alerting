import streamlit as st
import cv2
import tempfile
from ultralytics import YOLO
import numpy as np

alerting_classes = {
    0: 'People',
    2: 'Car',
    7: 'Truck',
    24: 'Backpack',
    65: 'Suspicious handheld device',
    26: 'Handbag',
    28: 'Suitcase',
}

red_tint = np.array([[[0, 0, 255]]], dtype=np.uint8)

model1 = YOLO('yolov8n.pt')

st.title("Object Detection and Recognition")

video_file = st.file_uploader("Choose a video file", type=["mp4"])

if video_file is not None:
    # Create temporary file for uploaded video
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())

    # Open video capture using temporary file path
    cap = cv2.VideoCapture(tfile.name)
    alert_set = set(alerting_classes.keys())
    alert_set.remove(0)

    # Create red-tinted overlay
    red_tinted_overlay = np.tile(red_tint, (1, 1, 1))

    stframe = st.empty()

    stop_button = st.button("Stop Inference")

    while cap.isOpened() and not stop_button:
        alert_flag = False
        alert_reason = []

        success, frame = cap.read()

        # if frame is read correctly ret is True
        if not success:
            st.warning("Can't receive frame (stream end?). Exiting ...")
            break

        if success:
            # Perform YOLO object detection
            results = model1(frame, conf=0.35, verbose=False, classes=list(alerting_classes.keys()))

            class_ids = results[0].boxes.cls.tolist()
            class_counts = {cls: class_ids.count(cls) for cls in set(class_ids)}

            for cls in alert_set:
                if cls in class_counts and class_counts[cls] > 0:
                    alert_flag = True
                    alert_reason.append((cls, class_counts[cls]))

            if class_counts.get(0, 0) > 5:
                alert_flag = True
                alert_reason.append((0, class_counts[0]))

            # Draw bounding boxes and alerts if necessary
            img = results[0].plot()
            if alert_flag:
                red_tinted_overlay = cv2.resize(red_tinted_overlay, (img.shape[1], img.shape[0]))
                img = cv2.addWeighted(img, 0.7, red_tinted_overlay, 0.3, 0)

            stframe.image(img, channels="BGR", caption="YOLOv8 Inference")

            del results
    cap.release()
    cv2.destroyAllWindows()
    tfile.close()
