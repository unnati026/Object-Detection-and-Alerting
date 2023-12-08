import cv2
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

model1 = YOLO('yolov8s.pt')
video_path = 'walk.mp4'

cap = cv2.VideoCapture(video_path)
alert_set = set(alerting_classes.keys())
alert_set.remove(0)

# Create the red-tinted overlay image once
red_tinted_overlay = np.tile(red_tint, (1, 1, 1))

while cap.isOpened():
    alert_flag = False
    alert_reason = []

    success, frame = cap.read()

    if success:
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

        text = 'ALERT!'
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.75
        thickness = 2

        size = cv2.getTextSize(text, font, font_scale, thickness)
        x = 0
        y = int((2 + size[0][1]))

        img = results[0].plot()
        if alert_flag:
            # Resize the red-tinted overlay to match the image size
            red_tinted_overlay = cv2.resize(red_tinted_overlay, (img.shape[1], img.shape[0]))
            img = cv2.addWeighted(img, 0.7, red_tinted_overlay, 0.3, 0)
            cv2.putText(img, text, (x, y), font, font_scale, (0, 0, 0), thickness)

            y += int(size[0][1]) + 10  # Move to the next line

            for cls, count in alert_reason:
                alert_text = f'{count} {alerting_classes[cls]}'
                cv2.putText(img, alert_text, (x, y), font, font_scale, (0, 0, 0), thickness)
                y += int(size[0][1]) + 10  # Move to the next line

        cv2.imshow("YOLOv8 Inference", img)
        del results

        key = cv2.waitKey(1)
        if key != -1:
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()

pickle_out = open("classifier.pkl", "wb")
pickle.dump(classifier, pickle_out)
pickle_out.close()
