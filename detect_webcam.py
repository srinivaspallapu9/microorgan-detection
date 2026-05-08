from ultralytics import YOLO
import cv2
from collections import Counter

# Load trained YOLO model
model = YOLO("runs/detect/microorganism_yolov8/weights/best.pt")

# Open webcam
cap = cv2.VideoCapture(0)

# Confidence threshold
CONF_THRESHOLD = 0.75 #only when  the threshold is met, the detection will be displayed on the screen. Adjust this value based on your needs.

if not cap.isOpened():
    print("Camera not opened. Check camera permission or index.")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to read frame.")
        break

    # Run YOLO detection with threshold
    results = model(frame, conf=CONF_THRESHOLD, iou=0.45)

    # Annotated frame
    annotated_frame = results[0].plot()

    # Count detected organisms
    detections = results[0].boxes.cls.tolist()
    names = model.names
    detected_classes = [names[int(cls)] for cls in detections]
    counts = Counter(detected_classes)

    # Display counts
    y = 35

    if len(counts) == 0:
        cv2.putText(
            annotated_frame,
            "No microorganism detected",
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )
    else:
        cv2.putText(
            annotated_frame,
            "Detected Count:",
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2
        )

        y += 35

        for cls_name, count in counts.items():
            text = f"{cls_name}: {count}"

            cv2.putText(
                annotated_frame,
                text,
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            y += 35

    # Show frame
    cv2.imshow("Microorganism Detection", annotated_frame)

    # Press q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()