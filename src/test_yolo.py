from ultralytics import YOLO
import cv2

# Load pretrained YOLOv8 model
model = YOLO("yolov8n.pt")  # nano version (fast)

# Open test video
cap = cv2.VideoCapture("data/Jogging/Jogging 1.mp4")  # change path if needed

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection
    results = model(frame)

    # Plot detections
    annotated_frame = results[0].plot()

    cv2.imshow("YOLO Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()