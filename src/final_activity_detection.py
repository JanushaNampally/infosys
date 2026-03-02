import cv2
import mediapipe as mp
import numpy as np
import joblib
from ultralytics import YOLO

# YOLO model
yolo_model = YOLO("yolov8n.pt")

# RandomForest activity model
model, label_encoder = joblib.load("models/activity_model.pkl")

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

UNSAFE_OBJECTS = ["knife", "scissors","fire"]

SAFE_CLASSES = [
    "Walking",
    "Jogging",
    "Yoga_Meditation",
    "Sitting"
]

#cap = cv2.VideoCapture("data/Testing data/testing data 3.mp4")
#cap = cv2.VideoCapture("data/Jogging/Jogging 1.mp4")
cap = cv2.VideoCapture("data/Testing data/testing_data_2.mp4")
if not cap.isOpened():
    print("ERROR: Video not opened")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("output_recorded.mp4",
                      fourcc,
                      20.0,
                      (frame_width, frame_height))


while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = yolo_model(frame)

    person_boxes = []

    for box in results[0].boxes:
        class_id = int(box.cls[0])
        class_name = yolo_model.names[class_id]

        # Collect persons
        if class_name == "person":
            person_boxes.append(box)

        # Detect unsafe objects
        if class_name in UNSAFE_OBJECTS:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            cv2.rectangle(frame, (x1, y1), (x2, y2),
                          (0, 0, 255), 3)

            cv2.putText(frame,
                        f"UNSAFE OBJECT: {class_name}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2)

    for box in person_boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        person_crop = frame[y1:y2, x1:x2]

        # Skip invalid crops
        if person_crop.size == 0:
            continue

        rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb)

        if result.pose_landmarks:
            landmarks = []

            for lm in result.pose_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z, lm.visibility])

            landmarks = np.array(landmarks).reshape(1, -1)

            prediction = model.predict(landmarks)
            activity_name = label_encoder.inverse_transform(prediction)[0]

            # Check if safe
            if activity_name in SAFE_CLASSES:
                color = (0, 255, 0)
                label = f"SAFE: {activity_name}"
            else:
                color = (0, 0, 255)
                label = f"UNSAFE: {activity_name}"

        else:
            color = (0, 0, 255)
            label = "UNSAFE: No Pose"

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

        cv2.putText(frame,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2)

    # -----------------------------
    # Write & Display Frame
    # -----------------------------
    out.write(frame)

    cv2.imshow("Final Activity Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# -----------------------------
# Release Resources
# -----------------------------

cap.release()
out.release()
cv2.destroyAllWindows()

print("Processing completed.")
print("Output saved as: output_recorded.mp4")