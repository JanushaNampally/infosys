import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import joblib
import os

model, label_encoder = joblib.load("../models/activity_model.pkl")

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

VIDEO_PATH = "C:/Users/nampally janusha/OneDrive - K L University/Desktop/infosys/data/Jogging/Jogging 6.mp4"

OUTPUT_PATH = "../output/output_video.mp4"

os.makedirs("../output", exist_ok=True)

cap = cv2.VideoCapture(VIDEO_PATH)

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (frame_width, frame_height))

print("Processing video...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    predicted_label = "No Pose Detected"
    confidence = 0

    if results.pose_landmarks:
        row = []
        for landmark in results.pose_landmarks.landmark:
            row.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])

        X = pd.DataFrame([row])

        prediction = model.predict(X)
        probabilities = model.predict_proba(X)

        predicted_label = label_encoder.inverse_transform(prediction)[0]
        confidence = np.max(probabilities) * 100

    # Put prediction text
    cv2.putText(
        frame,
        f"Activity: {predicted_label}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Confidence: {confidence:.2f}%",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 0, 0),
        2
    )

    # Write frame to output video
    out.write(frame)

    # Show frame (optional)
    cv2.imshow("Activity Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print("Full video processed!")