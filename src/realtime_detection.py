import cv2
import mediapipe as mp
import numpy as np
import joblib

# Load trained model
model, label_encoder = joblib.load("../models/activity_model.pkl")

# Initialize MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Start webcam
cap = cv2.VideoCapture(0)

print("Press Q to exit...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    if results.pose_landmarks:
        landmarks = []

        for landmark in results.pose_landmarks.landmark:
            landmarks.extend([
                landmark.x,
                landmark.y,
                landmark.z,
                landmark.visibility
            ])

        # Convert to numpy array
        input_data = np.array(landmarks).reshape(1, -1)

        # Predict
        prediction = model.predict(input_data)
        label = label_encoder.inverse_transform(prediction)[0]

        # Draw pose
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

        # Display prediction
        cv2.putText(
            frame,
            f"Activity: {label}",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

    cv2.imshow("Activity Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()