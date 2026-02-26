import cv2
import mediapipe as mp
import os
import csv
import numpy as np
from tqdm import tqdm

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False)
mp_drawing = mp.solutions.drawing_utils

# Paths
DATA_PATH = "../data"
OUTPUT_FILE = "../dataset.csv"

# Activities
activities = ["Jogging", "Sitting", "Walking", "Yoga_Meditation"]

# Create CSV file
with open(OUTPUT_FILE, mode='w', newline='') as f:
    writer = csv.writer(f)

    # 33 landmarks * (x,y,z,visibility) = 132 features
    header = []
    for i in range(33):
        header += [f"x{i}", f"y{i}", f"z{i}", f"v{i}"]
    header.append("label")
    writer.writerow(header)

    # Loop through each activity
    for label in activities:
        folder_path = os.path.join(DATA_PATH, label)

        for video_name in tqdm(os.listdir(folder_path), desc=f"Processing {label}"):
            video_path = os.path.join(folder_path, video_name)
            cap = cv2.VideoCapture(video_path)

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(frame_rgb)

                if results.pose_landmarks:
                    row = []
                    for landmark in results.pose_landmarks.landmark:
                        row.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])

                    row.append(label)
                    writer.writerow(row)

            cap.release()

print("Dataset creation completed!")