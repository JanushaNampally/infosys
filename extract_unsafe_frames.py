import cv2
import os

# Path to unsafe videos folder
VIDEO_FOLDER = "data/unsafe"   # change if different
OUTPUT_FOLDER = "unsafe_yolo/images/train"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

frame_count = 0

for video_name in os.listdir(VIDEO_FOLDER):
    video_path = os.path.join(VIDEO_FOLDER, video_name)

    cap = cv2.VideoCapture(video_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Save every 10th frame (to reduce duplicates)
        if frame_count % 10 == 0:
            image_name = f"frame_{frame_count}.jpg"
            cv2.imwrite(os.path.join(OUTPUT_FOLDER, image_name), frame)

        frame_count += 1

    cap.release()

print("Frame extraction completed.")