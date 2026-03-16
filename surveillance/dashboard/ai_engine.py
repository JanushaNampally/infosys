import os
import warnings

# Suppress TensorFlow Lite / MediaPipe C++ log noise before importing mediapipe
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("GLOG_minloglevel", "3")
os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "3")

import cv2
import joblib
import mediapipe as mp
import pandas as pd
import subprocess
import numpy as np
from collections import Counter
from contextlib import nullcontext

from ultralytics import YOLO
from django.conf import settings

warnings.filterwarnings(
    "ignore",
    message=r"SymbolDatabase\.GetPrototype\(\) is deprecated.*",
    category=UserWarning,
)

MODEL_PATH = os.path.join(settings.BASE_DIR.parent, "models", "activity_model.pkl")
_RUNTIME = None


def _ensure_runtime():
    global _RUNTIME
    if _RUNTIME is not None:
        return _RUNTIME

    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Activity model file not found: {MODEL_PATH}")

    loaded = joblib.load(MODEL_PATH)
    model, label_encoder = loaded

    feature_columns = list(
        getattr(
            model,
            "feature_names_in_",
            [axis + str(i) for i in range(33) for axis in ("x", "y", "z", "v")],
        )
    )

    pose_api = None
    if hasattr(mp, "solutions") and hasattr(mp.solutions, "pose"):
        pose_api = mp.solutions.pose

    yolo_model = YOLO("yolov8n.pt")
    _RUNTIME = {
        "model": model,
        "label_encoder": label_encoder,
        "feature_columns": feature_columns,
        "mp_pose": pose_api,
        "yolo_model": yolo_model,
    }
    return _RUNTIME

SAFE_CLASSES = {"Walking", "Running", "Jogging", "Yoga_Meditation", "Sitting"}

SPEED_PROFILES = {
    "fast": {
        "frame_skip": 4,
        "yolo_imgsz": 320,
        "yolo_conf": 0.50,
        "max_people": 1,
        "output_fps": 8.0,
        "ffmpeg_crf": "31",
        "ffmpeg_preset": "ultrafast",
    },
    "balanced": {
        "frame_skip": 3,
        "yolo_imgsz": 416,
        "yolo_conf": 0.45,
        "max_people": 2,
        "output_fps": 10.0,
        "ffmpeg_crf": "30",
        "ffmpeg_preset": "ultrafast",
    },
    "accurate": {
        "frame_skip": 1,
        "yolo_imgsz": 640,
        "yolo_conf": 0.30,
        "max_people": 4,
        "output_fps": 20.0,
        "ffmpeg_crf": "24",
        "ffmpeg_preset": "fast",
    },
}


def _runtime_config(speed_mode):
    mode = (speed_mode or "balanced").lower()
    base = dict(SPEED_PROFILES.get(mode, SPEED_PROFILES["balanced"]))

    base["frame_skip"] = int(os.getenv("AI_FRAME_SKIP", str(base["frame_skip"])))
    base["yolo_imgsz"] = int(os.getenv("AI_YOLO_IMGSZ", str(base["yolo_imgsz"])))
    base["yolo_conf"] = float(os.getenv("AI_YOLO_CONF", str(base["yolo_conf"])))
    base["max_people"] = int(os.getenv("AI_MAX_PEOPLE", str(base["max_people"])))
    base["output_fps"] = float(os.getenv("AI_OUTPUT_FPS", str(base["output_fps"])))

    return mode, base


def _clamp_box(x1, y1, x2, y2, width, height):
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width - 1))
    y2 = max(0, min(y2, height - 1))
    return x1, y1, x2, y2

def process_video(input_path, output_path, speed_mode="balanced"):
    runtime = _ensure_runtime()
    model = runtime["model"]
    label_encoder = runtime["label_encoder"]
    feature_columns = runtime["feature_columns"]
    mp_pose = runtime["mp_pose"]
    yolo_model = runtime["yolo_model"]

    speed_mode, config = _runtime_config(speed_mode)

    cap = cv2.VideoCapture(input_path)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    target_fps = config["output_fps"] if config["output_fps"] > 0 else source_fps
    if not target_fps or np.isnan(target_fps):
        target_fps = 10.0

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, target_fps, (frame_width, frame_height))

    activity_counts = Counter()
    safe_frame_count = 0
    unsafe_frame_count = 0
    no_pose_count = 0
    warning = None
    frame_index = -1
    last_detections = []

    if mp_pose is None:
        warning = (
            "MediaPipe Pose API is unavailable in the installed mediapipe package. "
            "Showing person detections only; activity labels are disabled."
        )

    pose_context = (
        mp_pose.Pose(model_complexity=0, static_image_mode=False)
        if mp_pose is not None
        else nullcontext(None)
    )

    with pose_context as pose:
        while True:

            ret, frame = cap.read()

            if not ret:
                break

            frame_index += 1

            should_infer = frame_index % max(1, config["frame_skip"]) == 0

            if should_infer:
                detections_for_frame = []

                results = yolo_model(
                    frame,
                    verbose=False,
                    classes=[0],
                    conf=config["yolo_conf"],
                    imgsz=config["yolo_imgsz"],
                )

                person_boxes = []
                for box in results[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    x1, y1, x2, y2 = _clamp_box(x1, y1, x2, y2, frame_width, frame_height)
                    area = max(0, x2 - x1) * max(0, y2 - y1)
                    person_boxes.append((area, x1, y1, x2, y2))

                person_boxes.sort(reverse=True, key=lambda item: item[0])
                person_boxes = person_boxes[:max(1, config["max_people"])]

                for _, x1, y1, x2, y2 in person_boxes:
                    crop = frame[y1:y2, x1:x2]

                    if crop.size == 0:
                        continue

                    if pose is not None:
                        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                        result = pose.process(rgb)
                    else:
                        result = None

                    if result is not None and result.pose_landmarks:
                        landmarks = []

                        for lm in result.pose_landmarks.landmark:
                            landmarks.extend([lm.x, lm.y, lm.z, lm.visibility])

                        X = pd.DataFrame([landmarks], columns=feature_columns)
                        pred = model.predict(X)

                        activity = label_encoder.inverse_transform(pred)[0]
                        activity_counts[activity] += 1

                        if activity in SAFE_CLASSES:
                            color = (0, 255, 0)
                            label = f"SAFE: {activity}"
                            safe_frame_count += 1
                        else:
                            color = (0, 0, 255)
                            label = f"UNSAFE: {activity}"
                            unsafe_frame_count += 1
                    else:
                        color = (0, 165, 255)
                        label = "PERSON DETECTED"
                        no_pose_count += 1

                    detections_for_frame.append((x1, y1, x2, y2, color, label))

                last_detections = detections_for_frame

            for x1, y1, x2, y2, color, label in last_detections:
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2,
                )

            out.write(frame)

    cap.release()
    out.release()

    # Re-encode to H.264 + faststart so browsers can play the video inline
    tmp_path = output_path + ".tmp.mp4"
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", output_path,
                "-vcodec", "libx264", "-crf", config["ffmpeg_crf"], "-preset", config["ffmpeg_preset"],
                "-movflags", "+faststart",
                "-an",
                tmp_path,
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        os.replace(tmp_path, output_path)
    except (subprocess.CalledProcessError, FileNotFoundError):
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    safe_activities = sorted(
        activity for activity in activity_counts if activity in SAFE_CLASSES
    )
    unsafe_activities = sorted(
        activity for activity in activity_counts if activity not in SAFE_CLASSES
    )

    total_classified_frames = safe_frame_count + unsafe_frame_count

    if activity_counts and total_classified_frames:
        primary_activity, primary_count = activity_counts.most_common(1)[0]
        confidence_score = round((primary_count / total_classified_frames) * 100, 2)
    else:
        primary_activity = "No Activity Detected"
        confidence_score = 0.0

    if unsafe_activities:
        status = "Unsafe"
    elif safe_activities:
        status = "Safe"
    else:
        status = "Unknown"

    return {
        "safe_count": len(safe_activities),
        "unsafe_count": len(unsafe_activities),
        "safe_activities": safe_activities,
        "unsafe_activities": unsafe_activities,
        "primary_activity": primary_activity,
        "status": status,
        "confidence_score": confidence_score,
        "classified_frames": total_classified_frames,
        "no_pose_frames": no_pose_count,
        "has_unsafe_activity": bool(unsafe_activities),
        "warning": warning,
    }