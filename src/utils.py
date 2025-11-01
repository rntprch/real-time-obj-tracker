"""
Utility functions for video reading, detection, drawing and saving.
"""
import numpy as np
from pathlib import Path

import cv2

from ultralytics import YOLO


def draw_boxes(frame, results, class_names):
    """Draw bounding boxes and labels for persons only.

    Args:
        frame (np.ndarray): Image frame to draw on.
        results: YOLO detection results.
        class_names (dict): Mapping from class index to name.

    Returns:
        np.ndarray: Frame with rendered detections.
    """
    for box in results.boxes:
        cls = int(box.cls[0])
        # Keep only "person" class
        if class_names[cls] != "person":
            continue
        
        x1, y1, x2, y2 = box.xyxy[0].int().tolist()
        conf = float(box.conf[0])
        label = f"{class_names[cls]} {conf:.2f}"
        
        # Draw rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw label
        cv2.putText(
            frame, label, (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
            (0, 255, 0), 2
        )
    return frame

def save_frame(frame: np.ndarray, output: cv2.VideoWriter):
    """Write a single frame to the output video file.

    Args:
        frame (np.ndarray): Image frame to save.
        output (cv2.VideoWriter): Initialized OpenCV video writer.
    """
    output.write(frame)

def process_video(video_path: Path, output_path: Path, model: YOLO):
    """Run YOLO detection on a video and save annotated output.

    Args:
        video_path (Path): Input video path.
        output_path (Path): Path where the output video will be saved.
        model (YOLO): YOLOv8 model instance.

    Raises:
        FileNotFoundError: If the video does not exist.
        RintimeError: If cannot open the video
    """
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h)
    )

    print("Processing video...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)[0]
        frame = draw_boxes(frame, results, model.names)

        save_frame(frame, out)

    cap.release()
    out.release()
    print(f"Saved annotated video to: {output_path}")
