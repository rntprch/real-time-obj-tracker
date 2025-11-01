"""
Module for loading YOLO models.
"""

import torch
from ultralytics import YOLO


def load_model(weights_path: str = "yolov8m.pt") -> YOLO:
    """Load a YOLOv model.

    Args:
        weights_path (str): Path to YOLO weights file.

    Returns:
        YOLO: Loaded YOLO model on the best available device.
    """
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model
    model = YOLO(weights_path)
    model.to(device)
    
    print(f"Model loaded on {device}")
    
    return model
