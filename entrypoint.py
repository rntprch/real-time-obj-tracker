"""
Main entry point of the YOLOv8 tracking application.
"""

import argparse
from pathlib import Path

from src.model_loader import load_model
from src.utils import process_video


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run YOLOv8 detection on a video and save annotated output."
    )
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to input video."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="output.mp4",
        help="Path to save annotated video."
    )
    return parser.parse_args()


def main():
    """Script entry point."""
    args = parse_args()

    model = load_model("yolov8m.pt")
    process_video(
        video_path=Path(args.input_path),
        output_path=Path(args.output_path),
        model=model
    )


if __name__ == "__main__":
    main()
