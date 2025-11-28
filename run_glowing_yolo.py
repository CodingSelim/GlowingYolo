#!/usr/bin/env python3
import argparse
import hashlib
import os
from typing import Dict, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

CLASS_COLOR_OVERRIDES: Dict[str, Tuple[int, int, int]] = {
    "person": (255, 92, 156),
    "bicycle": (120, 222, 106),
    "car": (76, 154, 255),
    "motorcycle": (77, 209, 187),
    "bus": (255, 189, 76),
    "truck": (255, 127, 80),
    "train": (165, 105, 255),
    "traffic light": (255, 240, 140),
    "stop sign": (255, 115, 192),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run YOLOv8 with glowing, per-class bounding boxes."
    )
    parser.add_argument(
        "--input",
        default="myvideo.mp4",
        help="Path to input video.",
    )
    parser.add_argument(
        "--output",
        default="runs/glowing_yolo/output.mp4",
        help="Path to save the rendered video.",
    )
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="YOLOv8 model weights.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for detections.",
    )
    parser.add_argument(
        "--glow-radius",
        type=int,
        default=12,
        help="Gaussian blur radius for the glow effect.",
    )
    parser.add_argument(
        "--line-thickness",
        type=int,
        default=2,
        help="Thickness of the crisp box outline drawn on top of the glow.",
    )
    return parser.parse_args()


def hsv_to_bgr(h: float, s: float, v: float) -> Tuple[int, int, int]:
    hsv_pixel = np.uint8([[[h * 179, s * 255, v * 255]]])
    bgr_pixel = cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2BGR)[0][0]
    return int(bgr_pixel[0]), int(bgr_pixel[1]), int(bgr_pixel[2])


def color_for_class(name: str) -> Tuple[int, int, int]:
    if name in CLASS_COLOR_OVERRIDES:
        return CLASS_COLOR_OVERRIDES[name]

    digest = hashlib.sha1(name.encode("utf-8")).hexdigest()
    hue = (int(digest[:8], 16) % 360) / 360.0
    return hsv_to_bgr(hue, 0.72, 1.0)


def draw_glowing_box(
    frame: np.ndarray,
    xyxy: Tuple[int, int, int, int],
    color: Tuple[int, int, int],
    glow_radius: int,
    line_thickness: int,
) -> None:
    x1, y1, x2, y2 = map(int, xyxy)
    glow_layer = np.zeros_like(frame)
    glow_thickness = max(glow_radius * 2, line_thickness + 2)

    cv2.rectangle(glow_layer, (x1, y1), (x2, y2), color, glow_thickness)
    glow_layer = cv2.GaussianBlur(glow_layer, (0, 0), sigmaX=glow_radius)
    cv2.addWeighted(glow_layer, 0.5, frame, 1.0, 0, dst=frame)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, line_thickness)


def ensure_output_dir(path: str) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)


def process_video(
    model: YOLO,
    input_path: str,
    output_path: str,
    conf: float,
    glow_radius: int,
    line_thickness: int,
) -> None:
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open input video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    ensure_output_dir(output_path)
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    class_color_map = {
        idx: color_for_class(name) for idx, name in model.model.names.items()
    }

    frame_idx = 0
    try:
        while True:
            success, frame = cap.read()
            if not success:
                break

            results = model(frame, conf=conf, verbose=False)[0]
            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cls_id = int(box.cls[0])
                color = class_color_map.get(cls_id, (255, 255, 255))
                draw_glowing_box(
                    frame,
                    (x1, y1, x2, y2),
                    color=color,
                    glow_radius=glow_radius,
                    line_thickness=line_thickness,
                )

            writer.write(frame)
            frame_idx += 1
            if frame_idx % 50 == 0:
                print(f"Processed {frame_idx} frames...", end="\r", flush=True)
    finally:
        cap.release()
        writer.release()
        print(f"\nSaved: {output_path}")


def main() -> None:
    args = parse_args()
    model = YOLO(args.model)
    process_video(
        model=model,
        input_path=args.input,
        output_path=args.output,
        conf=args.conf,
        glow_radius=args.glow_radius,
        line_thickness=args.line_thickness,
    )


if __name__ == "__main__":
    main()
