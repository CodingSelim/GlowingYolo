# Glowing YOLO

Lightweight demo that runs Ultralytics YOLOv8 on a video and renders label-free, per-class colored bounding boxes with a soft neon glow. Includes sample weights, a test clip, the rendered output, and a preview GIF.

## Preview
![Glowing YOLO preview](runs/glowing_yolo/preview.gif)

## Quick start
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run_glowing_yolo.py --input myvideo.mp4 --output runs/glowing_yolo/output.mp4 --model yolov8n.pt --conf 0.25 --glow-radius 12 --line-thickness 2
```

## How it works
- Loads a YOLOv8 model (default `yolov8n.pt`), runs per-frame inference on the input video.
- Deterministic per-class colors (overrides for common traffic classes, otherwise hash-based hues).
- Renders two-layer boxes: a blurred glow plus a crisp outline on top.
- Writes the processed video to `runs/glowing_yolo/output.mp4`; creates parent directories when missing.

## Assets included
- `myvideo.mp4` – sample input clip.
- `yolov8n.pt` and `yolov8l.pt` – model weights (use `--model` to switch).
- `runs/glowing_yolo/output.mp4` – rendered output from the sample clip.
- `runs/glowing_yolo/preview.gif` – 4-second excerpt (at 4s–8s) for quick viewing.

## Make your own preview GIF
```bash
ffmpeg -y -ss 4 -t 4 -i runs/glowing_yolo/output.mp4 -vf "fps=15,scale=640:-1:flags=lanczos,palettegen" /tmp/palette.png
ffmpeg -y -ss 4 -t 4 -i runs/glowing_yolo/output.mp4 -i /tmp/palette.png -lavfi "fps=15,scale=640:-1:flags=lanczos [x]; [x][1:v] paletteuse" runs/glowing_yolo/preview.gif
```
