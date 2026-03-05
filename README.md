# Traffic Detection Pipeline

An overhead highway traffic analysis pipeline using YOLOv4 + background subtraction to detect, track, and count vehicles across two lanes in real time.

## Features
- **Vehicle detection** via YOLOv4 (car, truck, bus, motorbike, bicycle)
- **Infrastructure detection** (traffic lights, signs, gantry boards) — always classified static
- **Dynamic / Static classification** using MOG2 background subtraction + perspective-normalised centroid displacement
- **Multi-frame centroid tracker** with IoU + distance matching
- **Lane counting** — two horizontal trip-wire lines count vehicles crossing in each direction
- Outputs annotated colour video, grayscale confirmation video, per-object CSV, per-frame summary CSV

## Project Structure
```
traffic_project/
├── main.py                  # Pipeline entry point
├── config.py                # All tunable parameters
├── requirements.txt
├── steps/
│   ├── step2_crop.py        # Trim end-credits from video
│   ├── step3_detection.py   # YOLOv4 detector
│   ├── step4_count.py       # Per-frame object counter
│   ├── step5_subtraction.py # MOG2 background subtractor
│   ├── step6_union.py       # fg_ratio classification
│   └── step7_overlap.py     # Tracker + lane crossing counter
└── utils/
    ├── video_utils.py       # VideoCapture / VideoWriter helpers
    └── visualizer.py        # Drawing utilities
```

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download YOLOv4 model files
Place the following in a `models/` directory:
```
models/yolov4.cfg
models/yolov4.weights     # https://github.com/AlexeyAB/darknet/releases
models/coco.names         # https://github.com/pjreddie/darknet/blob/master/data/coco.names
```

### 3. Configure
Edit `config.py`:
```python
INPUT_VIDEO = "your_video.mp4"

# Lane counting lines (fraction of frame height)
LINE1_Y_FRAC = 0.35   # upper line — visual reference
LINE2_Y_FRAC = 0.65   # lower line — counting wire for both lanes
```

### 4. Run
```bash
python main.py
```

## Lane Counting Logic

Two horizontal lines span the full frame width:

| Line | Position | Role |
|------|----------|------|
| L1 (cyan)    | 35% from top | Visual reference |
| L2 (magenta) | 65% from top | Counting wire |

- **Lane 1** (left carriageway, bottom → top): counted when centroid crosses L2 going **upward**
- **Lane 2** (right carriageway, top → bottom): counted when centroid crosses L2 going **downward**

Each track ID is counted only once per lane.

## Outputs
All outputs are saved to the `output/` directory:

| File | Description |
|------|-------------|
| `annotated.mp4` | Colour video with bounding boxes, HUD, lane lines |
| `confirmation_grayscale.mp4` | White = dynamic, grey = static, black = background |
| `detection_log.csv` | Per-detection log (every processed frame) |
| `frame_summary.csv` | Per-frame summary including lane counts |

## Key Config Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `PROCESS_EVERY` | 5 | Run YOLO every N frames (performance) |
| `CONF_THRESH` | 0.4 | YOLO confidence threshold |
| `FG_RATIO_DYNAMIC_THRESH` | 0.08 | MOG2 fg pixel ratio to call a vehicle dynamic |
| `STATIC_PIXEL_THRESH` | 8 | Normalised displacement (px/frame) below = static |
| `FRAMES_TO_CONFIRM` | 5 | Frames before static classification is confirmed |
| `LINE1_Y_FRAC` | 0.35 | Upper counting line position |
| `LINE2_Y_FRAC` | 0.65 | Lower counting line (active counting wire) |
