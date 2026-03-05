# ─── utils/video_utils.py ─────────────────────────────────────
import cv2, os

import config

def open_video(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    fps    = cap.get(cv2.CAP_PROP_FPS)
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return cap, fps, total, w, h

def make_writer(out_path, fps, w, h):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*getattr(config, "OUTPUT_CODEC", "mp4v"))
    return cv2.VideoWriter(out_path, fourcc, fps, (w, h))