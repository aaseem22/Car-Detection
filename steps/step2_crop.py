# ─── steps/step2_crop.py ──────────────────────────────────────
"""
Step 2: Crop the end-credits portion of the video.
Outputs a trimmed video (and optionally crops pixel rows per frame).
"""
import cv2
from utils.video_utils import open_video, make_writer
import config

def crop_video(input_path: str, output_path: str) -> str:
    cap, fps, total, w, h = open_video(input_path)

    # Calculate last frame to keep
    trim_frames = int(config.CREDITS_TRIM_SECONDS * fps)
    keep_until  = total - trim_frames
    print(f"[Step 2] Total frames: {total} | Keeping: {keep_until} | Trimming last {trim_frames} frames")

    # Adjust height if pixel-row crop is set
    if config.CROP_ROWS:
        r0, r1 = config.CROP_ROWS
        out_h = r1 - r0
    else:
        r0, r1 = 0, h
        out_h  = h

    writer = make_writer(output_path, fps, w, out_h)
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame_idx >= keep_until:
            break
        cropped = frame[r0:r1, :]
        writer.write(cropped)
        frame_idx += 1

    cap.release()
    writer.release()
    print(f"[Step 2] Saved cropped video → {output_path}")
    return output_path