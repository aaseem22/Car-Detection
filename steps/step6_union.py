# ─── steps/step6_union.py ─────────────────────────────────────
"""
Step 6: Union of fg mask + YOLO bboxes.

ADDITIONAL FIX: When fg_ratio=0.00 for all cars (MOG2 failure),
fall back to bbox SIZE change as motion signal.
Small/distant cars that are moving will have consistent detection
but their bbox won't stay perfectly still frame-to-frame.

Primary:   fg_ratio >= FG_RATIO_DYNAMIC_THRESH  → dynamic
Secondary: stored in det for step7 to use
"""
import numpy as np
import config

def mask_overlap_ratio(mask, bbox) -> float:
    x, y, w, h = bbox
    x  = max(0, x);  y  = max(0, y)
    x2 = min(mask.shape[1], x + w)
    y2 = min(mask.shape[0], y + h)
    if x2 <= x or y2 <= y:
        return 0.0
    roi       = mask[y:y2, x:x2]
    active_px = np.count_nonzero(roi)
    total_px  = (x2 - x) * (y2 - y)
    return active_px / total_px if total_px > 0 else 0.0

def classify_detections(detections: list, fg_mask) -> list:
    for det in detections:
        ratio = mask_overlap_ratio(fg_mask, det["bbox"])
        det["fg_ratio"] = ratio
        if ratio >= config.FG_RATIO_DYNAMIC_THRESH:
            det["motion_label"] = "dynamic"
        else:
            det["motion_label"] = "unknown"   # let step7 tracker decide
    return detections