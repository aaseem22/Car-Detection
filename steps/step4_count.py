# ─── steps/step4_count.py ─────────────────────────────────────
"""
Step 4: Count objects per frame, broken down by class label.
"""
from collections import Counter

def count_objects(detections: list) -> dict:
    """
    Args:
        detections: list of detection dicts from step3
    Returns:
        dict: {label: count, ..., 'total': N}
    """
    labels = [d["label"] for d in detections]
    counts = dict(Counter(labels))
    counts["total"] = len(detections)
    return counts