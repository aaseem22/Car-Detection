# ─── utils/visualizer.py ──────────────────────────────────────
import cv2

COLORS = {
    "dynamic": (0, 255, 0),    # green
    "static":  (0, 0, 255),    # red
    "unknown": (255, 255, 0),  # yellow
}

def draw_detections(frame, detections, static_ids=set()):
    """
    detections: list of dicts with keys:
        id, label, confidence, bbox=(x,y,w,h), status='dynamic'|'static'
    """
    for det in detections:
        x, y, w, h = det["bbox"]
        status = "static" if det["id"] in static_ids else "dynamic"
        color  = COLORS[status]
        label  = f"[{det['id']}] {det['label']} {det['conf']:.2f} | {status}"
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, max(y-6, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return frame

def draw_count(frame, counts: dict):
    y = 20
    for label, cnt in counts.items():
        cv2.putText(frame, f"{label}: {cnt}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        y += 22
    return frame