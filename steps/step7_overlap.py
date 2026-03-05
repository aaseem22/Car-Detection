# ─── steps/step7_overlap.py ───────────────────────────────────
"""
Step 7: Multi-frame centroid tracking + Lane Counting.

GEOMETRY (bridge overhead view, two vertical carriageways):
  Lane 1 (LEFT  side): cars enter from BOTTOM, travel UPWARD   (Y decreases)
  Lane 2 (RIGHT side): cars enter from TOP,    travel DOWNWARD (Y increases)

Two horizontal trip-wire lines span the full frame width:
  L1 — upper line  (LINE1_Y_FRAC, default 0.35)  [CYAN]
  L2 — lower line  (LINE2_Y_FRAC, default 0.65)  [MAGENTA]

Counting rules — both use L2 as the single counting wire:
  Lane 1: centroid crosses L2 going UPWARD    (prev_cy > L2  AND  curr_cy <= L2)
  Lane 2: centroid crosses L2 going DOWNWARD  (prev_cy < L2  AND  curr_cy >= L2)

Using L2 (lower line) for both means:
  - Lane 1 cars (bottom→top) always pass through L2 before exiting top
  - Lane 2 cars (top→bottom) always pass through L2 before exiting bottom
  Every car is guaranteed to cross it exactly once.

L1 is drawn for visual reference only.

ROBUSTNESS: works even with PROCESS_EVERY frame skips — a fast car that
jumps across L2 between two processed frames is still caught because we
only need prev and curr to be on opposite sides of L2.
"""
import numpy as np
import config

REFERENCE_HEIGHT = 80

LINE1_Y_FRAC = getattr(config, "LINE1_Y_FRAC", 0.35)
LINE2_Y_FRAC = getattr(config, "LINE2_Y_FRAC", 0.65)


def iou(boxA, boxB) -> float:
    ax, ay, aw, ah = boxA
    bx, by, bw, bh = boxB
    ix  = max(ax, bx);   iy  = max(ay, by)
    ix2 = min(ax+aw, bx+bw); iy2 = min(ay+ah, by+bh)
    inter = max(0, ix2-ix) * max(0, iy2-iy)
    union = aw*ah + bw*bh - inter
    return inter/union if union > 0 else 0.0


def centroid(bbox):
    x, y, w, h = bbox
    return np.array([x + w/2, y + h/2], dtype=float)


def normalized_displacement(bbox1, bbox2) -> float:
    raw_disp = np.linalg.norm(centroid(bbox1) - centroid(bbox2))
    avg_h    = (bbox1[3] + bbox2[3]) / 2.0
    if avg_h < 1:
        return 0.0
    return raw_disp * (REFERENCE_HEIGHT / avg_h)


class StaticObjectTracker:
    def __init__(self, frame_h=720):
        self.tracks     = {}
        self.static_ids = set()
        self._next_id   = 0

        self.frame_h = frame_h
        self.line1_y = int(LINE1_Y_FRAC * frame_h)   # upper — visual only
        self.line2_y = int(LINE2_Y_FRAC * frame_h)   # lower — counting wire

        self.lane1_count = 0   # left  side, bottom→top: crosses L2 upward
        self.lane2_count = 0   # right side, top→bottom: crosses L2 downward

        self._counted_lane1: set = set()
        self._counted_lane2: set = set()
        self._prev_cy:       dict = {}

    def set_frame_size(self, frame_h: int):
        self.frame_h = frame_h
        self.line1_y = int(LINE1_Y_FRAC * frame_h)
        self.line2_y = int(LINE2_Y_FRAC * frame_h)

    # ------------------------------------------------------------------
    def _check_crossing(self, tid: int, prev_cy: float, curr_cy: float):
        l2 = self.line2_y

        # Lane 1 — moving UPWARD (Y decreasing), crossing L2
        if prev_cy > l2 and curr_cy <= l2:
            if tid not in self._counted_lane1:
                self.lane1_count += 1
                self._counted_lane1.add(tid)

        # Lane 2 — moving DOWNWARD (Y increasing), crossing L2
        if prev_cy < l2 and curr_cy >= l2:
            if tid not in self._counted_lane2:
                self.lane2_count += 1
                self._counted_lane2.add(tid)

    # ------------------------------------------------------------------
    def _match(self, bbox, existing_tracks):
        best_id, best_score = None, 0.15
        c_new = centroid(bbox)
        for tid, history in existing_tracks.items():
            last_bbox = history[-1][0]
            score = iou(bbox, last_bbox)
            dist  = np.linalg.norm(c_new - centroid(last_bbox))
            scale = max(bbox[3], last_bbox[3]) / REFERENCE_HEIGHT
            if dist < 60 * max(scale, 0.3):
                score = max(score, 0.16)
            if score > best_score:
                best_score = score
                best_id    = tid
        return best_id

    # ------------------------------------------------------------------
    def update(self, detections: list) -> set:
        updated = {}

        for det in detections:
            bbox     = det["bbox"]
            fg_ratio = det.get("fg_ratio", 0.0)

            tid = self._match(bbox, self.tracks)
            if tid is None:
                tid = self._next_id
                self._next_id += 1

            history = self.tracks.get(tid, [])
            history.append((bbox, fg_ratio))
            if len(history) > config.FRAMES_TO_CONFIRM:
                history = history[-config.FRAMES_TO_CONFIRM:]
            updated[tid] = history
            det["track_id"] = tid

            # Crossing check
            curr_cy = centroid(bbox)[1]
            if tid in self._prev_cy:
                self._check_crossing(tid, self._prev_cy[tid], curr_cy)
            self._prev_cy[tid] = curr_cy

        self.tracks     = updated
        self.static_ids = set()

        for tid, history in self.tracks.items():
            if len(history) < config.FRAMES_TO_CONFIRM:
                continue

            bboxes    = [h[0] for h in history]
            fg_ratios = [h[1] for h in history]
            avg_fg    = np.mean(fg_ratios)

            norm_disps = [normalized_displacement(bboxes[i], bboxes[i+1])
                          for i in range(len(bboxes)-1)]
            avg_norm_disp = np.mean(norm_disps)

            fg_says_moving   = avg_fg        >= config.FG_RATIO_DYNAMIC_THRESH
            disp_says_moving = avg_norm_disp >= config.STATIC_PIXEL_THRESH

            if not (fg_says_moving or disp_says_moving):
                self.static_ids.add(tid)

        return self.static_ids

    def get_lane_counts(self):
        return self.lane1_count, self.lane2_count