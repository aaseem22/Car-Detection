# ─── steps/step5_subtraction.py ───────────────────────────────
"""
Step 5: Background subtraction.

FIX: With PROCESS_EVERY=5, MOG2 was learning cars INTO the background
(cars appear in same position for 4 skipped frames → model thinks it's bg).

Solutions applied:
  1. learningRate=0 on skipped frames  → bg model only updates on processed frames
  2. Higher history (500) → bg model slower to absorb moving objects
  3. Higher varThreshold  → less sensitive to small pixel changes
"""
import cv2
import numpy as np
import config

class BackgroundSubtractor:
    def __init__(self):
        self.subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500,           # was 50 — much slower to learn moving objects as bg
            varThreshold=60,       # was 40 — less noise sensitivity
            detectShadows=False
        )
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                config.MORPH_KERNEL)
        self._is_processed_frame = True

    def apply(self, frame, is_processed=True) -> np.ndarray:
        """
        is_processed: True = YOLO frame, False = skipped frame.
        On skipped frames we still update bg model but with learningRate=0
        so it doesn't absorb parked/slow objects into background.
        """
        lr = 0.005 if is_processed else 0.0   # KEY FIX: freeze bg on skipped frames
        mask = self.subtractor.apply(frame, learningRate=lr)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  self.kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)
        return mask