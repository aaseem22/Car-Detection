# ─── steps/step3_detection.py ─────────────────────────────────
"""
Step 3: YOLOv4 detection with two detection modes:

VEHICLE classes  → subject to dynamic/static classification
STATIC classes   → always labelled static immediately (infrastructure)

COCO classes used:
  Vehicles (dynamic candidates): car=2, motorbike=3, bus=5, truck=7, bicycle=1
  Infrastructure (always static): 
    stop sign=11, bench=13, traffic light=9,
    chair=56, dining table=60 (catches gantry/overhead boards)
    
  For highway overhead footage, the key always-static objects are:
    - Overhead gantry boards → often detected as "bench" or "dining table"
    - Traffic signs          → "stop sign"
    - Traffic lights         → "traffic light"
"""
import cv2
import numpy as np
import config

# Vehicle class IDs — these get dynamic/static classification
VEHICLE_CLASS_IDS = {1, 2, 3, 5, 7}   # bicycle, car, motorbike, bus, truck

# Infrastructure class IDs — always static, detected separately
INFRASTRUCTURE_CLASS_IDS = {
    9,   # traffic light
    11,  # stop sign
    12,  # parking meter
    13,  # bench          (catches gantry boards)
    56,  # chair
    57,  # couch
    60,  # dining table   (catches large overhead boards)
    72,  # tv/monitor     (catches electronic signs)
    73,  # laptop
}

class TinyYOLODetector:
    def __init__(self):
        print(f"  Loading: {config.YOLO_CFG}")
        print(f"  Weights: {config.YOLO_WEIGHTS}")
        self.net = cv2.dnn.readNetFromDarknet(config.YOLO_CFG, config.YOLO_WEIGHTS)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        with open(config.YOLO_NAMES) as f:
            self.classes = [l.strip() for l in f.readlines()]

        layer_names = self.net.getLayerNames()
        out_indices = self.net.getUnconnectedOutLayers()
        if hasattr(out_indices, 'flatten'):
            out_indices = out_indices.flatten()
        self.out_layers = [layer_names[i - 1] for i in out_indices]
        print(f"  Output layers: {self.out_layers}")

    def detect(self, frame):
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            frame, 1/255.0, config.INPUT_SIZE, swapRB=True, crop=False
        )
        self.net.setInput(blob)
        outputs = self.net.forward(self.out_layers)

        boxes, confidences, class_ids = [], [], []

        for output in outputs:
            for det in output:
                scores = det[5:]
                cls_id = int(np.argmax(scores))
                conf   = float(scores[cls_id])

                if conf < config.CONF_THRESH:
                    continue

                # Accept vehicles AND infrastructure objects
                is_vehicle = cls_id in VEHICLE_CLASS_IDS
                is_infra   = cls_id in INFRASTRUCTURE_CLASS_IDS

                if not (is_vehicle or is_infra):
                    continue

                cx, cy, bw, bh = (det[0]*w, det[1]*h, det[2]*w, det[3]*h)
                x = int(cx - bw/2)
                y = int(cy - bh/2)
                boxes.append([x, y, int(bw), int(bh)])
                confidences.append(conf)
                class_ids.append(cls_id)

        if not boxes:
            return []

        indices = cv2.dnn.NMSBoxes(
            boxes, confidences, config.CONF_THRESH, config.NMS_THRESH
        )
        if len(indices) == 0:
            return []

        results = []
        for i in indices.flatten():
            cls_id   = class_ids[i]
            is_infra = cls_id in INFRASTRUCTURE_CLASS_IDS
            results.append({
                "id":           i,
                "label":        self.classes[cls_id],
                "conf":         confidences[i],
                "bbox":         tuple(boxes[i]),
                "class_id":     cls_id,
                "is_infrastructure": is_infra,   # flag for main.py
            })
        return results