# ─── main.py ──────────────────────────────────────────────────
import os, cv2, csv, time
import numpy as np
import config
from utils.video_utils  import open_video, make_writer
from utils.visualizer   import draw_detections, draw_count

from steps.step2_crop        import crop_video
from steps.step3_detection   import TinyYOLODetector
from steps.step4_count       import count_objects
from steps.step5_subtraction import BackgroundSubtractor
from steps.step6_union       import classify_detections
from steps.step7_overlap     import StaticObjectTracker

# ── 1. Crop credits ───────────────────────────────────────────
os.makedirs(config.OUTPUT_DIR, exist_ok=True)
cropped_path = os.path.join(config.OUTPUT_DIR, "cropped.mp4")
crop_video(config.INPUT_VIDEO, cropped_path)

# ── 2. Init models ────────────────────────────────────────────
print("[Step 3] Loading YOLO...")
detector   = TinyYOLODetector()
subtractor = BackgroundSubtractor()

# ── 3. Open cropped video ─────────────────────────────────────
cap, fps, total, w, h = open_video(cropped_path)
print(f"[Info] Video: {w}x{h} @ {fps:.1f} FPS | {total} frames")
print(f"[Info] Processing every {config.PROCESS_EVERY} frame(s)")

# ── Init tracker with frame height for line positions ─────────
tracker = StaticObjectTracker(frame_h=h)

# Retrieve line Y positions for drawing
LINE1_Y = tracker.line1_y
LINE2_Y = tracker.line2_y
print(f"[Info] Counting lines: L1 y={LINE1_Y}px  L2 y={LINE2_Y}px")

out_path      = os.path.join(config.OUTPUT_DIR, "annotated.mp4")
gray_out_path = os.path.join(config.OUTPUT_DIR, "confirmation_grayscale.mp4")
writer      = make_writer(out_path,      fps, w, h)
gray_writer = make_writer(gray_out_path, fps, w, h)

# ── CSV setup ─────────────────────────────────────────────────
csv_path   = os.path.join(config.OUTPUT_DIR, "detection_log.csv")
csv_fields = ["frame", "timestamp_sec", "timestamp_hms",
              "track_id", "label", "confidence",
              "bbox_x", "bbox_y", "bbox_w", "bbox_h",
              "status", "fg_ratio", "object_type"]
csv_file   = open(csv_path, "w", newline="")
csv_writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
csv_writer.writeheader()

summary_path   = os.path.join(config.OUTPUT_DIR, "frame_summary.csv")
summary_fields = ["frame", "timestamp_sec", "timestamp_hms",
                  "total_objects", "dynamic", "static",
                  "cars", "trucks", "buses",
                  "infrastructure", "cumulative_unique",
                  "lane1_total", "lane2_total"]
summary_file   = open(summary_path, "w", newline="")
summary_writer = csv.DictWriter(summary_file, fieldnames=summary_fields)
summary_writer.writeheader()

# ── State ─────────────────────────────────────────────────────
cumulative_track_ids = set()
last_det_with_status = []
last_counts          = {"total": 0}
last_fg_mask         = np.zeros((h, w), dtype=np.uint8)
pipeline_start       = time.time()

print(f"[Pipeline] Processing {total} frames... (press Q to quit)")

for frame_idx in range(total):
    ret, frame = cap.read()
    if not ret:
        break

    timestamp_sec = frame_idx / fps
    mins  = int(timestamp_sec // 60)
    secs  = int(timestamp_sec % 60)
    msecs = int((timestamp_sec % 1) * 1000)
    timestamp_hms = f"{mins:02d}:{secs:02d}.{msecs:03d}"

    is_processed = (frame_idx % config.PROCESS_EVERY == 0)

    if is_processed:
        fg_mask          = subtractor.apply(frame, is_processed=True)
        detections       = detector.detect(frame)
        counts           = count_objects(detections)
        detections       = classify_detections(detections, fg_mask)
        static_track_ids = tracker.update(detections)
        last_fg_mask     = fg_mask

        det_with_status = []
        for det in detections:
            tid = det.get("track_id")

            if det.get("is_infrastructure"):
                status      = "static"
                object_type = "infrastructure"
            elif det.get("fg_ratio", 0) >= config.FG_RATIO_DYNAMIC_THRESH:
                status      = "dynamic"
                object_type = "vehicle"
            elif tid in static_track_ids:
                status      = "static"
                object_type = "vehicle"
            else:
                status      = "dynamic"
                object_type = "vehicle"

            det_with_status.append({**det,
                "status":      status,
                "object_type": object_type,
            })
            if tid is not None:
                cumulative_track_ids.add(tid)

        last_det_with_status = det_with_status
        last_counts          = counts

        for det in det_with_status:
            x, y, bw, bh = det["bbox"]
            csv_writer.writerow({
                "frame":         frame_idx,
                "timestamp_sec": round(timestamp_sec, 3),
                "timestamp_hms": timestamp_hms,
                "track_id":      det.get("track_id", ""),
                "label":         det["label"],
                "confidence":    round(det["conf"], 3),
                "bbox_x": x, "bbox_y": y, "bbox_w": bw, "bbox_h": bh,
                "status":        det["status"],
                "fg_ratio":      round(det.get("fg_ratio", 0), 3),
                "object_type":   det.get("object_type", "vehicle"),
            })
    else:
        subtractor.apply(frame, is_processed=False)
        det_with_status = last_det_with_status
        counts          = last_counts

    # ── Get lane counts ───────────────────────────────────────
    lane1_count, lane2_count = tracker.get_lane_counts()

    # ── Grayscale confirmation frame ──────────────────────────
    gray_canvas = np.zeros((h, w), dtype=np.uint8)
    for det in det_with_status:
        x, y, bw, bh = det["bbox"]
        x  = max(0, x);  y  = max(0, y)
        x2 = min(w, x + bw); y2 = min(h, y + bh)
        if det["status"] == "dynamic":
            gray_canvas[y:y2, x:x2] = 255
        else:
            gray_canvas[y:y2, x:x2] = 60

    fg_blended = cv2.addWeighted(gray_canvas, 0.75, last_fg_mask, 0.25, 0)

    for det in det_with_status:
        x, y, bw, bh = det["bbox"]
        shade = 255 if det["status"] == "dynamic" else 80
        cv2.rectangle(fg_blended, (x, y), (x+bw, y+bh), shade, 2)
        tag = "DYN" if det["status"] == "dynamic" else "STA"
        lbl = f"{det['label']} | {tag}"
        cv2.putText(fg_blended, lbl, (x, max(y-4, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, shade, 1, cv2.LINE_AA)

    # Draw counting lines on grayscale
    cv2.line(fg_blended, (0, LINE1_Y), (w, LINE1_Y), 200, 2)
    cv2.line(fg_blended, (0, LINE2_Y), (w, LINE2_Y), 200, 2)
    cv2.putText(fg_blended, "L1", (5, LINE1_Y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, 200, 1, cv2.LINE_AA)
    cv2.putText(fg_blended, "L2", (5, LINE2_Y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, 200, 1, cv2.LINE_AA)

    cv2.putText(fg_blended, "WHITE=Dynamic  GRAY=Static  BLACK=Background",
                (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, 180, 1, cv2.LINE_AA)
    cv2.putText(fg_blended, timestamp_hms,
                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, 200, 1, cv2.LINE_AA)

    gray_bgr = cv2.cvtColor(fg_blended, cv2.COLOR_GRAY2BGR)
    gray_writer.write(gray_bgr)

    # ── Annotate colour frame ─────────────────────────────────
    vis = frame.copy()

    # ── Draw counting lines ───────────────────────────────────
    # L1 — cyan line (upper)
    cv2.line(vis, (0, LINE1_Y), (w, LINE1_Y), (255, 255, 0), 2)
    cv2.putText(vis, "L1  (Lane 2: crosses upward)",
                (10, LINE1_Y - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)

    # L2 — magenta line (lower)
    cv2.line(vis, (0, LINE2_Y), (w, LINE2_Y), (255, 0, 255), 2)
    cv2.putText(vis, "L2  (Lane 1: crosses downward)",
                (10, LINE2_Y - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1, cv2.LINE_AA)

    # ── Draw detections ───────────────────────────────────────
    for det in det_with_status:
        x, y, bw, bh = det["bbox"]
        is_infra = det.get("object_type") == "infrastructure"

        if det["status"] == "static":
            color = (0, 165, 255) if is_infra else (0, 0, 255)
        else:
            color = (0, 255, 0)

        label = (f"[{det.get('track_id','?')}] {det['label']} "
                 f"{det['conf']:.2f} | {det['status']}"
                 + (" [INFRA]" if is_infra else ""))
        cv2.rectangle(vis, (x, y), (x+bw, y+bh), color, 2)
        cv2.putText(vis, label, (x, max(y-5, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)

    # ── HUD ───────────────────────────────────────────────────
    n_dynamic = sum(1 for d in det_with_status if d["status"] == "dynamic")
    n_static  = sum(1 for d in det_with_status if d["status"] == "static")
    n_infra   = sum(1 for d in det_with_status if d.get("object_type") == "infrastructure")
    n_total   = len(det_with_status)
    cumulative_unique = len(cumulative_track_ids)

    overlay = vis.copy()
    cv2.rectangle(overlay, (0, 0), (360, 240), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, vis, 0.45, 0, vis)

    hud = [
        (f"Time  : {timestamp_hms}",              (10, 22),  (200, 200, 200)),
        (f"Frame : {frame_idx}/{total}",           (10, 44),  (200, 200, 200)),
        (f"FPS   : {fps:.1f}",                     (10, 66),  (200, 200, 200)),
        (f"This frame  : {n_total} objects",       (10, 96),  (255, 255, 255)),
        (f"  Dynamic   : {n_dynamic}",             (10, 118), (50,  220,  50)),
        (f"  Static    : {n_static}",              (10, 140), (50,  100, 255)),
        (f"  Infra     : {n_infra}",               (10, 162), (0,   165, 255)),
        (f"Total unique: {cumulative_unique}",     (10, 187), (0,   220, 255)),
        # Lane counts
        (f"Lane 1 (L1→L2): {lane1_count}",        (10, 215), (255, 0,   255)),
        (f"Lane 2 (L2→L1): {lane2_count}",        (10, 237), (255, 255,   0)),
    ]

    # Extend the HUD box for the extra rows
    overlay2 = vis.copy()
    cv2.rectangle(overlay2, (0, 0), (360, 255), (0, 0, 0), -1)
    cv2.addWeighted(overlay2, 0.55, vis, 0.45, 0, vis)

    for text, pos, color in hud:
        cv2.putText(vis, text, pos,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.54, color, 1, cv2.LINE_AA)

    writer.write(vis)

    summary_writer.writerow({
        "frame":             frame_idx,
        "timestamp_sec":     round(timestamp_sec, 3),
        "timestamp_hms":     timestamp_hms,
        "total_objects":     n_total,
        "dynamic":           n_dynamic,
        "static":            n_static,
        "cars":              last_counts.get("car", 0),
        "trucks":            last_counts.get("truck", 0),
        "buses":             last_counts.get("bus", 0),
        "infrastructure":    n_infra,
        "cumulative_unique": cumulative_unique,
        "lane1_total":       lane1_count,
        "lane2_total":       lane2_count,
    })

    if config.SHOW_PREVIEW:
        cv2.imshow("Traffic Detection Pipeline", vis)
        cv2.imshow("Grayscale Confirmation (White=Moving, Black=Static)", fg_blended)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Interrupted by user.")
            break

    if frame_idx % 100 == 0 and frame_idx > 0:
        elapsed   = time.time() - pipeline_start
        est_total = (elapsed / frame_idx) * total
        remaining = max(0, est_total - elapsed)
        print(f"  [{frame_idx:5d}/{total}] {timestamp_hms} | "
              f"objects={n_total} dyn={n_dynamic} sta={n_static} | "
              f"Lane1={lane1_count} Lane2={lane2_count} | "
              f"ETA {remaining/60:.1f}min")

cap.release()
writer.release()
gray_writer.release()
csv_file.close()
summary_file.close()
cv2.destroyAllWindows()

elapsed = time.time() - pipeline_start
lane1_count, lane2_count = tracker.get_lane_counts()

print(f"\n✅ Done in {elapsed/60:.1f} minutes!")
print(f"   Annotated video         → {out_path}")
print(f"   Grayscale confirmation  → {gray_out_path}")
print(f"   Per-object CSV          → {csv_path}")
print(f"   Per-frame summary       → {summary_path}")
print(f"   Total unique objects    : {len(cumulative_track_ids)}")
print(f"   Lane 1 count (L1:L2)   : {lane1_count}")
print(f"   Lane 2 count (L2:mL1)   : {lane2_count}")