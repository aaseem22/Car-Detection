# ─── config.py ────────────────────────────────────────────────

# ── Video Input ───────────────────────────────────────────────
INPUT_VIDEO = "videoplayback.mp4"

# ── Step 2: Crop Credits ──────────────────────────────────────
CREDITS_TRIM_SECONDS = 5
CROP_ROWS = None

# ── Step 3: YOLO Detection ────────────────────────────────────
YOLO_CFG     = "models/yolov4.cfg"
YOLO_WEIGHTS = "models/yolov4.weights"
YOLO_NAMES   = "models/coco.names"

CONF_THRESH   = 0.4
NMS_THRESH    = 0.3
INPUT_SIZE    = (416, 416)
VEHICLES_ONLY = True

# ── Performance ───────────────────────────────────────────────
PROCESS_EVERY = 5

# ── Step 5: Frame Subtraction ─────────────────────────────────
BG_HISTORY   = 500
BG_THRESH    = 60
MORPH_KERNEL = (5, 5)

# ── Step 6 ────────────────────────────────────────────────────
UNION_MIN_AREA = 500

# ── Step 7: Static/Dynamic Classification ────────────────────
FG_RATIO_DYNAMIC_THRESH = 0.08
STATIC_PIXEL_THRESH     = 8
FRAMES_TO_CONFIRM       = 5
OVERLAP_STATIC_THRESH   = 0.55

# ── Lane Counting Lines ───────────────────────────────────────
# Fractional Y position from top of frame (0.0 = top, 1.0 = bottom).
# L1 is the UPPER line, L2 is the LOWER line.
#
# Lane 1 (left / downward traffic):  car centroid crosses L1 → L2 (top→bottom)
# Lane 2 (right / upward traffic):   car centroid crosses L2 → L1 (bottom→top)
#
# For the highway footage in this project (overhead bridge view):
#   Left carriageway  flows downward in frame  → counted at L2 crossing
#   Right carriageway flows upward in frame    → counted at L1 crossing
#
# Tune these to sit just above/below the busiest traffic band.
LINE1_Y_FRAC = 0.35   # upper counting line — 35 % from top
LINE2_Y_FRAC = 0.65   # lower counting line — 65 % from top

# ── Output ────────────────────────────────────────────────────
OUTPUT_DIR   = "output"
OUTPUT_CODEC = "avc1"
SHOW_PREVIEW = True