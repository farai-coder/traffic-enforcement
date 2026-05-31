"""
Configuration for the Traffic Enforcement System.
Adjust these values to match your camera setup and intersection layout.

Camera URL is now sourced from settings.json (editable from the GUI).
"""

import settings as _settings

# --- Camera Settings ---
# Phone/iPad IP stream: USE_PHONE_STREAM True (uses settings.json IP).
# External USB webcam: USE_PHONE_STREAM False and set CAMERA_USB_INDEX (try 0 or 1).
USE_PHONE_STREAM = False
CAMERA_USB_INDEX = 1  # external USB cam; use 0 if the built-in laptop cam opens instead
CAMERA_SOURCE = (
    _settings.camera_source() if USE_PHONE_STREAM else CAMERA_USB_INDEX
)
ANPR_CAMERA_SOURCE = CAMERA_USB_INDEX
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
# Must match between calibrate.py and capture/camera.py (main pipeline).
# False for a normal upright external webcam; True only if the image is sideways.
CAMERA_ROTATE_90 = False
FRAME_OUTPUT_HEIGHT = 500

# --- YOLOv8 Settings ---
YOLO_MODEL = "yolov8n.pt"  # nano model for speed
VEHICLE_CLASSES = [2, 3, 5, 7]  # COCO classes: car, motorcycle, bus, truck
CONFIDENCE_THRESHOLD = 0.10  # lower helps small toy cars; raise if too many false positives
YOLO_IMGSZ = 960  # larger = better on small / front-facing toys (slower)
# Toy board mode: also accept non-vehicle YOLO boxes + motion inside the road area.
TOY_MODE = True
TOY_CONF = 0.06
TOY_MOTION_ASSIST = True
TOY_MIN_BOX_AREA = 350  # pixels² — ignore tiny specks
TOY_EXCLUDE_CLASSES = [0, 14, 15, 16, 56, 57, 60]  # person, bird, cat, dog, chair, couch, table
FRAME_SKIP = 1  # legacy alias
# Run YOLO/detection every N display frames (1=every frame, 2=half rate, faster preview).
DETECTION_FRAME_SKIP = 2
# Live preview uses async camera thread (latest frame only — avoids buffer lag).
USE_ASYNC_CAMERA = True
# Stable ID across frames (fixes one toy car getting many IDs when moving).
TRACK_IOU_MATCH = 0.25
TRACK_MAX_LOST_FRAMES = 25
TRACK_MERGE_IOU = 0.50  # merge duplicate boxes (YOLO + motion) before tracking

# --- Traffic Light Detection (HSV ranges) ---
# Define the Region of Interest where the traffic light is in the frame
# Format: (x, y, width, height) in pixels — adjust to your camera view
TRAFFIC_LIGHT_ROI = [580, 50, 120, 200]

# HSV color ranges for traffic light detection
RED_HSV_LOWER_1 = (0, 100, 100)
RED_HSV_UPPER_1 = (10, 255, 255)
RED_HSV_LOWER_2 = (160, 100, 100)
RED_HSV_UPPER_2 = (180, 255, 255)

YELLOW_HSV_LOWER = (15, 100, 100)
YELLOW_HSV_UPPER = (35, 255, 255)

GREEN_HSV_LOWER = (40, 100, 100)
GREEN_HSV_UPPER = (90, 255, 255)

MIN_LIGHT_AREA = 50  # Minimum contour area to count as a detected light (low for toy setup)

# --- Violation Zones ---
# Stop line: y-coordinate in the frame. Vehicles below this line when light is red = violation.
STOP_LINE_Y = 281
STOP_LINE_POINTS = [(111, 277), (616, 286)]

# Lane boundaries: list of x-coordinates defining lane dividers (left to right)
# Vehicles crossing these while moving = illegal lane change
LANE_BOUNDARIES = []
LANE_LINES = [[[109, 277], [191, 55]], [[361, 283], [372, 55]], [[615, 287], [560, 60]], [(109, 275), (191, 55)], [(361, 281), (371, 63)], [(614, 286), (564, 70)]]

# Number of frames a vehicle must be tracked crossing a lane boundary to trigger violation
LANE_CHANGE_FRAMES = 3  # fewer frames needed on toy board for lane_change to register

# --- Intersection / traffic-light pairs (4 lights, 2 synced pairs) ---
# Which approach your camera and stop line monitor:
#   "east_west" = TL1 + TL3 (paired)  |  "north_south" = TL2 + TL4 (paired)
CAMERA_APPROACH = "east_west"
SERIAL_PORT = _settings.load().get("serial_port", "COM5")
SERIAL_BAUD = int(_settings.load().get("serial_baud", 9600))

# --- ANPR Settings ---
ANPR_MODEL = "yolov8n.pt"  # Model for plate detection (will be replaced with plate-specific model)
PLATE_CONFIDENCE = 0.3
OCR_LANGUAGES = ["en"]
TROCR_MODEL = "microsoft/trocr-small-printed"  # lighter; pre-download when online once
OCR_ENABLED = True  # set False to skip TrOCR entirely (violations still work)

# --- Violation Logging ---
VIOLATIONS_DIR = "data/violations"
VIOLATIONS_CSV = "data/violations/violations.csv"
# One video (+ API) per vehicle within this window (seconds).
VIOLATION_COOLDOWN_SECONDS = 60
# Extra photos while still in violation (first photo is always on the offense frame).
VIOLATION_PHOTO_INTERVAL_SECONDS = 0.5
# Short evidence clip for motorists (pre + post seconds around the violation).
VIOLATION_VIDEO_PRE_SECONDS = 1.0
VIOLATION_VIDEO_POST_SECONDS = 0.5
VIOLATION_SAVE_VIDEO = True
VIOLATION_POST_API = True  # POST once per incident to VIOLATIONS_API_URL
# Do not save photos/video/API unless OCR returns a valid plate (AAA1234 format).
VIOLATION_REQUIRE_VALID_PLATE = True

# --- Violation API ---
VIOLATIONS_API_URL = "http://localhost:8000/api/violations"  # POST endpoint, set to None to disable
VIOLATIONS_API_TOKEN = "4fc06642eaa3d24b8188de4a3f74027f04eda821ad5f0956b9d0d487a627724e"  # Bearer token, or None for no auth
VIOLATIONS_API_TIMEOUT = 5  # seconds
