"""
Configuration for the Traffic Enforcement System.
Adjust these values to match your camera setup and intersection layout.

Camera URL is now sourced from settings.json (editable from the GUI).
"""

import settings as _settings

# --- Camera Settings ---
# Effective source: a saved webcam index/URL (settings.json camera_source) or,
# if none is set, the phone IP stream. Each machine remembers its own camera.
CAMERA_SOURCE = _settings.effective_camera_source()
ANPR_CAMERA_SOURCE = 1  # Laptop webcam for plate reading
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720


def default_rotation_for(source):
    """Webcams are upright (no rotation); the phone IP stream is mounted
    sideways and needs a 90° clockwise rotation."""
    if isinstance(source, int) or (isinstance(source, str) and source.isdigit()):
        return None
    return "cw"


# --- Frame Orientation ---
# Rotation applied to every captured frame BEFORE detection/calibration, so the
# calibrated zones and the detector always see the same orientation.
# Options: None (no rotation), "cw" (90° clockwise), "ccw" (90° counter-clockwise), "180".
# Derived from the camera source: webcam index -> None, phone URL -> "cw".
# The GUI camera selector updates this when you switch sources.
CAMERA_ROTATION = default_rotation_for(CAMERA_SOURCE)
TARGET_FRAME_HEIGHT = 500  # every frame is resized to this height (width scaled to match)

# --- YOLOv8 Settings ---
YOLO_MODEL = "toy_cars.pt"  # custom toy-car detector (trained via dataset_tools); was yolov8n.pt
VEHICLE_CLASSES = [0]  # custom model has ONE class: 0 = car (stock YOLO used [2,3,5,7])
CONFIDENCE_THRESHOLD = 0.15
FRAME_SKIP = 1  # 1 = run YOLO on every new frame the capture thread delivers

# --- Traffic Light Detection (HSV ranges) ---
# Define the Region of Interest where the traffic light is in the frame
# Format: (x, y, width, height) in pixels — adjust to your camera view
TRAFFIC_LIGHT_ROI = (580, 50, 120, 200)

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
STOP_LINE_Y = 272
STOP_LINE_POINTS = [(11, 263), (829, 282)]

# Red-light line: a second line deeper into the intersection (past the stop line).
# Crossing this on red = the vehicle actually ran the light (logged as red_light, separate
# from a stop_line encroachment). None = no red-light line calibrated yet.
RED_LIGHT_LINE_POINTS = [(5, 357), (859, 360)]

# Lane boundaries: list of x-coordinates defining lane dividers (left to right)
# Vehicles crossing these while moving = illegal lane change
LANE_BOUNDARIES = []
LANE_LINES = [[(8, 241), (111, 58)], [(416, 264), (407, 55)], [(827, 266), (704, 50)]]

# Number of frames a vehicle must be tracked crossing a lane boundary to trigger violation
LANE_CHANGE_FRAMES = 5

# --- ESP32 Serial Settings (sourced from settings.json, editable from GUI) ---
SERIAL_PORT = _settings.load().get("serial_port", "COM5")
SERIAL_BAUD = int(_settings.load().get("serial_baud", 9600))

# --- ANPR Settings ---
ANPR_MODEL = "yolov8n.pt"  # Model for plate detection (will be replaced with plate-specific model)
PLATE_CONFIDENCE = 0.3
OCR_LANGUAGES = ["en"]

# --- Violation Logging ---
VIOLATIONS_DIR = "data/violations"
VIOLATIONS_CSV = "data/violations/violations.csv"

# --- Violation API ---
VIOLATIONS_API_URL = "http://localhost:8001/api/violations"  # POST endpoint, set to None to disable
VIOLATIONS_API_TOKEN = "4fc06642eaa3d24b8188de4a3f74027f04eda821ad5f0956b9d0d487a627724e"  # Bearer token, or None for no auth
VIOLATIONS_API_TIMEOUT = 5  # seconds
