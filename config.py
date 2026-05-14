"""
Configuration for the Traffic Enforcement System.
Adjust these values to match your camera setup and intersection layout.

Camera URL is now sourced from settings.json (editable from the GUI).
"""

import settings as _settings

# --- Camera Settings ---
CAMERA_SOURCE = _settings.camera_source()  # Phone camera (IP Webcam app)
ANPR_CAMERA_SOURCE = 0  # Laptop webcam for plate reading
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# --- YOLOv8 Settings ---
YOLO_MODEL = "yolov8n.pt"  # nano model for speed
VEHICLE_CLASSES = [2, 3, 5, 7]  # COCO classes: car, motorcycle, bus, truck
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
STOP_LINE_Y = 201
STOP_LINE_POINTS = [(3, 261), (123, 141)]

# Lane boundaries: list of x-coordinates defining lane dividers (left to right)
# Vehicles crossing these while moving = illegal lane change
LANE_BOUNDARIES = []
LANE_LINES = [[(3, 269), (278, 353)], [(45, 221), (276, 281)], [(93, 179), (273, 227)]]

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
VIOLATIONS_API_TOKEN = None  # Bearer token, or None for no auth
VIOLATIONS_API_TIMEOUT = 5  # seconds
