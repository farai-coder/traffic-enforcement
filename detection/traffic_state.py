import cv2
import numpy as np
import threading
import config


class TrafficLightDetector:
    """Detects traffic light state (red, yellow, green).

    Supports three modes:
    - serial: reads STATE:xxx messages from ESP32 over Serial (most reliable)
    - hsv: detects color using HSV masking from camera frame
    - manual: press R/Y/G keys to set state manually
    """

    def __init__(self, mode="manual"):
        self.roi = config.TRAFFIC_LIGHT_ROI
        self.current_state = "unknown"
        self.mode = mode
        self.manual_mode = mode == "manual"
        self._serial_conn = None
        self._serial_thread = None

        if mode == "serial":
            self._start_serial()

    def _start_serial(self):
        """Start a background thread to read light state from ESP32."""
        import serial
        try:
            self._serial_conn = serial.Serial(
                config.SERIAL_PORT, config.SERIAL_BAUD, timeout=1
            )
            self._serial_thread = threading.Thread(
                target=self._serial_reader, daemon=True
            )
            self._serial_thread.start()
            print(f"[SERIAL] Connected to ESP32 on {config.SERIAL_PORT}")
        except Exception as e:
            print(f"[SERIAL] Failed to connect to {config.SERIAL_PORT}: {e}")
            print("[SERIAL] Falling back to manual mode. Press R/Y/G to set state.")
            self.mode = "manual"
            self.manual_mode = True

    def _serial_reader(self):
        """Background thread: continuously reads STATE:xxx from ESP32."""
        while True:
            try:
                if self._serial_conn and self._serial_conn.in_waiting:
                    line = self._serial_conn.readline().decode("utf-8").strip()
                    if line.startswith("STATE:"):
                        state = line.split(":", 1)[1].lower()
                        if state in ("red", "yellow", "green"):
                            self.current_state = state
            except Exception:
                pass

    def set_state(self, state):
        """Manually set the traffic light state."""
        if state in ("red", "yellow", "green", "unknown"):
            self.current_state = state

    def detect(self, frame):
        """Detect the traffic light color in the frame.

        Returns one of: 'red', 'yellow', 'green', 'unknown'
        In serial mode, returns the latest state from ESP32.
        In manual mode, returns the manually set state.
        In hsv mode, detects from camera frame.
        """
        if self.mode in ("manual", "serial"):
            return self.current_state

        x, y, w, h = self.roi
        roi_frame = frame[y:y + h, x:x + w]

        if roi_frame.size == 0:
            return "unknown"

        hsv = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)

        # Detect red (two ranges because red wraps around in HSV)
        red_mask1 = cv2.inRange(hsv,
                                np.array(config.RED_HSV_LOWER_1),
                                np.array(config.RED_HSV_UPPER_1))
        red_mask2 = cv2.inRange(hsv,
                                np.array(config.RED_HSV_LOWER_2),
                                np.array(config.RED_HSV_UPPER_2))
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)

        # Detect yellow
        yellow_mask = cv2.inRange(hsv,
                                  np.array(config.YELLOW_HSV_LOWER),
                                  np.array(config.YELLOW_HSV_UPPER))

        # Detect green
        green_mask = cv2.inRange(hsv,
                                 np.array(config.GREEN_HSV_LOWER),
                                 np.array(config.GREEN_HSV_UPPER))

        # Count pixels for each color
        red_area = cv2.countNonZero(red_mask)
        yellow_area = cv2.countNonZero(yellow_mask)
        green_area = cv2.countNonZero(green_mask)

        areas = {"red": red_area, "yellow": yellow_area, "green": green_area}
        max_color = max(areas, key=areas.get)

        if areas[max_color] >= config.MIN_LIGHT_AREA:
            self.current_state = max_color
        else:
            self.current_state = "unknown"

        return self.current_state

    def draw(self, frame):
        """Draw the ROI rectangle and detected state on the frame."""
        x, y, w, h = self.roi
        color_map = {
            "red": (0, 0, 255),
            "yellow": (0, 255, 255),
            "green": (0, 255, 0),
            "unknown": (128, 128, 128),
        }
        color = color_map.get(self.current_state, (128, 128, 128))
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, f"Light: {self.current_state.upper()}",
                    (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        return frame
