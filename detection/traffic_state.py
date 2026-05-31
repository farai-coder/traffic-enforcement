import cv2
import numpy as np
import threading
import config

# Normalized approach keys (must match ESP32 ew/ns labels)
APPROACH_EW = "east_west"
APPROACH_NS = "north_south"
APPROACH_ALIASES = {
    "ew": APPROACH_EW,
    "east_west": APPROACH_EW,
    "east-west": APPROACH_EW,
    "e_w": APPROACH_EW,
    "ns": APPROACH_NS,
    "north_south": APPROACH_NS,
    "north-south": APPROACH_NS,
    "n_s": APPROACH_NS,
}


def normalize_approach(name: str) -> str:
    return APPROACH_ALIASES.get((name or "").lower().strip(), APPROACH_EW)


class TrafficLightDetector:
    """Traffic light state for a 4-light intersection (2 synchronized pairs).

    ESP32 sends: STATE:ew:green,ns:red
      - east_west  = TL1 + TL3
      - north_south = TL2 + TL4

    Violations use the state of CAMERA_APPROACH only (the road the camera watches).
    """

    def __init__(self, mode="manual"):
        self.roi = config.TRAFFIC_LIGHT_ROI
        self.monitored_approach = normalize_approach(
            getattr(config, "CAMERA_APPROACH", APPROACH_EW)
        )
        self.approach_states = {
            APPROACH_EW: "unknown",
            APPROACH_NS: "unknown",
        }
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
            print(
                f"[SERIAL] Monitoring approach: {self.monitored_approach} "
                f"(pairs: {APPROACH_EW}=TL1+TL3, {APPROACH_NS}=TL2+TL4)",
                flush=True,
            )
        except Exception as e:
            print(f"[SERIAL] Failed to connect to {config.SERIAL_PORT}: {e}")
            print("[SERIAL] Falling back to manual mode. Press R/Y/G to set state.")
            self.mode = "manual"
            self.manual_mode = True

    @staticmethod
    def _parse_serial_line(line: str) -> dict[str, str] | None:
        """Parse STATE:ew:green,ns:red or legacy STATE:green."""
        if not line.startswith("STATE:"):
            return None
        body = line[6:].strip().lower()
        # New format: ew:green,ns:red
        if ",ns:" in body or body.startswith("ew:"):
            out = {}
            for part in body.split(","):
                part = part.strip()
                if ":" not in part:
                    continue
                key, val = part.split(":", 1)
                key = normalize_approach(key)
                if key in (APPROACH_EW, APPROACH_NS) and val in (
                    "red", "yellow", "green",
                ):
                    out[key] = val
            return out if out else None
        # Legacy: STATE:green → apply to monitored approach only
        if body in ("red", "yellow", "green"):
            return {APPROACH_EW: body}
        return None

    def _serial_reader(self):
        """Background thread: reads paired STATE messages from ESP32."""
        while True:
            try:
                if self._serial_conn and self._serial_conn.in_waiting:
                    line = self._serial_conn.readline().decode("utf-8", errors="ignore").strip()
                    parsed = self._parse_serial_line(line)
                    if parsed:
                        self.approach_states.update(parsed)
                        self.current_state = self.approach_states.get(
                            self.monitored_approach, "unknown"
                        )
            except Exception:
                pass

    def set_state(self, state, approach=None):
        """Manually set light state for one pair (default: monitored approach)."""
        if state not in ("red", "yellow", "green", "unknown"):
            return
        key = normalize_approach(approach) if approach else self.monitored_approach
        self.approach_states[key] = state
        self.current_state = self.approach_states.get(
            self.monitored_approach, "unknown"
        )

    def get_monitored_state(self):
        return self.approach_states.get(self.monitored_approach, self.current_state)

    def get_pair_states(self):
        return dict(self.approach_states)

    def detect(self, frame):
        """Return light color for the monitored approach (camera / stop line)."""
        if self.mode in ("manual", "serial"):
            return self.get_monitored_state()

        x, y, w, h = self.roi
        roi_frame = frame[y:y + h, x:x + w]

        if roi_frame.size == 0:
            return "unknown"

        hsv = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)

        red_mask1 = cv2.inRange(hsv,
                                np.array(config.RED_HSV_LOWER_1),
                                np.array(config.RED_HSV_UPPER_1))
        red_mask2 = cv2.inRange(hsv,
                                np.array(config.RED_HSV_LOWER_2),
                                np.array(config.RED_HSV_UPPER_2))
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)

        yellow_mask = cv2.inRange(hsv,
                                  np.array(config.YELLOW_HSV_LOWER),
                                  np.array(config.YELLOW_HSV_UPPER))

        green_mask = cv2.inRange(hsv,
                                 np.array(config.GREEN_HSV_LOWER),
                                 np.array(config.GREEN_HSV_UPPER))

        red_area = cv2.countNonZero(red_mask)
        yellow_area = cv2.countNonZero(yellow_mask)
        green_area = cv2.countNonZero(green_mask)

        areas = {"red": red_area, "yellow": yellow_area, "green": green_area}
        max_color = max(areas, key=areas.get)

        if areas[max_color] >= config.MIN_LIGHT_AREA:
            self.approach_states[self.monitored_approach] = max_color
            self.current_state = max_color
        else:
            self.current_state = "unknown"

        return self.current_state

    def draw_status_overlay(self, frame):
        """Show both pairs + which one is used for enforcement."""
        h = frame.shape[0]
        ew = self.approach_states.get(APPROACH_EW, "?").upper()
        ns = self.approach_states.get(APPROACH_NS, "?").upper()
        mon = self.monitored_approach.replace("_", " ").upper()
        enforced = self.get_monitored_state().upper()

        cv2.putText(
            frame, f"EW(TL1+3):{ew}  NS(TL2+4):{ns}",
            (10, h - 58), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1,
        )
        cv2.putText(
            frame, f"Enforcing: {mon} = {enforced}",
            (10, h - 38), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2,
        )
        return frame

    def draw(self, frame):
        """Draw HSV ROI in camera mode; always show both pairs on overlay."""
        if self.mode == "hsv":
            x, y, w, h = self.roi
            color_map = {
                "red": (0, 0, 255),
                "yellow": (0, 255, 255),
                "green": (0, 255, 0),
                "unknown": (128, 128, 128),
            }
            color = color_map.get(self.get_monitored_state(), (128, 128, 128))
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"Light: {self.get_monitored_state().upper()}",
                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        return self.draw_status_overlay(frame)
