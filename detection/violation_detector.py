import cv2
from ultralytics import YOLO
import config


def _point_side_of_line(point, line_start, line_end):
    """Return the cross product sign to determine which side of a line a point is on.
    Positive = left side, negative = right side, 0 = on the line."""
    return ((line_end[0] - line_start[0]) * (point[1] - line_start[1]) -
            (line_end[1] - line_start[1]) * (point[0] - line_start[0]))


def _crossed_line(p1, p2, line_start, line_end):
    """Check if the movement from p1 to p2 crosses the line defined by line_start→line_end."""
    side1 = _point_side_of_line(p1, line_start, line_end)
    side2 = _point_side_of_line(p2, line_start, line_end)
    return (side1 > 0 and side2 < 0) or (side1 < 0 and side2 > 0)


class ViolationDetector:
    """Detects traffic violations using YOLOv8 vehicle detection + zone logic."""

    def __init__(self):
        self.model = YOLO(config.YOLO_MODEL)
        self.vehicle_classes = config.VEHICLE_CLASSES
        self.stop_line_y = config.STOP_LINE_Y
        # Support diagonal lines: list of [(x1,y1),(x2,y2)] pairs
        self.lane_lines = getattr(config, "LANE_LINES", [])
        # Support diagonal stop line
        self.stop_line_points = getattr(config, "STOP_LINE_POINTS", None)
        # Deeper red-light line (past the stop line) — crossing it on red = ran the light
        self.red_light_line_points = getattr(config, "RED_LIGHT_LINE_POINTS", None)
        # Fallback to old vertical-only lane boundaries
        self.lane_boundaries = config.LANE_BOUNDARIES
        # Track vehicles: {track_id: {"positions": [(cx, cy), ...], "violated": set()}}
        self.tracked_vehicles = {}

    def detect_vehicles(self, frame):
        """Run YOLOv8 on frame, return list of vehicle detections."""
        results = self.model.track(frame, persist=True, conf=config.CONFIDENCE_THRESHOLD,
                                   classes=self.vehicle_classes, verbose=False)

        detections = []
        if results[0].boxes is None or results[0].boxes.id is None:
            return detections

        boxes = results[0].boxes
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes.xyxy[i].tolist()
            cls = int(boxes.cls[i])
            conf = float(boxes.conf[i])
            track_id = int(boxes.id[i])

            detections.append({
                "bbox": (int(x1), int(y1), int(x2), int(y2)),
                "class": cls,
                "conf": conf,
                "track_id": track_id,
            })

        return detections

    def check_violations(self, detections, light_state):
        """Check each detected vehicle for violations based on light state and zones."""
        violations = []

        for det in detections:
            track_id = det["track_id"]
            x1, y1, x2, y2 = det["bbox"]
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            front_y = y2

            if track_id not in self.tracked_vehicles:
                self.tracked_vehicles[track_id] = {"positions": [], "violated": set()}

            tracker = self.tracked_vehicles[track_id]
            tracker["positions"].append((cx, cy, front_y))
            if len(tracker["positions"]) > 30:
                tracker["positions"] = tracker["positions"][-30:]

            # --- Stop-line & red-light violations (both enforced on red, on different lines) ---
            # Stop line (near): crossing it on red = stop_line (encroached over the line).
            # Red-light line (deeper): crossing it on red = red_light (actually ran the light).
            # A vehicle that blows through crosses both lines and triggers both violations.
            if light_state == "red":
                rect = (x1, y1, x2 - x1, y2 - y1)

                # Stop line
                if self.stop_line_points:
                    crossed_stop, _, _ = cv2.clipLine(
                        rect,
                        tuple(self.stop_line_points[0]),
                        tuple(self.stop_line_points[1]),
                    )
                else:
                    crossed_stop = y1 <= self.stop_line_y <= y2

                if crossed_stop and "stop_line" not in tracker["violated"]:
                    tracker["violated"].add("stop_line")
                    violations.append({
                        "track_id": track_id,
                        "type": "stop_line",
                        "bbox": det["bbox"],
                    })

                # Red-light line (only if calibrated)
                if self.red_light_line_points:
                    crossed_red, _, _ = cv2.clipLine(
                        rect,
                        tuple(self.red_light_line_points[0]),
                        tuple(self.red_light_line_points[1]),
                    )
                    if crossed_red and "red_light" not in tracker["violated"]:
                        tracker["violated"].add("red_light")
                        violations.append({
                            "track_id": track_id,
                            "type": "red_light",
                            "bbox": det["bbox"],
                        })

            # --- Illegal Lane Change ---
            if len(tracker["positions"]) >= config.LANE_CHANGE_FRAMES:
                recent = tracker["positions"][-config.LANE_CHANGE_FRAMES:]
                start_pos = (recent[0][0], recent[0][1])
                end_pos = (recent[-1][0], recent[-1][1])

                # Check diagonal lane lines first
                for lane_line in self.lane_lines:
                    crossed = _crossed_line(
                        start_pos, end_pos,
                        tuple(lane_line[0]), tuple(lane_line[1])
                    )
                    if crossed and "lane_change" not in tracker["violated"]:
                        tracker["violated"].add("lane_change")
                        violations.append({
                            "track_id": track_id,
                            "type": "lane_change",
                            "bbox": det["bbox"],
                        })

                # Fallback: check old vertical boundaries
                if not self.lane_lines:
                    for boundary in self.lane_boundaries:
                        crossed = ((start_pos[0] < boundary <= end_pos[0]) or
                                   (end_pos[0] < boundary <= start_pos[0]))
                        if crossed and "lane_change" not in tracker["violated"]:
                            tracker["violated"].add("lane_change")
                            violations.append({
                                "track_id": track_id,
                                "type": "lane_change",
                                "bbox": det["bbox"],
                            })

        return violations

    def draw(self, frame, detections, violations, light_state=None):
        """Draw bounding boxes, stop line, lane lines, and violation labels."""
        h, w = frame.shape[:2]

        # Draw stop line (only if configured) — bright red when enforced, dim when not.
        if self.stop_line_points:
            enforced = light_state == "red"
            color = (0, 0, 255) if enforced else (90, 90, 90)
            thickness = 3 if enforced else 1
            cv2.line(frame, tuple(self.stop_line_points[0]), tuple(self.stop_line_points[1]),
                     color, thickness)
            if enforced:
                cv2.putText(frame, "STOP LINE", (self.stop_line_points[0][0], self.stop_line_points[0][1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Draw red-light line (deeper) — orange when enforced, dim when not.
        if self.red_light_line_points:
            enforced = light_state == "red"
            color = (0, 128, 255) if enforced else (90, 90, 90)
            thickness = 3 if enforced else 1
            cv2.line(frame, tuple(self.red_light_line_points[0]), tuple(self.red_light_line_points[1]),
                     color, thickness)
            if enforced:
                cv2.putText(frame, "RED LIGHT LINE",
                            (self.red_light_line_points[0][0], self.red_light_line_points[0][1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Draw diagonal lane lines
        for lane_line in self.lane_lines:
            cv2.line(frame, tuple(lane_line[0]), tuple(lane_line[1]), (255, 255, 0), 2)

        # Fallback: draw old vertical boundaries
        if not self.lane_lines:
            for lx in self.lane_boundaries:
                cv2.line(frame, (lx, 0), (lx, h), (255, 255, 0), 1)

        # Draw vehicle bounding boxes
        violation_ids = {v["track_id"] for v in violations}
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            is_violating = det["track_id"] in violation_ids
            color = (0, 0, 255) if is_violating else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"ID:{det['track_id']} {det['conf']:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Draw violation labels
        for v in violations:
            x1, y1, x2, y2 = v["bbox"]
            violation_text = v["type"].replace("_", " ").upper()
            cv2.putText(frame, f"VIOLATION: {violation_text}",
                        (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        return frame
