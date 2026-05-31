import cv2
import numpy as np
from ultralytics import YOLO
import config
from calibration.store import load as load_board_calibration
from detection.stable_tracker import StableTracker


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


def _bbox_crosses_stop_line(x1, y1, x2, y2, p1, p2, margin=10):
    """True if vehicle bbox touches or crosses the stop line segment."""
    p1, p2 = tuple(p1), tuple(p2)
    rect = (x1, y1, x2 - x1, y2 - y1)
    crossed, _, _ = cv2.clipLine(rect, p1, p2)
    if crossed:
        return True
    # Fallback for small toy boxes: bbox vertical span crosses the line band
    line_y = (p1[1] + p2[1]) / 2
    x_min, x_max = min(p1[0], p2[0]), max(p1[0], p2[0])
    y_lo, y_hi = line_y - margin, line_y + margin
    if y1 <= y_hi and y2 >= y_lo and x1 <= x_max and x2 >= x_min:
        return True
    return False


def _lane_x_at_y(line, y):
    (x1, y1), (x2, y2) = line
    if y2 == y1:
        return x1
    t = (y - y1) / (y2 - y1)
    return x1 + t * (x2 - x1)


def _in_road(cx, cy, lane_lines, stop_y, frame_h, frame_w):
    """Keep detections on the model road surface (between lane lines)."""
    if cy < 45 or cy > min(frame_h - 10, stop_y + 55):
        return False
    if lane_lines and len(lane_lines) >= 2:
        xs = sorted(_lane_x_at_y(line, cy) for line in lane_lines)
        return xs[0] + 12 <= cx <= xs[-1] - 12
    return int(frame_w * 0.08) < cx < int(frame_w * 0.92)


def _box_area(x1, y1, x2, y2):
    return max(0, x2 - x1) * max(0, y2 - y1)


def _iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    union = _box_area(ax1, ay1, ax2, ay2) + _box_area(bx1, by1, bx2, by2) - inter
    return inter / max(union, 1)


def _merge_detections(detections, iou_thresh=0.45):
    """Drop overlapping boxes, keeping higher confidence."""
    detections = sorted(detections, key=lambda d: d["conf"], reverse=True)
    kept = []
    for det in detections:
        if all(_iou(det["bbox"], k["bbox"]) < iou_thresh for k in kept):
            kept.append(det)
    return kept


class ViolationDetector:
    """Detects traffic violations using YOLOv8 vehicle detection + zone logic."""

    def __init__(self):
        self.model = YOLO(config.YOLO_MODEL)
        self.vehicle_classes = config.VEHICLE_CLASSES
        cal = load_board_calibration()
        self.stop_line_y = cal.get("stop_line_y", config.STOP_LINE_Y)
        self.stop_line_points = cal["stop_line_points"] if cal.get("stop_line_points") else getattr(
            config, "STOP_LINE_POINTS", None
        )
        self.lane_lines = cal["lane_lines"] if cal.get("lane_lines") else getattr(
            config, "LANE_LINES", []
        )
        self.lane_boundaries = config.LANE_BOUNDARIES
        self.toy_mode = getattr(config, "TOY_MODE", False)
        self._prev_gray = None
        if self.toy_mode:
            print("[INIT] Toy mode ON — broad detect + motion assist for front-facing toys",
                  flush=True)
        self._stable = StableTracker(
            iou_threshold=getattr(config, "TRACK_IOU_MATCH", 0.25),
            max_lost_frames=getattr(config, "TRACK_MAX_LOST_FRAMES", 25),
        )
        # Track vehicles: {track_id: {"positions": [(cx, cy), ...], "violated": set()}}
        self.tracked_vehicles = {}

    def _parse_yolo_boxes(self, boxes, frame_h, frame_w):
        out = []
        if boxes is None or len(boxes) == 0:
            return out
        min_area = getattr(config, "TOY_MIN_BOX_AREA", 350)
        exclude = set(getattr(config, "TOY_EXCLUDE_CLASSES", []))
        for i in range(len(boxes)):
            x1, y1, x2, y2 = [int(v) for v in boxes.xyxy[i].tolist()]
            cls = int(boxes.cls[i])
            conf = float(boxes.conf[i])
            if cls in exclude:
                continue
            if _box_area(x1, y1, x2, y2) < min_area:
                continue
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            if not _in_road(cx, cy, self.lane_lines, self.stop_line_y, frame_h, frame_w):
                continue
            out.append({
                "bbox": (x1, y1, x2, y2),
                "class": cls,
                "conf": conf,
            })
        return out

    def _motion_boxes(self, frame):
        """Moving blobs on the road — catches front-facing toys YOLO misses."""
        if not getattr(config, "TOY_MOTION_ASSIST", False):
            return []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        if self._prev_gray is None:
            self._prev_gray = gray
            return []
        diff = cv2.absdiff(gray, self._prev_gray)
        self._prev_gray = gray
        _, mask = cv2.threshold(diff, 18, 255, cv2.THRESH_BINARY)
        mask = cv2.dilate(mask, np.ones((7, 7), np.uint8), iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        h, w = frame.shape[:2]
        min_area = getattr(config, "TOY_MIN_BOX_AREA", 350)
        out = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area or area > h * w * 0.25:
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            x1, y1, x2, y2 = x, y, x + bw, y + bh
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            if not _in_road(cx, cy, self.lane_lines, self.stop_line_y, h, w):
                continue
            ar = bw / max(bh, 1)
            if ar < 0.15 or ar > 5.0:
                continue
            out.append({
                "bbox": (x1, y1, x2, y2),
                "class": -1,
                "conf": 0.5,
            })
        return out

    def detect_vehicles(self, frame):
        """Run YOLOv8 on frame, return list of vehicle detections."""
        h, w = frame.shape[:2]
        imgsz = getattr(config, "YOLO_IMGSZ", 640)
        conf = getattr(config, "TOY_CONF", config.CONFIDENCE_THRESHOLD) if self.toy_mode else config.CONFIDENCE_THRESHOLD

        results = self.model.track(
            frame, persist=True, conf=conf,
            classes=self.vehicle_classes, imgsz=imgsz, verbose=False,
        )
        detections = self._parse_yolo_boxes(results[0].boxes, h, w)

        if self.toy_mode:
            # Second pass: any object class on the road (front-facing toys often aren't "car")
            extra = self.model.predict(
                frame, conf=conf * 0.85, imgsz=imgsz, verbose=False,
            )
            for box_result in extra:
                detections.extend(self._parse_yolo_boxes(box_result.boxes, h, w))

            detections.extend(self._motion_boxes(frame))
            merge_iou = getattr(config, "TRACK_MERGE_IOU", 0.50)
            detections = _merge_detections(detections, iou_thresh=merge_iou)

        detections = self._stable.update(detections)
        self._prune_stale_tracks()
        return detections

    def reset_tracking(self):
        self.tracked_vehicles.clear()
        self._stable.clear()
        self._prev_gray = None

    def _prune_stale_tracks(self):
        """Drop violation history for vehicles that left the scene."""
        active = self._stable.active_ids()
        for tid in list(self.tracked_vehicles):
            if tid not in active:
                del self.tracked_vehicles[tid]

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

            # --- Red Light + Stop Line Violation (red only) ---
            if light_state == "red":
                crossed = False
                if self.stop_line_points:
                    crossed = _bbox_crosses_stop_line(
                        x1, y1, x2, y2,
                        self.stop_line_points[0], self.stop_line_points[1],
                    )
                else:
                    crossed = y1 <= self.stop_line_y <= y2

                if crossed:
                    if "red_light" not in tracker["violated"]:
                        tracker["violated"].add("red_light")
                        tracker["violated"].add("stop_line")
                        violations.append({
                            "track_id": track_id,
                            "type": "red_light",
                            "bbox": det["bbox"],
                            "conf": det.get("conf", 0.0),
                            "just_committed": True,
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
                            "conf": det.get("conf", 0.0),
                            "just_committed": True,
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
                                "conf": det.get("conf", 0.0),
                                "just_committed": True,
                            })

        return violations

    def _stop_line_crossed(self, x1, y1, x2, y2):
        if self.stop_line_points:
            return _bbox_crosses_stop_line(
                x1, y1, x2, y2,
                self.stop_line_points[0], self.stop_line_points[1],
            )
        return y1 <= self.stop_line_y <= y2

    def _lane_crossed_recently(self, tracker):
        if not self.lane_lines or len(tracker["positions"]) < config.LANE_CHANGE_FRAMES:
            return False
        recent = tracker["positions"][-config.LANE_CHANGE_FRAMES:]
        start_pos = (recent[0][0], recent[0][1])
        end_pos = (recent[-1][0], recent[-1][1])
        for lane_line in self.lane_lines:
            if _crossed_line(start_pos, end_pos, tuple(lane_line[0]), tuple(lane_line[1])):
                return True
        return False

    def committed_violation_types(self, track_id, det, light_state):
        """All violation types committed or active for this vehicle."""
        types = set()
        tracker = self.tracked_vehicles.get(track_id)
        if tracker:
            types.update(tracker.get("violated", set()))

        x1, y1, x2, y2 = det["bbox"]
        if light_state == "red" and self._stop_line_crossed(x1, y1, x2, y2):
            types.add("red_light")
            types.add("stop_line")

        if tracker and self._lane_crossed_recently(tracker):
            types.add("lane_change")

        return types

    def subjects_for_evidence(self, detections, light_state):
        """Vehicles currently in violation — used for repeated photo capture."""
        subjects = []
        for det in detections:
            track_id = det["track_id"]
            types = self.committed_violation_types(track_id, det, light_state)
            if not types:
                continue
            subjects.append({
                "track_id": track_id,
                "types": types,
                "bbox": det["bbox"],
                "conf": det.get("conf", 0.0),
            })
        return subjects

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
            if det.get("class") == -1:
                label = f"ID:{det['track_id']} motion"
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Draw violation labels
        for v in violations:
            x1, y1, x2, y2 = v["bbox"]
            violation_text = v["type"].replace("_", " ").upper()
            cv2.putText(frame, f"VIOLATION: {violation_text}",
                        (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        return frame
