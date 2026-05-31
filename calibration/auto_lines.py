"""
Detect stop line (crosswalk / white bar) and lane markings on a model road board.

Tuned for boards with:
  - A horizontal zebra crossing or white stop bar (lower part of frame)
  - White dashed lane lines along the road (perspective, roughly parallel)
"""

from __future__ import annotations

import math

import cv2
import numpy as np


def _length(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])


def _angle_deg(p1, p2):
    return math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0])) % 180


def _midpoint(p1, p2):
    return ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)


def _extend_to_frame(p1, p2, w, h, margin=8):
    x1, y1 = p1
    x2, y2 = p2
    if x1 == x2 and y1 == y2:
        return p1, p2
    pts = []
    for edge, fixed, var_expr in (
        (0, 0, lambda t: int(x1 + t * (x2 - x1))),
        (1, h - 1, lambda t: int(x1 + t * (x2 - x1))),
        (2, 0, lambda t: int(y1 + t * (y2 - y1))),
        (3, w - 1, lambda t: int(y1 + t * (y2 - y1))),
    ):
        if edge < 2:
            if y2 == y1:
                continue
            t = (fixed - y1) / (y2 - y1)
            if 0 <= t <= 1:
                x = var_expr(t)
                if margin <= x < w - margin:
                    pts.append((x, fixed))
        else:
            if x2 == x1:
                continue
            t = (fixed - x1) / (x2 - x1)
            if 0 <= t <= 1:
                y = var_expr(t)
                if margin <= y < h - margin:
                    pts.append((fixed, y))
    if len(pts) >= 2:
        pts.sort(key=lambda p: _length(p1, p))
        return pts[0], pts[-1]
    return p1, p2


def _white_mask(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    bright = cv2.inRange(hsv, np.array([0, 0, 140]), np.array([180, 80, 255]))
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    white = cv2.bitwise_or(bright, otsu)
    white = cv2.morphologyEx(
        white, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1
    )
    return white


def _detect_crosswalk_stop_line(white, h, w):
    """
  Find the zebra / stop bar: horizontal white band in the lower half of the frame.
    """
    y_start = int(h * 0.30)
    roi = white[y_start:h, :].copy()
    if roi.size == 0:
        return None

    # Connect zebra stripes horizontally
    kh = cv2.getStructuringElement(cv2.MORPH_RECT, (max(21, w // 12), 5))
    closed = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, kh)

    row_fill = np.sum(closed > 0, axis=1) / max(w, 1)
    # Prefer lower rows (closer to camera / near crosswalk)
    weights = np.linspace(0.4, 1.0, len(row_fill))
    scored_rows = row_fill * weights

    if scored_rows.max() < 0.06:
        return None

    cy_local = int(np.argmax(scored_rows))
    cy = y_start + cy_local

    # Fit a line through white pixels in a band around cy (handles slight tilt)
    y_lo = max(0, cy_local - 12)
    y_hi = min(roi.shape[0], cy_local + 13)
    band = closed[y_lo:y_hi, :]
    ys, xs = np.where(band > 0)
    if len(xs) < 20:
        x1, x2 = int(w * 0.08), int(w * 0.92)
        return [[x1, cy], [x2, cy]]

    ys = ys + y_lo + y_start
    pts = np.column_stack([xs.astype(np.float32), ys.astype(np.float32)])
    vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).flatten()

    if abs(float(vx)) < 1e-3:
        return [[int(w * 0.08), cy], [int(w * 0.92), cy]]

    # Endpoints at left/right margins
    def y_at(x):
        return int(y0 + (x - x0) * (vy / vx))

    x1, x2 = int(w * 0.06), int(w * 0.94)
    p1 = [x1, y_at(x1)]
    p2 = [x2, y_at(x2)]
    p1[1] = int(np.clip(p1[1], 0, h - 1))
    p2[1] = int(np.clip(p2[1], 0, h - 1))
    return [p1, p2]


def _segments_from_hough(mask, min_length, roi_rect=None):
    x0, y0, rw, rh = roi_rect or (0, 0, mask.shape[1], mask.shape[0])
    crop = mask[y0 : y0 + rh, x0 : x0 + rw]
    lines = cv2.HoughLinesP(
        crop, 1, np.pi / 180, threshold=35,
        minLineLength=min_length, maxLineGap=12,
    )
    if lines is None:
        return []
    out = []
    for seg in lines:
        x1, y1, x2, y2 = seg[0]
        p1 = (int(x1 + x0), int(y1 + y0))
        p2 = (int(x2 + x0), int(y2 + y0))
        if _length(p1, p2) >= min_length:
            out.append((p1, p2))
    return out


def _mask_line_score(p1, p2, mask, samples=20):
    hits = 0
    for t in np.linspace(0, 1, samples):
        x = int(p1[0] + t * (p2[0] - p1[0]))
        y = int(p1[1] + t * (p2[1] - p1[1]))
        if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and mask[y, x] > 0:
            hits += 1
    return hits / samples


def _angle_diff(a, b):
    d = abs(a - b) % 180
    return min(d, 180 - d)


def _detect_lane_lines(white, h, w, stop_line, max_lanes=3):
    """
    Lane markings: long white segments in the road area, roughly parallel,
    not horizontal (crosswalk) and not vertical (poles).
    """
    diag = math.hypot(w, h)
    min_len = max(40, int(diag * 0.18))

    mx0, mx1 = int(w * 0.12), int(w * 0.88)
    my0, my1 = int(h * 0.06), int(h * 0.92)
    roi = (mx0, my0, mx1 - mx0, my1 - my0)

    dil = cv2.dilate(white, np.ones((3, 3), np.uint8), iterations=1)
    segments = _segments_from_hough(dil, min_len, roi)

    stop_ang = None
    stop_y = h // 2
    if stop_line and len(stop_line) == 2:
        stop_ang = _angle_deg(tuple(stop_line[0]), tuple(stop_line[1]))
        stop_y = (stop_line[0][1] + stop_line[1][1]) // 2

    candidates = []
    for p1, p2 in segments:
        ang = _angle_deg(p1, p2)
        mid = _midpoint(p1, p2)
        L = _length(p1, p2)
        white_s = _mask_line_score(p1, p2, white)

        # Reject horizontal (crosswalk) and true vertical (poles)
        if ang < 15 or ang > 165:
            continue
        if 87 < ang < 93 and L < diag * 0.45:
            continue
        if stop_ang is not None and _angle_diff(ang, stop_ang) < 12:
            continue
        y_min, y_max = min(p1[1], p2[1]), max(p1[1], p2[1])
        # Lane must reach toward the crosswalk (not only far background)
        if y_max < stop_y - int(h * 0.22):
            continue
        if y_min > stop_y + int(h * 0.04):
            continue
        if y_max - y_min < int(h * 0.12):
            continue
        # Ignore top strip (table / background edges)
        if y_max < h * 0.15:
            continue

        if white_s < 0.15:
            continue

        candidates.append({
            "p1": p1, "p2": p2, "length": L, "angle": ang,
            "mid": mid, "white_s": white_s,
            "score": L * (1.0 + 2.0 * white_s),
        })

    if not candidates:
        return []

    # Cluster by similar angle, keep longest per cluster
    candidates.sort(key=lambda c: c["score"], reverse=True)
    clusters = []
    for c in candidates:
        placed = False
        for cluster in clusters:
            if _angle_diff(c["angle"], cluster["angle"]) < 14:
                cluster["items"].append(c)
                placed = True
                break
        if not placed:
            clusters.append({"angle": c["angle"], "items": [c]})

    lane_lines = []
    clusters.sort(key=lambda cl: max(i["score"] for i in cl["items"]), reverse=True)
    for cluster in clusters[:max_lanes]:
        best = max(cluster["items"], key=lambda c: c["score"])
        a, b = _extend_to_frame(best["p1"], best["p2"], w, h)
        lane_lines.append([list(a), list(b)])

    return lane_lines


def detect_board_lines(frame, max_lane_lines=3):
    """
    Return (stop_line_points, lane_lines, debug_info).
    """
    h, w = frame.shape[:2]
    white = _white_mask(frame)

    stop_line = _detect_crosswalk_stop_line(white, h, w)
    method_stop = "crosswalk"

    if stop_line is None:
        # Fallback: longest near-horizontal white line in lower 50%
        segments = _segments_from_hough(white, max(30, w // 8))
        best, best_s = None, 0
        for p1, p2 in segments:
            ang = _angle_deg(p1, p2)
            if ang > 25 and ang < 155:
                continue
            my = _midpoint(p1, p2)[1]
            if my < h * 0.35:
                continue
            s = _length(p1, p2) * _mask_line_score(p1, p2, white)
            if s > best_s:
                best_s = s
                best = p1, p2
        if best:
            a, b = _extend_to_frame(best[0], best[1], w, h)
            stop_line = [list(a), list(b)]
            method_stop = "hough_horizontal"
        else:
            return None, [], {
                "message": "No crosswalk/stop line found — show the white crossing clearly.",
            }

    lane_lines = _detect_lane_lines(white, h, w, stop_line, max_lane_lines)

    return stop_line, lane_lines, {
        "stop_method": method_stop,
        "lane_count": len(lane_lines),
        "frame_size": (w, h),
    }


def draw_detected(frame, stop_line, lane_lines):
    out = frame.copy()
    if stop_line and len(stop_line) == 2:
        cv2.line(out, tuple(stop_line[0]), tuple(stop_line[1]), (0, 0, 255), 3)
        cv2.putText(
            out, "STOP / CROSSWALK", tuple(stop_line[0]),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2,
        )
    for i, lane in enumerate(lane_lines):
        cv2.line(out, tuple(lane[0]), tuple(lane[1]), (0, 255, 255), 2)
        cv2.putText(
            out, f"LANE {i + 1}", tuple(lane[0]),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2,
        )
    return out
