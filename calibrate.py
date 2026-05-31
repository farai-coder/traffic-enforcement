"""
Calibration Tool for Traffic Enforcement System
================================================
Opens your webcam and lets you draw:
  1. Traffic Light ROI (drag a rectangle)
  2. Stop Line (click two points to draw a line)
  4. Red-Light Line (click two points - deeper than the stop line)
  3. Lane Boundaries (click two points per line - can be diagonal)

Controls:
    1       - Switch to Traffic Light ROI mode
    2       - Switch to Stop Line mode (click 2 points)
    4       - Switch to Red-Light Line mode (click 2 points)
    3       - Switch to Lane Boundary mode (click 2 points per line)
    Z       - Undo last lane boundary
    C       - Clear all calibration
    S       - Save calibration to config.py
    Q/ESC   - Quit
"""

import cv2
import numpy as np
import json
import config
from capture.camera import preprocess_frame

# State
mode = "stop_line"
mode_names = {
    "light_roi": "TRAFFIC LIGHT ROI",
    "stop_line": "STOP LINE",
    "red_light_line": "RED LIGHT LINE",
    "lane_boundary": "LANE BOUNDARIES",
}

# Calibration values
light_roi = None                # (x, y, w, h)
stop_line_points = []           # [(x1,y1), (x2,y2)]
red_light_line_points = []      # [(x1,y1), (x2,y2)] — deeper line, ran-the-light
lane_lines = []                 # [[(x1,y1),(x2,y2)], ...]

# Temp state for current line being drawn
current_points = []             # points being collected for current line
dragging = False
drag_start = None
drag_end = None


def mouse_callback(event, x, y, flags, param):
    global mode, light_roi, stop_line_points, red_light_line_points, lane_lines
    global current_points, dragging, drag_start, drag_end

    if mode == "light_roi":
        if event == cv2.EVENT_LBUTTONDOWN:
            dragging = True
            drag_start = (x, y)
            drag_end = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and dragging:
            drag_end = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and dragging:
            dragging = False
            drag_end = (x, y)
            x1 = min(drag_start[0], drag_end[0])
            y1 = min(drag_start[1], drag_end[1])
            x2 = max(drag_start[0], drag_end[0])
            y2 = max(drag_start[1], drag_end[1])
            if x2 - x1 > 5 and y2 - y1 > 5:
                light_roi = (x1, y1, x2 - x1, y2 - y1)
                print(f"[SET] Traffic Light ROI: {light_roi}")

    elif mode == "stop_line":
        if event == cv2.EVENT_LBUTTONDOWN:
            current_points.append((x, y))
            if len(current_points) == 1:
                print(f"[STOP LINE] First point: ({x}, {y}) - click second point")
            elif len(current_points) == 2:
                stop_line_points = list(current_points)
                print(f"[SET] Stop Line: {stop_line_points[0]} -> {stop_line_points[1]}")
                current_points.clear()

    elif mode == "red_light_line":
        if event == cv2.EVENT_LBUTTONDOWN:
            current_points.append((x, y))
            if len(current_points) == 1:
                print(f"[RED LIGHT LINE] First point: ({x}, {y}) - click second point")
            elif len(current_points) == 2:
                red_light_line_points = list(current_points)
                print(f"[SET] Red Light Line: {red_light_line_points[0]} -> {red_light_line_points[1]}")
                current_points.clear()

    elif mode == "lane_boundary":
        if event == cv2.EVENT_LBUTTONDOWN:
            current_points.append((x, y))
            if len(current_points) == 1:
                print(f"[LANE] First point: ({x}, {y}) - click second point")
            elif len(current_points) == 2:
                lane_lines.append(list(current_points))
                print(f"[SET] Lane line #{len(lane_lines)}: {current_points[0]} -> {current_points[1]}")
                current_points.clear()


def save_config():
    """Write calibration values back to config.py."""
    with open("config.py", "r") as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        if line.startswith("TRAFFIC_LIGHT_ROI") and light_roi is not None:
            new_lines.append(f"TRAFFIC_LIGHT_ROI = {light_roi}\n")
        elif line.startswith("STOP_LINE_Y") and stop_line_points:
            # Keep STOP_LINE_Y for backward compat but add the line points
            avg_y = (stop_line_points[0][1] + stop_line_points[1][1]) // 2
            new_lines.append(f"STOP_LINE_Y = {avg_y}\n")
        elif line.startswith("STOP_LINE_POINTS"):
            # Skip - we'll add it after STOP_LINE_Y
            continue
        elif line.startswith("RED_LIGHT_LINE_POINTS") and red_light_line_points:
            new_lines.append(f"RED_LIGHT_LINE_POINTS = {red_light_line_points}\n")
        elif line.startswith("LANE_BOUNDARIES") and lane_lines:
            # Replace old vertical-only format with new line format
            new_lines.append(f"LANE_BOUNDARIES = []\n")
        elif line.startswith("LANE_LINES"):
            # Skip - we'll add after LANE_BOUNDARIES
            continue
        else:
            new_lines.append(line)

    # Insert new config entries if they don't exist yet
    config_text = "".join(new_lines)
    if "STOP_LINE_POINTS" not in config_text and stop_line_points:
        config_text = config_text.replace(
            f"STOP_LINE_Y = {(stop_line_points[0][1] + stop_line_points[1][1]) // 2}\n",
            f"STOP_LINE_Y = {(stop_line_points[0][1] + stop_line_points[1][1]) // 2}\n"
            f"STOP_LINE_POINTS = {stop_line_points}\n"
        )
    if "LANE_LINES" not in config_text and lane_lines:
        config_text = config_text.replace(
            "LANE_BOUNDARIES = []\n",
            f"LANE_BOUNDARIES = []\nLANE_LINES = {lane_lines}\n"
        )
    if "RED_LIGHT_LINE_POINTS" not in config_text and red_light_line_points and stop_line_points:
        config_text = config_text.replace(
            f"STOP_LINE_POINTS = {stop_line_points}\n",
            f"STOP_LINE_POINTS = {stop_line_points}\n"
            f"RED_LIGHT_LINE_POINTS = {red_light_line_points}\n"
        )

    with open("config.py", "w") as f:
        f.write(config_text)

    print("\n[SAVED] Config updated! Values written to config.py")
    print(f"  Traffic Light ROI: {light_roi}")
    print(f"  Stop Line: {stop_line_points}")
    print(f"  Red Light Line: {red_light_line_points}")
    print(f"  Lane Lines: {lane_lines}")


def load_existing():
    """Preload zones already saved in config.py so they show on startup."""
    global light_roi, stop_line_points, red_light_line_points, lane_lines

    roi = getattr(config, "TRAFFIC_LIGHT_ROI", None)
    if roi:
        light_roi = tuple(roi)

    sl = getattr(config, "STOP_LINE_POINTS", None)
    if sl:
        stop_line_points = [tuple(p) for p in sl]

    rl = getattr(config, "RED_LIGHT_LINE_POINTS", None)
    if rl:
        red_light_line_points = [tuple(p) for p in rl]

    ll = getattr(config, "LANE_LINES", None)
    if ll:
        lane_lines = [[tuple(p) for p in lane] for lane in ll]


def main():
    global mode, light_roi, stop_line_points, red_light_line_points, lane_lines
    global current_points, dragging, drag_start, drag_end

    load_existing()

    cap = cv2.VideoCapture(config.CAMERA_SOURCE)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    if not cap.isOpened():
        print("[ERROR] Cannot open webcam!")
        return

    window_name = "Calibration Tool"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback)

    print("\n" + "=" * 60)
    print("  CALIBRATION TOOL")
    print("  Press 1=Light ROI | 2=Stop Line | 4=Red-Light Line | 3=Lane Lines")
    print("  Press Z=Undo last lane | C=Clear all | S=Save | Q=Quit")
    print("=" * 60)
    print(f"\nCurrent mode: {mode_names[mode]}")
    print("Click two points to draw the stop line.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = preprocess_frame(frame)
        display = frame.copy()
        h, w = display.shape[:2]

        # Draw traffic light ROI
        if light_roi is not None:
            rx, ry, rw, rh = light_roi
            cv2.rectangle(display, (rx, ry), (rx + rw, ry + rh), (0, 255, 255), 2)

        if dragging and drag_start and drag_end:
            cv2.rectangle(display, drag_start, drag_end, (0, 255, 255), 2)

        # Draw stop line
        if len(stop_line_points) == 2:
            cv2.line(display, stop_line_points[0], stop_line_points[1], (0, 0, 255), 3)

        # Draw red-light line (orange, to distinguish from the red stop line)
        if len(red_light_line_points) == 2:
            cv2.line(display, red_light_line_points[0], red_light_line_points[1], (0, 128, 255), 3)

        # Draw lane lines
        for lane in lane_lines:
            cv2.line(display, tuple(lane[0]), tuple(lane[1]), (255, 255, 0), 2)

        # Draw current partial clicks (first point placed, waiting for second)
        if len(current_points) == 1:
            cv2.circle(display, current_points[0], 6, (0, 255, 0), -1)
            cv2.putText(display, "Click 2nd point", (current_points[0][0] + 10, current_points[0][1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Draw mode indicator
        mode_text = f"MODE: {mode_names[mode]} | 1/2/4/3=switch | Z=undo lane | S=Save | Q=Quit"
        cv2.rectangle(display, (0, h - 40), (w, h), (0, 0, 0), -1)
        cv2.putText(display, mode_text, (10, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow(window_name, display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break
        elif key == ord("1"):
            mode = "light_roi"
            current_points.clear()
            print(f"\nMode: {mode_names[mode]} - Drag a rectangle around the traffic light")
        elif key == ord("2"):
            mode = "stop_line"
            current_points.clear()
            print(f"\nMode: {mode_names[mode]} - Click two points to draw the stop line")
        elif key == ord("4"):
            mode = "red_light_line"
            current_points.clear()
            print(f"\nMode: {mode_names[mode]} - Click two points to draw the red-light line (deeper than the stop line)")
        elif key == ord("3"):
            mode = "lane_boundary"
            current_points.clear()
            print(f"\nMode: {mode_names[mode]} - Click two points per lane line (can be diagonal)")
        elif key == ord("z"):
            if lane_lines:
                removed = lane_lines.pop()
                print(f"[UNDO] Removed last lane line: {removed}")
            else:
                print("[UNDO] No lane lines to remove")
        elif key == ord("c"):
            light_roi = None
            stop_line_points = []
            red_light_line_points = []
            lane_lines = []
            current_points.clear()
            print("\n[CLEARED] All calibration reset.")
        elif key == ord("s"):
            save_config()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
