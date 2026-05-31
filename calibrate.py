"""
Calibration Tool for Traffic Enforcement System
================================================
Opens your webcam and lets you draw:
  1. Traffic Light ROI (drag a rectangle)
  2. Stop Line (click two points to draw a line)
  3. Lane Boundaries (click two points per line - can be diagonal)

Controls:
    1       - Switch to Traffic Light ROI mode
    2       - Switch to Stop Line mode (click 2 points)
    3       - Switch to Lane Boundary mode (click 2 points per line)
    Z       - Undo last lane boundary
    C       - Clear all calibration
    A       - Auto-detect stop line + lanes from board markings (current frame)
    S       - Save calibration to config.py
    Q/ESC   - Quit
"""

import cv2
import os
import config
from capture.camera import Camera
from calibration.auto_lines import detect_board_lines, draw_detected
from calibration.store import load as load_board_calibration, save as save_board_calibration

# State
mode = "stop_line"
mode_names = {"light_roi": "TRAFFIC LIGHT ROI", "stop_line": "STOP LINE", "lane_boundary": "LANE BOUNDARIES"}

# Calibration values
light_roi = None                # (x, y, w, h)
stop_line_points = []           # [(x1,y1), (x2,y2)]
lane_lines = []                 # [[(x1,y1),(x2,y2)], ...]

# Temp state for current line being drawn
current_points = []             # points being collected for current line
dragging = False
drag_start = None
drag_end = None


def mouse_callback(event, x, y, flags, param):
    global mode, light_roi, stop_line_points, lane_lines
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
    """Write calibration to board_calibration.json (and sync config.py)."""
    global stop_line_points, lane_lines, light_roi

    if not stop_line_points or len(stop_line_points) != 2:
        print("[WARN] Draw the stop line first (mode 2, two clicks), then save.")
        return False

    save_board_calibration(stop_line_points, lane_lines, light_roi)

    cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.py")
    with open(cfg_path, "r", encoding="utf-8") as f:
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
            if stop_line_points:
                new_lines.append(f"STOP_LINE_POINTS = {stop_line_points}\n")
            continue
        elif line.startswith("LANE_BOUNDARIES"):
            new_lines.append("LANE_BOUNDARIES = []\n")
            continue
        elif line.startswith("LANE_LINES"):
            if lane_lines:
                new_lines.append(f"LANE_LINES = {lane_lines}\n")
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

    with open(cfg_path, "w", encoding="utf-8") as f:
        f.write(config_text)

    print("\n[SAVED] Calibration saved.")
    print(f"  Traffic Light ROI: {light_roi}")
    print(f"  Stop Line: {stop_line_points}")
    print(f"  Lane Lines: {lane_lines}")
    print("  Restart main.py so it reloads these coordinates.")
    return True


def _key_is(key, char):
    """Match key regardless of Caps Lock / Shift (OpenCV reports uppercase often)."""
    return key in (ord(char.lower()), ord(char.upper()))


def main():
    global mode, light_roi, stop_line_points, lane_lines
    global current_points, dragging, drag_start, drag_end

    saved = load_board_calibration()
    if saved.get("stop_line_points"):
        stop_line_points = saved["stop_line_points"]
    if saved.get("lane_lines"):
        lane_lines = saved["lane_lines"]
    if saved.get("light_roi"):
        light_roi = saved["light_roi"]

    print(f"[INFO] Camera source: {config.CAMERA_SOURCE}")
    print(f"[INFO] Rotate 90°: {getattr(config, 'CAMERA_ROTATE_90', True)}  "
          f"| Output height: {getattr(config, 'FRAME_OUTPUT_HEIGHT', 500)} px")
    camera = Camera()
    try:
        camera.start()
    except RuntimeError as e:
        print(f"[ERROR] {e}")
        return

    window_name = "Calibration Tool"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback)

    print("\n" + "=" * 60)
    print("  CALIBRATION TOOL")
    print("  Press 1=Light ROI | 2=Stop Line | 3=Lane Lines | A=Auto-detect")
    print("  Press Z=Undo last lane | C=Clear all | S=Save | Q=Quit")
    print("=" * 60)
    print(f"\nCurrent mode: {mode_names[mode]}")
    print("Click two points to draw the stop line.\n")

    saved_flash = 0  # frames to show on-screen "SAVED" banner

    while True:
        frame = camera.read()
        if frame is None:
            continue

        display = frame.copy()
        h, w = display.shape[:2]

        # Draw traffic light ROI
        if light_roi is not None:
            rx, ry, rw, rh = light_roi
            cv2.rectangle(display, (rx, ry), (rx + rw, ry + rh), (0, 255, 255), 2)
            cv2.putText(display, "LIGHT ROI", (rx, ry - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        if dragging and drag_start and drag_end:
            cv2.rectangle(display, drag_start, drag_end, (0, 255, 255), 2)

        # Draw stop line
        if len(stop_line_points) == 2:
            cv2.line(display, stop_line_points[0], stop_line_points[1], (0, 0, 255), 3)
            cv2.putText(display, "STOP LINE", (stop_line_points[0][0], stop_line_points[0][1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Draw lane lines
        for i, lane in enumerate(lane_lines):
            cv2.line(display, tuple(lane[0]), tuple(lane[1]), (255, 255, 0), 2)
            mid_x = (lane[0][0] + lane[1][0]) // 2
            mid_y = (lane[0][1] + lane[1][1]) // 2
            cv2.putText(display, f"Lane {i+1}", (mid_x + 5, mid_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        # Draw current partial clicks (first point placed, waiting for second)
        if len(current_points) == 1:
            cv2.circle(display, current_points[0], 6, (0, 255, 0), -1)
            cv2.putText(display, "Click 2nd point", (current_points[0][0] + 10, current_points[0][1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Draw mode indicator
        mode_text = (f"MODE: {mode_names[mode]} | A=auto | 1/2/3=switch | "
                     f"Z=undo | S=Save | Q=Quit")
        cv2.rectangle(display, (0, h - 40), (w, h), (0, 0, 0), -1)
        cv2.putText(display, mode_text, (10, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if saved_flash > 0:
            cv2.rectangle(display, (0, 0), (w, 52), (0, 120, 0), -1)
            cv2.putText(display, "SAVED — restart main.py", (10, 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            saved_flash -= 1

        cv2.imshow(window_name, display)

        key = cv2.waitKey(1) & 0xFF
        if _key_is(key, "q") or key == 27:
            break
        elif _key_is(key, "1"):
            mode = "light_roi"
            current_points.clear()
            print(f"\nMode: {mode_names[mode]} - Drag a rectangle around the traffic light")
        elif _key_is(key, "2"):
            mode = "stop_line"
            current_points.clear()
            print(f"\nMode: {mode_names[mode]} - Click two points to draw the stop line")
        elif _key_is(key, "3"):
            mode = "lane_boundary"
            current_points.clear()
            print(f"\nMode: {mode_names[mode]} - Click two points per lane line (can be diagonal)")
        elif _key_is(key, "z"):
            if lane_lines:
                removed = lane_lines.pop()
                print(f"[UNDO] Removed last lane line: {removed}")
            else:
                print("[UNDO] No lane lines to remove")
        elif _key_is(key, "c"):
            light_roi = None
            stop_line_points = []
            lane_lines = []
            current_points.clear()
            print("\n[CLEARED] All calibration reset.")
        elif _key_is(key, "a"):
            sl, lanes, info = detect_board_lines(frame)
            if sl is None:
                print(f"[AUTO] Failed: {info}")
            else:
                stop_line_points = sl
                lane_lines = lanes
                display = draw_detected(frame, sl, lanes)
                print(f"[AUTO] Stop: {sl}")
                print(f"[AUTO] Lanes ({len(lanes)}): {lanes}")
                cv2.imshow(window_name, display)
                cv2.waitKey(800)
        elif _key_is(key, "s"):
            if save_config():
                saved_flash = 60

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
