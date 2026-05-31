"""
Automatic board calibration — detect stop line and lane markings from the camera.

Uses the same camera pipeline as main.py / calibrate.py.

Controls:
    SPACE   - Capture frame and auto-detect lines from the board
    S       - Save detected lines to config.py (after SPACE)
    R       - Re-run detection on last captured frame
    M       - Open manual calibrate.py instructions (run separately)
    Q/ESC   - Quit

Tips for best results:
    - Full board visible; zebra crossing / white stop bar in the lower part of view
    - Empty board (no cars); even lighting
    - Yellow overlay should sit ON the white crosswalk (red) and dashed lane lines
    - Keep camera fixed; verify before pressing S
"""

import sys
import time

import cv2
import config
from capture.camera import Camera
import calibrate
from calibration.auto_lines import detect_board_lines, draw_detected, _white_mask


def _average_frames(camera, n=8):
    frames = []
    for _ in range(n * 3):
      f = camera.read()
        if f is not None:
            frames.append(f.astype("float32"))
        if len(frames) >= n:
            break
        time.sleep(0.03)
    if not frames:
        return None
    return cv2.convertScaleAbs(sum(frames) / len(frames))


def _key_is(key, char):
    return key in (ord(char.lower()), ord(char.upper()))


def main():
    print("=" * 60)
    print("  AUTO CALIBRATION — detect lines on your board")
    print("=" * 60)
    print(f"  Camera: {config.CAMERA_SOURCE}")
    print("  SPACE = detect | S = save | R = retry | Q = quit\n")

    camera = Camera()
    try:
        camera.start()
    except RuntimeError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    window = "Auto Calibrate"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    frozen = None
    stop_line = None
    lane_lines = []
    info = {}

    while True:
        if frozen is not None and stop_line is not None:
            display = draw_detected(frozen, stop_line, lane_lines)
            status = (f"AUTO: stop + {len(lane_lines)} lane(s) | "
                      f"S=save R=retry SPACE=new Q=quit")
        else:
            frame = camera.read()
            if frame is None:
                continue
            display = frame.copy()
            status = "Point camera at board, then press SPACE to auto-detect"

        h, w = display.shape[:2]
        cv2.rectangle(display, (0, h - 36), (w, h), (0, 0, 0), -1)
        cv2.putText(display, status, (10, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        cv2.imshow(window, display)

        key = cv2.waitKey(1) & 0xFF
        if _key_is(key, "q") or key == 27:
            break
        if key == ord(" "):
            print("[INFO] Capturing frame...", flush=True)
            frozen = _average_frames(camera)
            if frozen is None:
                print("[WARN] No frame — check camera")
                continue
            stop_line, lane_lines, info = detect_board_lines(frozen)
            if stop_line is None:
                print(f"[WARN] {info.get('message', 'Detection failed')}")
            else:
                print(f"[OK] Stop line: {stop_line}")
                print(f"[OK] Lane lines ({len(lane_lines)}): {lane_lines}")
                print(f"[OK] Debug: {info}")
        elif _key_is(key, "r") and frozen is not None:
            stop_line, lane_lines, info = detect_board_lines(frozen)
            print(f"[RETRY] stop={stop_line} lanes={lane_lines} info={info}")
        elif _key_is(key, "d") and frozen is not None:
            fh, fw = frozen.shape[:2]
            cv2.imshow("White mask (debug)", cv2.resize(_white_mask(frozen), (fw // 2, fh // 2)))
        elif _key_is(key, "s"):
            if not stop_line:
                print("[WARN] Detect first with SPACE")
                continue
            calibrate.stop_line_points = stop_line
            calibrate.lane_lines = lane_lines
            calibrate.save_config()
            print("[DONE] Saved. Restart main.py.")

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
