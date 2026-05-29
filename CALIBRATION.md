# Calibration Guide

The traffic-enforcement system uses three calibrated zones, all stored as
plain values in **`config.py`** and written by **`calibrate.py`**:

| Value (`config.py`) | Line | Purpose |
|---------------------|------|---------|
| `TRAFFIC_LIGHT_ROI` | 25 | Where the system *looks* for the traffic light (not a violation zone) |
| `STOP_LINE_Y` / `STOP_LINE_POINTS` | 43–44 | Stop line — crossing it on red is a violation |
| `LANE_BOUNDARIES` / `LANE_LINES` | 48–49 | Lane dividers — crossing while moving is an illegal lane change |

---

## Recalibrate (draw fresh zones)

1. Make sure the phone camera is streaming at `192.168.60.208:8080/video`
   (the calibrator reads `config.CAMERA_SOURCE`).
2. Run the tool:
   ```powershell
   python calibrate.py
   ```
3. In the window, use these keys/mouse actions:

   | Key | Action |
   |-----|--------|
   | **C** | Clear the current overlay (start fresh) |
   | **2** | Stop-line mode → click **2 points** to draw the line |
   | **3** | Lane-line mode → click **2 points per lane line** (can be diagonal) |
   | **Z** | Undo the last lane line |
   | **1** | Light-ROI mode → **drag a rectangle** around the traffic light |
   | **S** | **Save** — writes the new values into `config.py` |
   | **Q** / **ESC** | Quit |

4. After saving, the new values are picked up automatically the next time you
   run `main.py` or the GUI.

---

## Delete the existing calibrations

> ⚠️ **`C` + `S` in `calibrate.py` does NOT fully wipe.** `save_config()` only
> strips the `STOP_LINE_POINTS` / `LANE_LINES` lines and leaves `STOP_LINE_Y`,
> `LANE_BOUNDARIES`, and `TRAFFIC_LIGHT_ROI` at their old values. `S` is reliable
> for *saving newly drawn* zones, but not for clearing them.

For a guaranteed clean delete, edit `config.py` by hand:

| Line | Change to |
|------|-----------|
| 43 `STOP_LINE_Y = 201` | `STOP_LINE_Y = -1` |
| 44 `STOP_LINE_POINTS = [(3, 261), (123, 141)]` | `STOP_LINE_POINTS = None` |
| 48 `LANE_BOUNDARIES = []` | leave as is (already empty) |
| 49 `LANE_LINES = [[(3, 269), ...], ...]` | `LANE_LINES = []` |

This disables stop-line and lane-change enforcement until you recalibrate.
Leave `TRAFFIC_LIGHT_ROI` (line 25) alone — it only tells the system *where to
look* for the light, not what counts as a violation.

The detector reads these values via `getattr` with safe fallbacks, so an
off-frame `STOP_LINE_Y` + `STOP_LINE_POINTS = None` means no stop-line checks,
and empty `LANE_LINES` / `LANE_BOUNDARIES` means no lane-change checks.

---

## Typical workflow

1. **Delete** old zones (edit `config.py` as above) — optional, mainly when the
   camera angle changed.
2. **Recalibrate** with `python calibrate.py` → draw → **S** to save.
3. **Run** `python main.py` (or the GUI) — it loads the updated `config.py`.
