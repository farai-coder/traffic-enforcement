# traffic-enforcement

Real-time traffic violation detection and ANPR (automatic number plate recognition).

The pipeline captures video, detects vehicles with YOLOv8, tracks the traffic
light state (manual / serial / HSV), flags red-light, stop-line, and lane-change
violations, reads the offending vehicle's plate with TrOCR, and logs the
violation to CSV with an evidence screenshot.

Two entry points:

- `python main.py` — terminal UI with OpenCV window
- `python gui.py` — Tkinter desktop UI

---

## Setup on a new machine

### 1. Install prerequisites

- **Python 3.10 or 3.11** — https://www.python.org/downloads/ (tick *Add Python to PATH*)
- **Git** — https://git-scm.com/downloads
- **(Optional) CUDA toolkit + matching NVIDIA driver** if you have an NVIDIA GPU and want faster YOLO/TrOCR inference

### 2. (Optional) Set up SSH for GitHub

Only needed if you'll push back. For read-only clone, skip and use the HTTPS URL in step 3.

```powershell
ssh-keygen -t ed25519 -C "you@example.com"
Get-Content $env:USERPROFILE\.ssh\id_ed25519.pub | Set-Clipboard
```

Paste the key into GitHub → *Settings* → *SSH and GPG keys* → *New SSH key*. Test:

```powershell
ssh -T git@github.com
```

### 3. Clone the repo

```powershell
git clone git@github.com:farai-coder/traffic-enforcement.git
cd traffic-enforcement
```

(HTTPS alternative: `git clone https://github.com/farai-coder/traffic-enforcement.git`)

### 4. Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If activation is blocked, run once: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

On macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 5. Install Python dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

This pulls `ultralytics`, `opencv-python`, `numpy`, `pyserial`, `transformers`, `torch`, `pillow`.

**GPU users** — replace CPU torch with the CUDA build (example for CUDA 12.1):

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 6. Pre-download models (recommended)

Both models download lazily on first run, but you can fetch them upfront so
the first run isn't blocked on a long download.

**YOLOv8 nano (~6 MB)** — auto-cached by ultralytics:

```powershell
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

**TrOCR small printed (~250 MB)** — cached at `%USERPROFILE%\.cache\huggingface\hub\`:

```powershell
python -c "from transformers import TrOCRProcessor, VisionEncoderDecoderModel; TrOCRProcessor.from_pretrained('microsoft/trocr-small-printed'); VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-small-printed'); print('TrOCR cached')"
```

To put the HuggingFace cache somewhere other than the default:

```powershell
$env:HF_HOME = "D:\models\hf"
# rerun the TrOCR command above
```

For a fully offline machine: pre-download on a connected box, copy
`%USERPROFILE%\.cache\huggingface\hub\models--microsoft--trocr-small-printed\`
to the same path on the target, and set `$env:TRANSFORMERS_OFFLINE = "1"`.

### 7. Configure for your environment

Open `config.py` and update:

| Setting | What to change |
|---|---|
| `CAMERA_SOURCE` | IP of the phone running *IP Webcam* (e.g. `http://192.168.x.x:8080/video`), or `0` for the local webcam |
| `ANPR_CAMERA_SOURCE` | Webcam index for plate reading |
| `SERIAL_PORT` | ESP32 port — find it in Device Manager (e.g. `COM3`, `COM5`) |
| `TRAFFIC_LIGHT_ROI`, `STOP_LINE_*`, `LANE_LINES` | Re-calibrate for the new camera framing — run `python calibrate.py` |

### 8. Create the output folder

```powershell
New-Item -ItemType Directory -Force data\violations | Out-Null
```

(macOS/Linux: `mkdir -p data/violations`)

### 9. (Optional) Hardware setup

- **Phone camera** — install *IP Webcam* (Android), start the server, copy the `http://…/video` URL into `CAMERA_SOURCE`. Phone and PC must be on the same Wi-Fi.
- **ESP32 traffic light** — flash one of the sketches in `esp32/`, plug in via USB, confirm the COM port matches `SERIAL_PORT`.

### 10. Run it

```powershell
python main.py        # OpenCV terminal UI
python gui.py         # Tkinter desktop UI
```

**Controls**: `R`/`Y`/`G` set light state, `SPACE` pauses, `T` resets tracking, `S` screenshots, `Q` quits.

> **ANPR is automatic.** Plate capture is triggered by a detected violation,
> not by any keypress. When YOLOv8 flags a vehicle for `red_light`, `stop_line`,
> or `lane_change`, the pipeline crops the offending vehicle, runs the plate
> detector, and OCRs it with TrOCR — the result is cached per `track_id` so
> the same vehicle is never read twice. The plate number lands in the CSV
> next to the violation. `S` is a manual *screenshot* of the whole frame
> (debug aid) — it does **not** trigger ANPR.

---

## Usage — running the system end-to-end

After setup, here's the path from "deps installed" to "watching it catch a violation".

### 1. Activate the venv (every new terminal session)

```powershell
cd traffic-enforcement
.\.venv\Scripts\Activate.ps1
```

Prompt should show `(.venv)`. If activation is blocked: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

### 2. Point the camera

Open `config.py` and pick one:

- **Phone (IP Webcam app)** → `CAMERA_SOURCE = "http://192.168.1.x:8080/video"` (the IP your phone shows)
- **Built-in webcam** → `CAMERA_SOURCE = 0`
- **Pre-recorded video** → `CAMERA_SOURCE = "data/test.mp4"`

Phone and PC must share the same Wi-Fi. Confirm with `ping 192.168.1.x`.

### 3. Verify camera + light detection

```powershell
python test_traffic_light.py
```

A window opens showing the live feed with the detected light state in the corner. Press `Q` to quit. If the window doesn't appear, the camera URL is wrong.

### 4. Calibrate the geometry

The stop line, lane lines, and traffic-light ROI need to match where things appear in **your** camera frame.

```powershell
python calibrate.py
```

Click in the window to set the points. The values get written back to `config.py`. Re-run any time you reposition the camera.

### 5. Verify ANPR works on a still photo

Take a phone photo of a plate, save it as `test_plate.jpg` in the project folder, then:

```powershell
python test_anpr.py test_plate.jpg
```

Expect a cropped plate window and `[RESULT] Plate text: XXX1234` in the console. First run is slow (TrOCR weights loading); subsequent reads are fast.

With no argument, `python test_anpr.py` opens the webcam and waits for SPACE to capture a frame to OCR.

### 6. Run the full system

```powershell
python main.py        # OpenCV terminal UI
# or
python gui.py         # Tkinter desktop UI
```

A window opens showing the live feed with overlays:

- **Green box** → vehicle detected, no violation
- **Red box** → violating vehicle
- **Yellow line** → lane boundary
- **Red line** → stop line
- **`PLATE: XXX1234`** appears above the violator's box once OCR completes

### 7. Drive the simulation

You're in manual light mode, so you control the light from the keyboard:

| Key | Action |
|---|---|
| `R` | Set light to RED |
| `Y` | Set light to YELLOW |
| `G` | Set light to GREEN |
| `T` | Reset all tracking + plate cache |
| `S` | Save a debug screenshot of the current frame |
| `SPACE` | Pause / resume |
| `Q` | Quit |

**Trigger a red-light violation**: press `R`, then drive a toy car (or wave a real one) across the stop line. Within ~1 second you should see:

- The vehicle's box flip to red
- Console: `[ANPR] Vehicle #N -> plate: ...` then `[VIOLATION] RED_LIGHT | Vehicle #N | Plate: ...`
- A new row in `data/violations/violations.csv`
- A JPG in `data/violations/red_light_N_<timestamp>.jpg`

**Trigger a lane-change violation**: light state doesn't matter — drive a vehicle across one of the yellow lane lines.

### 8. Check what got logged

```powershell
ii data\violations\violations.csv
```

(`ii` = `Invoke-Item`, opens it in the default app.) Rows look like:

```
2026-05-06 14:23:11, red_light, 7, AEG4521, 0.87, data/violations/red_light_7_20260506_142311.jpg
```

Each row pairs with its evidence JPG in the same folder.

### 9. When done

Press `Q` to quit. The console prints `[DONE] Total violations logged: N`. The venv stays active until you `deactivate` or close the terminal.

### Common runtime gotchas

1. **Black/empty window** → wrong `CAMERA_SOURCE` or the phone isn't streaming.
2. **No vehicles detected** → confidence too high. Lower `CONFIDENCE_THRESHOLD = 0.3` in `config.py`.
3. **Vehicle detected but no violation** → stop line / lane lines aren't where you think; re-run `calibrate.py`.
4. **`[ANPR] -> plate unreadable`** → plate is too small, blurry, or angled; move the camera closer or improve lighting.
5. **ANPR slow** → first run only; weights are loading. After that it's <1 s per plate on CPU.

---

## Project layout

```
traffic-enforcement/
├── main.py                  # OpenCV pipeline (violation detection + ANPR)
├── gui.py                   # Tkinter desktop UI (same pipeline)
├── config.py                # All tunables (camera, ROIs, model paths, serial)
├── calibrate.py             # Click-to-set stop line, lane lines, light ROI
├── add_plates.py            # Generate Zim plates on toy cars via Gemini
├── api_format.md            # Spec for POST /api/violations (not yet wired)
├── anpr/
│   ├── plate_detector.py    # White-region + morph + contour plate localization
│   └── ocr.py               # TrOCR (microsoft/trocr-small-printed)
├── capture/camera.py        # Threaded video reader (IP cam / webcam)
├── detection/
│   ├── traffic_state.py     # manual / serial (ESP32) / HSV light detection
│   └── violation_detector.py# YOLOv8 tracking + zone-crossing logic
├── violations/logger.py     # CSV + evidence-image logging
└── esp32/                   # Arduino sketches for the toy traffic light
```

---

## Troubleshooting

- **`cv2` import error** → `pip install --force-reinstall opencv-python`
- **Phone stream won't open** → ping the phone IP from the PC; check firewall; confirm same Wi-Fi
- **`SerialException` on COM port** → close Arduino IDE / other apps holding the port, or change `SERIAL_PORT`
- **TrOCR / EasyOCR slow first run** → models are downloading; subsequent runs are fast
- **CUDA not used** → `python -c "import torch; print(torch.cuda.is_available())"` should print `True`
- **CUDA available but model still slow** → first OCR on each plate is slow because TrOCR loads weights into VRAM; subsequent reads are fast.
