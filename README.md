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

---

## Smoke tests

```powershell
python test_traffic_light.py   # verify light detection
python test_anpr.py            # verify plate reading
```

`test_anpr.py <image.jpg>` runs ANPR on a single image; with no args it opens the webcam and waits for SPACE to capture.

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
- **`gui.py` crashes on launch with `TypeError`** → known: `gui.py` calls `TrafficLightDetector(manual_mode=True)` but the constructor takes `mode="manual"`. Patch: change that one line.
