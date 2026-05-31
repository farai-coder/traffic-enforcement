# Training a Custom Toy-Car Detector

The stock `yolov8n.pt` is trained on real-world COCO images and **does not
recognize toy cars** as vehicles — on this setup it classifies them as
`chair` / `parking meter`, so `detect_vehicles()` (which filters to COCO
vehicle classes `[2,3,5,7]`) discards everything and no boxes appear.

> ⚠️ The `dataset_tools/build_*` auto-labelers **cannot** bootstrap this — they
> reuse the same model that can't see the toys. Toy-car labels must be drawn
> **by hand** once, after which you train a model that *can* see them.

This is a one-time, single-class (`car`) workflow.

---

## 1. Capture frames

Point the webcam at the toy cars and capture varied frames (different
positions, angles, lighting). Aim for **150–300+** frames.

```powershell
python dataset_tools/capture_toy_frames.py --auto --interval 1.0
```

- **SPACE** save one frame · **A** toggle auto-capture · **Q** quit
- Frames are preprocessed exactly like the live pipeline (rotate 90° CW,
  resize to 500px tall) and saved to `data/datasets/toy_cars/images/all/`.

Move the cars around between captures so the model sees variety.

## 2. Annotate (draw boxes)

Label every toy car in each frame as the single class **`car`**, exporting in
**YOLO format**. Put the resulting `.txt` files in
`data/datasets/toy_cars/labels/all/` (same basename as each image, e.g.
`toy_00001.jpg` → `toy_00001.txt`).

Use whichever tool you like:

| Tool | Notes |
|------|-------|
| **Roboflow** | Web-based; export as "YOLOv8". Easiest. |
| **CVAT** | Web/self-host; export "YOLO 1.1". |
| **LabelImg** | `pip install labelImg`; set format to YOLO, one class `car`. |

A YOLO label line is: `class cx cy w h` (all normalized 0–1). For one class,
`class` is always `0`.

## 3. Split into train/val + write data.yaml

```powershell
python dataset_tools/prepare_toy_dataset.py --val-split 0.2
```

Produces `images/train`, `images/val`, `labels/train`, `labels/val`, and
`data.yaml` under `data/datasets/toy_cars/`. Images with no label file are
skipped (pass `--allow-unlabeled` to keep them as background negatives).

## 4. Train

```powershell
python dataset_tools/train_toy_model.py --epochs 80
```

Fine-tunes `yolov8n.pt` on your dataset. When done it copies the best weights
to `toy_cars.pt` in the repo root and prints the config changes.

## 5. Wire the model into the system

Edit **`config.py`**:

```python
YOLO_MODEL = "toy_cars.pt"
VEHICLE_CLASSES = [0]   # custom model has ONE class: 0 = car
```

> 🔑 **The class index matters.** The pipeline filters detections by
> `VEHICLE_CLASSES`. Stock YOLO uses `[2,3,5,7]` (COCO car/moto/bus/truck), but
> your custom single-class model emits class **`0`**. If you leave
> `VEHICLE_CLASSES = [2,3,5,7]` with the custom model, every detection is
> filtered out and you'll still see no boxes.

Then run the app as before:

```powershell
python run_webcam.py      # OpenCV window, local webcam
# or the GUI on the webcam:
python -c "import config; config.CAMERA_SOURCE=0; import gui; gui.TrafficEnforcementGUI().run()"
```

---

## Tips

- **More data beats more epochs.** 200 well-labeled frames at 80 epochs will
  outperform 40 frames at 300 epochs (which just overfits).
- **Match inference conditions.** Capture under the same lighting/angle the
  system will actually run in — that's why capture preprocesses like the live
  camera.
- **ANPR is separate.** `ANPR_MODEL` still uses `yolov8n.pt` for plate
  detection; this workflow only fixes *vehicle* detection. Improving plate
  reading on toys is a separate dataset (`build_anpr_dataset.py`).
- **Iterate.** If detection is weak in some spots, capture more frames there,
  re-annotate, re-run steps 3–4.
