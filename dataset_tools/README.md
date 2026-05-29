# dataset_tools — auto-label datasets from your own footage

These scripts turn raw images/videos into labeled datasets by **reusing the
models this project already runs** — they don't add or change any model, and
they don't modify any existing file. All output is written under
`data/datasets/` (already gitignored).

| Script | Produces | Models reused |
|--------|----------|---------------|
| `build_vehicle_dataset.py` | YOLO-format vehicle-detection dataset (`images/`, `labels/`, `dataset.yaml`) | YOLOv8 (`config.YOLO_MODEL`) |
| `build_plate_crops_dataset.py` | Vehicle crops + cropped plates + manifest | YOLOv8 + `PlateDetector` |
| `build_anpr_dataset.py` | Plate crops labeled with OCR text + `gt.csv` | YOLOv8 + `PlateDetector` + `PlateOCR` (TrOCR) |
| `build_violation_dataset.py` | Violation evidence frames + per-event JSON | `ViolationDetector` (YOLOv8 tracking + zone logic) |

## Input

Every script accepts `--input` as **a single image, a folder of images
(searched recursively), or a video file**. For videos, `--every N` keeps 1 of
every N frames and `--max-frames N` caps the total.

## Examples

```powershell
# 1. Vehicle-detection dataset from a clip (every 5th frame), with preview boxes
python dataset_tools/build_vehicle_dataset.py --input data/clips/intersection.mp4 --every 5 --preview

# 2. Plate-cropping dataset from your existing toy-car photos
python dataset_tools/build_plate_crops_dataset.py --input data/plates

# 3. ANPR dataset — already-cropped plates, OCR straight away
python dataset_tools/build_anpr_dataset.py --input data/plates --from-plate-crops

#    ANPR dataset — full pipeline from a video
python dataset_tools/build_anpr_dataset.py --input data/clips/cars.mp4 --every 10

# 4. Violation dataset — replay a red-light clip assuming the light is red
python dataset_tools/build_violation_dataset.py --input data/clips/redlight.mp4 --light red
```

## Notes

- **OCR text is a weak label.** `build_anpr_dataset.py` writes whatever TrOCR
  reads; review/correct `gt.csv` before treating it as ground truth.
- **Violation light state.** Recorded footage has no live signal, so you pass
  `--light red|green|yellow` to tell the detector what to enforce (red enforces
  the stop line; green/yellow allow lane-change only, matching the live system).
- Detection thresholds default to `config.CONFIDENCE_THRESHOLD`; override with
  `--conf`.
- Re-running a builder overwrites same-named outputs but appends nothing to the
  repo — safe to iterate.
