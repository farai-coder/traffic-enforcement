"""
dataset_tools
=============
Dataset-collection utilities for the Traffic Enforcement System.

These scripts REUSE the models already in this project to auto-label raw
footage into ready-to-use datasets. They do not modify any existing module
or model — they only read frames in and write labeled samples out under
`data/datasets/` (which is gitignored).

Models reused (unchanged):
    * YOLOv8            -> ultralytics, via config.YOLO_MODEL  (vehicle detection)
    * PlateDetector     -> anpr.plate_detector                (plate localization/crop)
    * PlateOCR / TrOCR  -> anpr.ocr (microsoft/trocr-base-printed)  (plate text)
    * ViolationDetector -> detection.violation_detector        (violation logic)

Builders:
    build_vehicle_dataset.py    YOLO-format vehicle-detection dataset
    build_plate_crops_dataset.py  vehicle + cropped-plate image pairs
    build_anpr_dataset.py       plate crops labeled with OCR text + manifest
    build_violation_dataset.py  violation evidence frames + JSON metadata
"""
