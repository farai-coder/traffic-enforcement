"""Read a licence plate from a full frame + vehicle bounding box."""

from anpr.plate_format import normalise_plate


def crop_vehicle(frame, bbox, pad_ratio=0.05):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    pad_x = int((x2 - x1) * pad_ratio)
    pad_y = int((y2 - y1) * pad_ratio)
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)
    return frame[y1:y2, x1:x2]


def read_plate_for_vehicle(frame, bbox, plate_detector, ocr):
    """Try several plate crops; return (valid_plate, crop_used) or (None, None)."""
    vehicle_crop = crop_vehicle(frame, bbox)
    if vehicle_crop is None or vehicle_crop.size == 0:
        return None, None

    for plate_crop in plate_detector.iter_plate_crops(vehicle_crop):
        raw = ocr.read_plate(plate_crop)
        valid = normalise_plate(raw)
        if valid:
            return valid, plate_crop

    return None, None
