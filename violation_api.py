"""
Receiver FastAPI for traffic violations.

Run with:
    uvicorn violation_api:app --host 0.0.0.0 --port 8001 --reload

Each POSTed violation is stored to:
    data/received/{timestamp}_{vehicle_id}_{violation_type}.json (metadata)
    data/received/{timestamp}_{vehicle_id}_{violation_type}.jpg  (image)
"""

import os
import json
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, File, Form, UploadFile

app = FastAPI(title="Traffic Violations Receiver", version="0.1")

RECEIVED_DIR = "data/received"
os.makedirs(RECEIVED_DIR, exist_ok=True)


@app.get("/")
def root():
    return {"service": "Traffic Violations Receiver", "endpoint": "POST /api/violations"}


@app.post("/api/violations")
async def create_violation(
    violation_type: str = Form(...),
    timestamp: str = Form(...),
    vehicle_id: int = Form(...),
    light_state: str = Form(...),
    image: UploadFile = File(...),
    plate_number: Optional[str] = Form(None),
    confidence: Optional[float] = Form(None),
):
    received_at = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    base = f"{received_at}_{vehicle_id}_{violation_type}"
    img_path = os.path.join(RECEIVED_DIR, f"{base}.jpg")
    meta_path = os.path.join(RECEIVED_DIR, f"{base}.json")

    img_bytes = await image.read()
    with open(img_path, "wb") as f:
        f.write(img_bytes)

    meta = {
        "violation_type": violation_type,
        "timestamp": timestamp,
        "vehicle_id": vehicle_id,
        "light_state": light_state,
        "plate_number": plate_number,
        "confidence": confidence,
        "image_filename": image.filename,
        "image_size_bytes": len(img_bytes),
        "received_at": received_at,
        "stored_image": img_path,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[RECEIVED] {violation_type} #{vehicle_id} plate={plate_number} "
          f"light={light_state} img={len(img_bytes)} bytes -> {base}", flush=True)

    return {
        "success": True,
        "data": {
            "id": received_at,
            "violation_type": violation_type,
            "plate_number": plate_number,
            "timestamp": timestamp,
            "vehicle_id": vehicle_id,
            "confidence": confidence,
            "light_state": light_state,
            "image_url": f"/storage/{base}.jpg",
            "status": "received",
        },
    }
