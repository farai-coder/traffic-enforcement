import os

import cv2
import re
import numpy as np
from PIL import Image

import config
from anpr.plate_format import normalise_plate


class NullPlateOCR:
    """Fallback when TrOCR cannot load (offline / no cache)."""

    def read_plate(self, plate_image):
        return None


class PlateOCR:
    """Extracts text from a license plate image using Microsoft TrOCR (printed text)."""

    def __init__(self, model_name=None):
        if not getattr(config, "OCR_ENABLED", True):
            raise OSError("OCR disabled in config (OCR_ENABLED = False)")

        self.model_name = model_name or getattr(
            config, "TROCR_MODEL", "microsoft/trocr-small-printed"
        )
        print(f"[OCR] Loading {self.model_name}...", flush=True)

        from transformers import TrOCRProcessor, VisionEncoderDecoderModel

        load_kwargs = {}
        # Prefer cached weights when Hugging Face is unreachable.
        try:
            self.processor = TrOCRProcessor.from_pretrained(
                self.model_name, local_files_only=True, **load_kwargs
            )
            self.model = VisionEncoderDecoderModel.from_pretrained(
                self.model_name, local_files_only=True, **load_kwargs
            )
        except OSError:
            print("[OCR] Not in local cache — downloading (needs internet)...", flush=True)
            self.processor = TrOCRProcessor.from_pretrained(self.model_name, **load_kwargs)
            self.model = VisionEncoderDecoderModel.from_pretrained(
                self.model_name, **load_kwargs
            )

        print("[OCR] Model loaded.", flush=True)

    def read_plate(self, plate_image):
        """Extract text from a plate crop image.

        Tries multiple preprocessing methods and returns the best result.
        Returns the plate text string or None if unreadable.
        """
        if plate_image is None or plate_image.size == 0:
            return None

        gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)

        # Resize small plates
        h, w = gray.shape
        if w < 200:
            scale = 200 / w
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        # Try multiple preprocessing approaches, pick best result
        candidates = []

        # Method 1: Original color image
        candidates.append(plate_image)

        # Method 2: Adaptive threshold (converted back to BGR for PIL)
        thresh1 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, 11, 2)
        candidates.append(cv2.cvtColor(thresh1, cv2.COLOR_GRAY2BGR))

        # Method 3: Otsu threshold
        _, thresh2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        candidates.append(cv2.cvtColor(thresh2, cv2.COLOR_GRAY2BGR))

        # Method 4: CLAHE enhanced
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        candidates.append(cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR))

        for img in candidates:
            text = self._run_trocr(img)
            if not text:
                continue
            clean = re.sub(r"[^A-Z0-9]", "", text.upper())
            plate = normalise_plate(clean)
            if plate:
                return plate
            plate = normalise_plate(self._extract_zim_plate(clean))
            if plate:
                return plate

        return None

    def _extract_zim_plate(self, text):
        """Try to extract a Zimbabwean plate pattern (AAA 1234) from raw OCR text."""
        match = re.search(r"([A-Z]{3}\d{3,4})", text)
        if match:
            return match.group(1)
        return None

    def _run_trocr(self, image):
        """Run TrOCR on a BGR image."""
        try:
            # Convert BGR to RGB PIL Image
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)

            # Resize to reasonable dimensions for TrOCR
            w, h = pil_img.size
            if w < 200:
                scale = 200 / w
                pil_img = pil_img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

            pixel_values = self.processor(images=pil_img, return_tensors="pt").pixel_values
            generated_ids = self.model.generate(pixel_values, max_new_tokens=20)
            text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

            return text.strip()
        except Exception:
            return None


def load_plate_ocr():
    """Load TrOCR or return a no-op OCR if offline / unavailable."""
    try:
        return PlateOCR()
    except OSError as e:
        print(
            "[WARN] OCR unavailable — violations will still run; plates logged as UNKNOWN.",
            flush=True,
        )
        print(f"[WARN] Reason: {e}", flush=True)
        print(
            "[WARN] Fix: connect to internet once and run:\n"
            "       python -c \"from anpr.ocr import PlateOCR; PlateOCR()\"",
            flush=True,
        )
        return NullPlateOCR()
