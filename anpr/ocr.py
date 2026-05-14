import cv2
import re
import numpy as np
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


class PlateOCR:
    """Extracts text from a license plate image using Microsoft TrOCR (printed text)."""

    def __init__(self):
        print("[OCR] Loading microsoft/trocr-base-printed model...")
        self.processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
        self.model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")
        print("[OCR] Model loaded.")

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

        best_text = None
        best_score = 0

        for img in candidates:
            text = self._run_trocr(img)
            if text:
                clean = re.sub(r'[^A-Z0-9]', '', text.upper())
                # Try to extract Zim plate pattern: 3 letters + 3-4 digits
                plate = self._extract_zim_plate(clean)
                if plate:
                    return plate
                score = len(clean)
                if score > best_score:
                    best_score = score
                    best_text = clean

        return best_text if best_text and len(best_text) >= 3 else None

    def _extract_zim_plate(self, text):
        """Try to extract a Zimbabwean plate pattern (AAA 1234) from raw OCR text."""
        # Zim plates: 3 letters followed by 3-4 digits
        match = re.search(r'([A-Z]{2,3}\d{3,4})', text)
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
        except Exception as e:
            return None
