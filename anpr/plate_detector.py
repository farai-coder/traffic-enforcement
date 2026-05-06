import cv2
import numpy as np
import config


class PlateDetector:
    """Detects and localizes license plates in a vehicle crop.

    Uses multiple strategies:
    1. Color-based detection (white plate with red/dark text - Zim plates)
    2. Morphological + contour detection
    3. Edge-based contour detection (fallback)
    """

    def __init__(self):
        pass

    def detect_plate(self, vehicle_crop):
        """Find a license plate in the vehicle image crop.

        Returns the plate crop image or None.
        """
        if vehicle_crop is None or vehicle_crop.size == 0:
            return None

        # Upscale small images for better detection
        h, w = vehicle_crop.shape[:2]
        if w < 400:
            scale = 400 / w
            vehicle_crop = cv2.resize(vehicle_crop, None, fx=scale, fy=scale,
                                       interpolation=cv2.INTER_CUBIC)

        # Primary: isolate the white rectangular plate region
        plate = self._crop_white_region(vehicle_crop)
        if plate is not None:
            return plate

        # Fallback: color-based detection
        plate = self._detect_by_color(vehicle_crop)
        if plate is not None:
            return plate

        # Fallback: morphological approach
        plate = self._detect_morphological(vehicle_crop)
        if plate is not None:
            return plate

        # Last resort: edge-based detection
        plate = self._detect_by_contour(vehicle_crop)
        return plate

    def _crop_white_region(self, image):
        """Crop the white plate region from the image using HSV masking.

        Filters out background (table, floor) by requiring the white region
        to be surrounded by a non-white (colored) area like a car body.
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, w = image.shape[:2]

        # Mask for white regions (the plate background)
        white_mask = cv2.inRange(hsv, np.array([0, 0, 150]), np.array([180, 80, 255]))

        # Clean up: close gaps in the plate text, remove small noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel_small, iterations=1)

        contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates = []
        img_area = w * h

        for contour in contours:
            x, y, cw, ch = cv2.boundingRect(contour)
            if ch == 0:
                continue
            aspect = cw / float(ch)
            area = cw * ch

            # Filter for plate-like rectangles
            if not (1.5 <= aspect <= 8.0 and cw > 20 and ch > 8):
                continue
            if not (area > img_area * 0.002 and area < img_area * 0.4):
                continue

            # Check that the white region has dark text inside it
            roi = image[y:y+ch, x:x+cw]
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            _, text_mask = cv2.threshold(gray_roi, 120, 255, cv2.THRESH_BINARY_INV)
            text_ratio = cv2.countNonZero(text_mask) / (cw * ch)
            if text_ratio < 0.05 or text_ratio > 0.7:
                continue

            # Check that the region is NOT touching the edge of the image
            # (table/floor white regions usually extend to edges)
            touches_edge = (x <= 2 or y <= 2 or x + cw >= w - 2 or y + ch >= h - 2)

            # Score: prefer plate-shaped, has text, not at edge
            aspect_score = 1.0 / (1.0 + abs(aspect - 4.0))
            size_score = min(area / float(img_area), 0.3)
            text_score = min(text_ratio / 0.3, 1.0)
            edge_penalty = 0.5 if touches_edge else 1.0
            score = (aspect_score * 0.4 + size_score * 0.3 + text_score * 0.3) * edge_penalty
            candidates.append((x, y, cw, ch, score))

        if not candidates:
            return None

        # Pick best candidate
        candidates.sort(key=lambda c: c[4], reverse=True)
        x, y, cw, ch, _ = candidates[0]

        # Tight crop with small padding
        pad_x = int(cw * 0.05)
        pad_y = int(ch * 0.1)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(w, x + cw + pad_x)
        y2 = min(h, y + ch + pad_y)
        crop = image[y1:y2, x1:x2]
        if crop.size > 0:
            return crop

        return None

    def _detect_by_color(self, image):
        """Detect plate by finding white rectangular regions.

        Zim plates are white with red text - look for bright white rectangles.
        Works well when plate is on a colored background (yellow, blue, etc.)
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, w = image.shape[:2]

        # White color range in HSV — broadened to catch plates under various lighting
        white_mask = cv2.inRange(hsv, np.array([0, 0, 150]), np.array([180, 80, 255]))

        # Clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel, iterations=1)

        contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates = []
        for contour in contours:
            x, y, cw, ch = cv2.boundingRect(contour)
            if ch == 0:
                continue
            aspect_ratio = cw / float(ch)
            area = cw * ch
            img_area = w * h

            # Plate-like: rectangular, correct aspect ratio, reasonable size
            if (1.2 <= aspect_ratio <= 8.0 and
                    cw > 15 and ch > 5 and
                    area > img_area * 0.002 and
                    area < img_area * 0.5):
                # Check if the region contains some non-white pixels (text)
                roi = image[y:y+ch, x:x+cw]
                gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                _, text_mask = cv2.threshold(gray_roi, 150, 255, cv2.THRESH_BINARY_INV)
                text_ratio = cv2.countNonZero(text_mask) / (cw * ch)
                # Plates have text covering ~15-50% of the area
                if 0.05 < text_ratio < 0.6:
                    candidates.append((x, y, cw, ch, area, text_ratio))

        if not candidates:
            return None

        # Pick best: prefer lower position, good aspect ratio
        best = None
        best_score = -1
        for (x, y, cw, ch, area, text_ratio) in candidates:
            aspect = cw / float(ch)
            aspect_score = 1.0 / (1.0 + abs(aspect - 4.0))
            position_score = y / float(h)
            size_score = area / float(img_area)
            score = aspect_score * 0.4 + position_score * 0.3 + size_score * 0.3
            if score > best_score:
                best_score = score
                best = (x, y, cw, ch)

        if best:
            x, y, cw, ch = best
            pad_x = int(cw * 0.08)
            pad_y = int(ch * 0.15)
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(w, x + cw + pad_x)
            y2 = min(h, y + ch + pad_y)
            return image[y1:y2, x1:x2]

        return None

    def _detect_morphological(self, image):
        """Plate detection using morphological operations."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

        _, thresh = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 5))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, close_kernel)
        closed = cv2.erode(closed, None, iterations=1)
        closed = cv2.dilate(closed, None, iterations=2)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates = []
        for contour in contours:
            x, y, cw, ch = cv2.boundingRect(contour)
            if ch == 0:
                continue
            aspect_ratio = cw / float(ch)
            area = cw * ch
            img_area = w * h

            if (1.5 <= aspect_ratio <= 6.0 and
                    cw > 20 and ch > 6 and
                    area > img_area * 0.01 and
                    area < img_area * 0.5):
                candidates.append((x, y, cw, ch, area))

        if not candidates:
            return None

        best = None
        best_score = -1
        for (x, y, cw, ch, area) in candidates:
            aspect = cw / float(ch)
            aspect_score = 1.0 / (1.0 + abs(aspect - 3.5))
            position_score = y / float(h)
            score = aspect_score * 0.6 + position_score * 0.4
            if score > best_score:
                best_score = score
                best = (x, y, cw, ch)

        if best:
            x, y, cw, ch = best
            pad_x = int(cw * 0.05)
            pad_y = int(ch * 0.1)
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(w, x + cw + pad_x)
            y2 = min(h, y + ch + pad_y)
            return image[y1:y2, x1:x2]

        return None

    def _detect_by_contour(self, image):
        """Fallback: edge detection + contour analysis."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        gray = cv2.bilateralFilter(gray, 11, 17, 17)
        edges = cv2.Canny(gray, 30, 200)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.dilate(edges, kernel, iterations=1)

        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:15]

        h, w = image.shape[:2]

        for contour in contours:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.018 * peri, True)

            if 4 <= len(approx) <= 6:
                x, y, cw, ch = cv2.boundingRect(approx)
                if ch == 0:
                    continue
                aspect_ratio = cw / float(ch)
                area = cw * ch

                if (1.5 <= aspect_ratio <= 6.0 and
                        cw > 30 and ch > 8 and
                        area > w * h * 0.01):
                    pad_x = int(cw * 0.05)
                    pad_y = int(ch * 0.1)
                    x1 = max(0, x - pad_x)
                    y1 = max(0, y - pad_y)
                    x2 = min(w, x + cw + pad_x)
                    y2 = min(h, y + ch + pad_y)
                    return image[y1:y2, x1:x2]

        return None
