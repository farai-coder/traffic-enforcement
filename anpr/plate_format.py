"""Validate and normalise licence plate text (reject OCR garbage)."""

import re

# Zimbabwe-style: 3 letters + 3–4 digits (e.g. ABC1234)
PLATE_PATTERN = re.compile(r"^[A-Z]{3}\d{3,4}$")

# Common false reads from boards, packaging, or background text
_REJECT_WORDS = frozenset({
    "UNKNOWN", "TITLE", "TOTAL", "TOTALS", "PURCHASE", "III", "THE",
    "STOP", "LINE", "LANE", "ROAD", "TOY", "CAR", "RED", "GREEN",
    "YELLOW", "LIGHT", "ZONE", "SPEED", "EXIT", "ENTRY", "ONLY",
    "CASH", "CASHIER", "TAX", "SALE", "SHOP", "STORE", "PRICE",
    "PAY", "PAID", "DUE", "AMT", "AMOUNT", "RECEIPT", "INVOICE",
})


def normalise_plate(text: str | None) -> str | None:
    """Return canonical plate text if valid, else None."""
    if not text:
        return None
    cleaned = re.sub(r"[^A-Za-z0-9]", "", text).upper()
    if len(cleaned) < 6 or len(cleaned) > 8:
        return None
    if cleaned in _REJECT_WORDS:
        return None
    if any(cleaned.startswith(w) or cleaned.endswith(w) for w in _REJECT_WORDS
           if len(w) >= 4):
        return None
    if not PLATE_PATTERN.match(cleaned):
        return None
    return cleaned


def is_valid_plate(text: str | None) -> bool:
    return normalise_plate(text) is not None
