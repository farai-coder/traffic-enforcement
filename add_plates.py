"""
Add Zimbabwean Number Plates to Toy Cars
=========================================
1. Opens webcam - press SPACE to capture a shot of your toy car
2. Sends the image to Gemini to add a Zim number plate (white with red text)
3. Saves the result

Controls:
    SPACE   - Capture photo
    Q       - Quit

Usage:
    set GEMINI_API_KEY=your-api-key-here
    python add_plates.py
"""

import cv2
import os
import sys
import base64
from datetime import datetime
from PIL import Image
import io

try:
    import google.generativeai as genai
except ImportError:
    print("[ERROR] Install google-generativeai: pip install google-generativeai")
    sys.exit(1)


def setup_gemini():
    """Configure Gemini API."""
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        api_key = input("Enter your Gemini API key: ").strip()
    if not api_key:
        print("[ERROR] No API key provided. Set GEMINI_API_KEY or enter it when prompted.")
        sys.exit(1)
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-2.0-flash-exp")


def add_plate_with_gemini(model, image_path, plate_text=None):
    """Send image to Gemini and ask it to add a Zim number plate."""
    img = Image.open(image_path)

    if plate_text is None:
        # Generate a random Zim-style plate
        import random
        import string
        letters = ''.join(random.choices(string.ascii_uppercase, k=3))
        numbers = ''.join(random.choices(string.digits, k=4))
        plate_text = f"{letters} {numbers}"

    prompt = (
        f"Edit this image of a toy car. Add a realistic Zimbabwean vehicle number plate "
        f"to the front or rear of the toy car. The plate should be WHITE background with "
        f"RED text reading '{plate_text}'. Zimbabwean plates are rectangular, about 52cm x 11cm "
        f"in real life. Make the plate look naturally attached to the toy car, properly scaled "
        f"and positioned. Keep the rest of the image unchanged. Return only the edited image."
    )

    print(f"[GEMINI] Sending image with plate text: {plate_text}")
    print("[GEMINI] Waiting for response...")

    try:
        response = model.generate_content(
            [prompt, img],
            generation_config=genai.GenerationConfig(
                response_mime_type="text/plain"
            )
        )

        # Check if Gemini returned an image
        if response.candidates:
            for part in response.candidates[0].content.parts:
                if hasattr(part, 'inline_data') and part.inline_data:
                    # Save the generated image
                    img_data = base64.b64decode(part.inline_data.data)
                    result_img = Image.open(io.BytesIO(img_data))
                    output_path = image_path.replace(".jpg", f"_plate_{plate_text.replace(' ', '')}.jpg")
                    result_img.save(output_path)
                    print(f"[SAVED] Image with plate saved to: {output_path}")
                    return output_path, plate_text

        # If no image returned, Gemini may have given text instructions instead
        print(f"[GEMINI] Response: {response.text[:500] if response.text else 'No response'}")
        print("[INFO] Gemini could not edit the image directly. Falling back to overlay method...")
        return overlay_plate(image_path, plate_text), plate_text

    except Exception as e:
        print(f"[ERROR] Gemini API error: {e}")
        print("[INFO] Falling back to overlay method...")
        return overlay_plate(image_path, plate_text), plate_text


def overlay_plate(image_path, plate_text):
    """Fallback: overlay a plate on the image using OpenCV."""
    img = cv2.imread(image_path)
    h, w = img.shape[:2]

    # Plate dimensions (scaled to image)
    plate_w = int(w * 0.35)
    plate_h = int(plate_w * 0.22)  # Zim plate ratio ~4.7:1

    # Position plate at bottom center of image
    px = (w - plate_w) // 2
    py = h - plate_h - int(h * 0.08)

    # Draw white plate background with border
    cv2.rectangle(img, (px, py), (px + plate_w, py + plate_h), (255, 255, 255), -1)
    cv2.rectangle(img, (px, py), (px + plate_w, py + plate_h), (0, 0, 0), 2)

    # Draw red text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = plate_h / 40.0
    thickness = max(2, int(font_scale * 2))

    # Get text size to center it
    (tw, th), _ = cv2.getTextSize(plate_text, font, font_scale, thickness)
    tx = px + (plate_w - tw) // 2
    ty = py + (plate_h + th) // 2

    # Red text (BGR: 0, 0, 200)
    cv2.putText(img, plate_text, (tx, ty), font, font_scale, (0, 0, 200), thickness)

    # Add "ZIMBABWE" small text at top of plate
    small_scale = font_scale * 0.35
    small_thick = max(1, int(small_scale * 2))
    (zw, zh), _ = cv2.getTextSize("ZIMBABWE", font, small_scale, small_thick)
    zx = px + (plate_w - zw) // 2
    zy = py + zh + 4
    cv2.putText(img, "ZIMBABWE", (zx, zy), font, small_scale, (0, 0, 200), small_thick)

    output_path = image_path.replace(".jpg", f"_plate_{plate_text.replace(' ', '')}.jpg")
    cv2.imwrite(output_path, img)
    print(f"[SAVED] Overlay plate image saved to: {output_path}")
    return output_path


def main():
    model = setup_gemini()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("[ERROR] Cannot open webcam")
        return

    os.makedirs("data/plates", exist_ok=True)
    shot_count = 0

    print("\n" + "=" * 60)
    print("  NUMBER PLATE GENERATOR")
    print("  SPACE = Capture toy car | Q = Quit")
    print("  After capture, enter plate text or press Enter for random")
    print("=" * 60 + "\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()
        cv2.putText(display, "SPACE=Capture | Q=Quit",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(display, f"Shots taken: {shot_count}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("Plate Generator - Position your toy car", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord(" "):
            shot_count += 1
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"data/plates/car_{shot_count}_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            print(f"\n[CAPTURED] Shot #{shot_count} saved to {filename}")

            # Show captured image
            cv2.imshow("Captured", frame)
            cv2.waitKey(500)

            # Ask for plate text
            plate_input = input("Enter plate text (e.g. ADM 3421) or press Enter for random: ").strip()
            plate_text = plate_input if plate_input else None

            # Send to Gemini
            result_path, used_plate = add_plate_with_gemini(model, filename, plate_text)

            # Show result if it's a file
            if result_path and os.path.exists(result_path):
                result_img = cv2.imread(result_path)
                if result_img is not None:
                    cv2.imshow(f"Result - Plate: {used_plate}", result_img)
                    print("[INFO] Press any key on the image window to continue...")
                    cv2.waitKey(0)

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n[DONE] {shot_count} shots taken. Check data/plates/ folder.")


if __name__ == "__main__":
    main()
