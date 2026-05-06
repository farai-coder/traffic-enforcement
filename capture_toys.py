"""
Capture Toy Car Photos
======================
Takes photos of your toy cars to paste into Gemini for plate generation.

Controls:
    SPACE   - Take a photo
    Q       - Quit
"""

import cv2
import os
from datetime import datetime

os.makedirs("data/plates", exist_ok=True)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("[ERROR] Cannot open webcam")
    exit()

shot_count = 0
print("\nSPACE = Take photo | Q = Quit\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    display = frame.copy()
    cv2.putText(display, f"SPACE=Capture | Q=Quit | Shots: {shot_count}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.imshow("Capture Toy Cars", display)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord(" "):
        shot_count += 1
        filename = f"data/plates/car_{shot_count}.jpg"
        cv2.imwrite(filename, frame)
        print(f"[SAVED] {filename}")

cap.release()
cv2.destroyAllWindows()
print(f"\nDone. {shot_count} photos saved in data/plates/")
