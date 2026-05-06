"""Quick test for traffic light detection using your webcam."""
import cv2
from detection.traffic_state import TrafficLightDetector

cap = cv2.VideoCapture(0)
detector = TrafficLightDetector()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    state = detector.detect(frame)
    detector.draw(frame)

    cv2.imshow("Traffic Light Test", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
