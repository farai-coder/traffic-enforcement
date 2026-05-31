"""Run the calibration tool against the local webcam (index 0).

Overrides config.CAMERA_SOURCE before calibrate.main() reads it, so no edits
to config.py are needed. Draw the stop line (key 2) and the red-light line
(key 4), then press S to save into config.py.
"""

import config

config.CAMERA_SOURCE = 0       # local webcam instead of the phone IP stream
config.CAMERA_ROTATION = None  # laptop webcam is upright (phone needs "cw")

import calibrate

if __name__ == "__main__":
    calibrate.main()
