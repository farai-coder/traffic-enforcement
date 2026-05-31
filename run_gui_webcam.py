"""Launch the GUI against the local webcam (index 0), upright (no rotation).

Overrides config before gui imports build their Camera()/ViolationDetector(),
so no edits to config.py are needed. Matches run_webcam.py but for the GUI.
"""

import config

config.CAMERA_SOURCE = 0       # local webcam instead of the phone IP stream
config.CAMERA_ROTATION = None  # laptop webcam is upright (phone needs "cw")

import gui

if __name__ == "__main__":
    gui.TrafficEnforcementGUI().run()
