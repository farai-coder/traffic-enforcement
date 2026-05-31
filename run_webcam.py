"""Launch the traffic-enforcement pipeline against the local webcam (index 0).

Overrides config.CAMERA_SOURCE before main() builds its Camera(), so no
edits to config.py / main.py are needed.
"""

import config

config.CAMERA_SOURCE = 0       # local webcam instead of the phone IP stream
config.CAMERA_ROTATION = None  # laptop webcam is upright (phone needs "cw")

import main

if __name__ == "__main__":
    main.main()
