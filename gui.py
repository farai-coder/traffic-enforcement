"""
Traffic Enforcement System - GUI
================================
Tkinter interface with:
  - Live video feed with overlays (stop line, lanes, detections)
  - Traffic light control buttons (Red / Yellow / Green)
  - Violations log table (type, vehicle ID, plate number, timestamp)
  - Status bar

Controls:
  R/Y/G keys or buttons to change light state
  T to reset tracking
  Q to quit
"""

import tkinter as tk
from tkinter import ttk
import cv2
import threading
import time
import config
from PIL import Image, ImageTk
from datetime import datetime

from capture.camera import Camera
from capture.frame_thread import FrameCaptureThread
from capture.video_buffer import VideoBuffer
from detection.traffic_state import TrafficLightDetector
from detection.violation_detector import ViolationDetector
from anpr.plate_detector import PlateDetector
from anpr.plate_cache import PlateReadCache
from anpr.reader import read_plate_for_vehicle
from anpr.ocr import load_plate_ocr
from violations.logger import ViolationLogger
from violations.incident import (
    IncidentGate,
    PhotoSampler,
    collapse_by_vehicle,
    label_from_subject,
)


class TrafficEnforcementGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Traffic Enforcement System")
        self.root.configure(bg="#1e1e2e")
        self.root.state("zoomed")  # Maximize window

        self.running = False
        self.paused = False
        self.frame_count = 0
        self.current_frame = None

        # Initialize modules
        if getattr(config, "USE_ASYNC_CAMERA", True):
            self.camera = FrameCaptureThread()
        else:
            self.camera = Camera()
        self._detect_skip = max(1, getattr(config, "DETECTION_FRAME_SKIP", 2))
        self._detect_tick = 0
        self._overlay = {
            "detections": [], "violations": [], "light_state": "unknown",
        }
        self.video_buffer = VideoBuffer(
            pre_seconds=getattr(config, "VIOLATION_VIDEO_PRE_SECONDS", 1.0),
            post_seconds=getattr(config, "VIOLATION_VIDEO_POST_SECONDS", 1.0),
        )
        self.incident_gate = IncidentGate(
            cooldown_seconds=getattr(config, "VIOLATION_COOLDOWN_SECONDS", 60),
        )
        self.photo_sampler = PhotoSampler(
            interval_seconds=getattr(config, "VIOLATION_PHOTO_INTERVAL_SECONDS", 0.5),
        )
        self.photos_saved = 0
        self.videos_recorded = 0
        self.light_detector = TrafficLightDetector(mode="manual")
        self.light_detector.set_state("green")
        self.violation_detector = ViolationDetector()
        self.plate_detector = PlateDetector()
        self.plate_ocr = load_plate_ocr()
        self.plate_cache = PlateReadCache(max_attempts=20)
        self.logger = ViolationLogger()

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.bind("<KeyPress>", self._on_key)

    def _build_ui(self):
        # Main container
        main_frame = tk.Frame(self.root, bg="#1e1e2e")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left side: video feed
        left_frame = tk.Frame(main_frame, bg="#1e1e2e")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Video title
        tk.Label(left_frame, text="LIVE FEED", font=("Consolas", 14, "bold"),
                 fg="#cdd6f4", bg="#1e1e2e").pack(pady=(0, 5))

        # Video canvas
        self.canvas = tk.Canvas(left_frame, bg="#11111b", highlightthickness=1,
                                highlightbackground="#45475a")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Right side: controls + violations
        right_frame = tk.Frame(main_frame, bg="#1e1e2e", width=400)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        right_frame.pack_propagate(False)

        # -- Traffic Light Control --
        light_frame = tk.LabelFrame(right_frame, text=" TRAFFIC LIGHT ",
                                     font=("Consolas", 11, "bold"),
                                     fg="#cdd6f4", bg="#1e1e2e",
                                     labelanchor="n", bd=2, relief="groove")
        light_frame.pack(fill=tk.X, pady=(0, 10))

        self.light_label = tk.Label(light_frame, text="GREEN",
                                     font=("Consolas", 20, "bold"),
                                     fg="#a6e3a1", bg="#1e1e2e")
        self.light_label.pack(pady=10)

        btn_frame = tk.Frame(light_frame, bg="#1e1e2e")
        btn_frame.pack(pady=(0, 10))

        self.btn_red = tk.Button(btn_frame, text="RED (R)", font=("Consolas", 10, "bold"),
                                  bg="#f38ba8", fg="white", width=10, bd=0,
                                  activebackground="#e06080",
                                  command=lambda: self._set_light("red"))
        self.btn_red.pack(side=tk.LEFT, padx=5)

        self.btn_yellow = tk.Button(btn_frame, text="YELLOW (Y)", font=("Consolas", 10, "bold"),
                                     bg="#f9e2af", fg="#1e1e2e", width=10, bd=0,
                                     activebackground="#e0c878",
                                     command=lambda: self._set_light("yellow"))
        self.btn_yellow.pack(side=tk.LEFT, padx=5)

        self.btn_green = tk.Button(btn_frame, text="GREEN (G)", font=("Consolas", 10, "bold"),
                                    bg="#a6e3a1", fg="#1e1e2e", width=10, bd=0,
                                    activebackground="#80c080",
                                    command=lambda: self._set_light("green"))
        self.btn_green.pack(side=tk.LEFT, padx=5)

        # -- Stats --
        stats_frame = tk.LabelFrame(right_frame, text=" STATUS ",
                                     font=("Consolas", 11, "bold"),
                                     fg="#cdd6f4", bg="#1e1e2e",
                                     labelanchor="n", bd=2, relief="groove")
        stats_frame.pack(fill=tk.X, pady=(0, 10))

        self.stats_label = tk.Label(stats_frame,
                                     text="Vehicles: 0 | Violations: 0 | Frame: 0",
                                     font=("Consolas", 10),
                                     fg="#bac2de", bg="#1e1e2e")
        self.stats_label.pack(pady=8)

        # -- Violations Log --
        log_frame = tk.LabelFrame(right_frame, text=" VIOLATIONS LOG ",
                                   font=("Consolas", 11, "bold"),
                                   fg="#cdd6f4", bg="#1e1e2e",
                                   labelanchor="n", bd=2, relief="groove")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        # Treeview for violations
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Violations.Treeview",
                         background="#181825",
                         foreground="#cdd6f4",
                         fieldbackground="#181825",
                         font=("Consolas", 9),
                         rowheight=28)
        style.configure("Violations.Treeview.Heading",
                         background="#313244",
                         foreground="#cdd6f4",
                         font=("Consolas", 9, "bold"))
        style.map("Violations.Treeview",
                   background=[("selected", "#45475a")])

        columns = ("time", "type", "vehicle", "plate")
        self.tree = ttk.Treeview(log_frame, columns=columns, show="headings",
                                  style="Violations.Treeview", height=15)
        self.tree.heading("time", text="Time")
        self.tree.heading("type", text="Violation")
        self.tree.heading("vehicle", text="Vehicle ID")
        self.tree.heading("plate", text="Plate")
        self.tree.column("time", width=80, anchor="center")
        self.tree.column("type", width=110, anchor="center")
        self.tree.column("vehicle", width=70, anchor="center")
        self.tree.column("plate", width=100, anchor="center")

        scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)

        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # -- Bottom buttons --
        bottom_frame = tk.Frame(right_frame, bg="#1e1e2e")
        bottom_frame.pack(fill=tk.X)

        tk.Button(bottom_frame, text="RESET (T)", font=("Consolas", 10, "bold"),
                  bg="#585b70", fg="white", bd=0, width=15,
                  command=self._reset_tracking).pack(side=tk.LEFT, padx=5)

        tk.Button(bottom_frame, text="QUIT (Q)", font=("Consolas", 10, "bold"),
                  bg="#f38ba8", fg="white", bd=0, width=15,
                  command=self._on_close).pack(side=tk.RIGHT, padx=5)

    def _set_light(self, state):
        self.light_detector.set_state(state)
        colors = {"red": "#f38ba8", "yellow": "#f9e2af", "green": "#a6e3a1"}
        fg = colors.get(state, "#cdd6f4")
        self.light_label.config(text=state.upper(), fg=fg)
        print(f"[LIGHT] Switched to {state.upper()}")

    def _reset_tracking(self):
        self.incident_gate.clear()
        self.photo_sampler.clear()
        self.plate_cache.clear()
        self.photos_saved = 0
        self.videos_recorded = 0
        self.violation_detector.reset_tracking()
        # Clear the treeview
        for item in self.tree.get_children():
            self.tree.delete(item)
        print("[RESET] Tracking and violations cleared.")

    def _on_key(self, event):
        key = event.char.lower()
        if key == "r":
            self._set_light("red")
        elif key == "g":
            self._set_light("green")
        elif key == "y":
            self._set_light("yellow")
        elif key == "t":
            self._reset_tracking()
        elif key == "q":
            self._on_close()

    def _add_violation(self, violation_type, track_id, plate_text):
        timestamp = datetime.now().strftime("%H:%M:%S")
        display_type = violation_type.replace("_", " ").upper()
        plate = plate_text or "UNKNOWN"

        # Add to top of list
        self.tree.insert("", 0, values=(timestamp, display_type, f"#{track_id}", plate))

        # Flash the row red briefly
        item = self.tree.get_children()[0]
        self.tree.tag_configure("flash", background="#f38ba8", foreground="white")
        self.tree.item(item, tags=("flash",))
        self.root.after(1500, lambda: self.tree.item(item, tags=()))

    def _video_loop(self):
        while self.running:
            if self.paused:
                time.sleep(0.03)
                continue

            raw = self.camera.read()
            frame = raw[0] if isinstance(raw, tuple) else raw
            if frame is None:
                continue

            saved_clip = self.video_buffer.add(frame)
            if saved_clip:
                print(f"[VIDEO] Saved {saved_clip}")
            self.frame_count += 1
            self._detect_tick += 1
            run_detection = self._detect_tick % self._detect_skip == 0

            if run_detection:
                light_state = self.light_detector.detect(frame)
                detections = self.violation_detector.detect_vehicles(frame)
                violations = collapse_by_vehicle(
                    self.violation_detector.check_violations(detections, light_state)
                )
                self._overlay["detections"] = detections
                self._overlay["violations"] = violations
                self._overlay["light_state"] = light_state
            else:
                light_state = self._overlay["light_state"]
                detections = self._overlay["detections"]
                violations = self._overlay["violations"]

            evidence_subjects = self.violation_detector.subjects_for_evidence(
                detections, light_state
            )
            new_offense_ids = (
                {v["track_id"] for v in violations} if run_detection else set()
            )
            conf_by_id = {d["track_id"]: d["conf"] for d in detections}
            require_plate = getattr(config, "VIOLATION_REQUIRE_VALID_PLATE", True)

            capture_queue = []
            if run_detection:
                for v in violations:
                    tid = v["track_id"]
                    capture_queue.append((
                        True,
                        {
                            "track_id": tid,
                            "bbox": v["bbox"],
                            "types": self.violation_detector.committed_violation_types(
                                tid, {"track_id": tid, "bbox": v["bbox"]}, light_state
                            ),
                            "conf": v.get("conf", conf_by_id.get(tid, 0.0)),
                        },
                    ))
            for sub in evidence_subjects:
                if sub["track_id"] not in new_offense_ids:
                    capture_queue.append((False, sub))

            for immediate_flag, sub in capture_queue:
                track_id = sub["track_id"]
                if require_plate and self.plate_cache.skip_capture(track_id):
                    continue
                urgent = immediate_flag or (
                    require_plate
                    and not self.plate_cache.get(track_id)
                    and not self.plate_cache.skip_capture(track_id)
                )
                if not self.photo_sampler.should_capture(track_id, immediate=urgent):
                    continue

                bbox = sub["bbox"]

                def _read_plate(bbox=bbox):
                    text, _ = read_plate_for_vehicle(
                        frame, bbox, self.plate_detector, self.plate_ocr,
                    )
                    return text

                plate_number = self.plate_cache.resolve(track_id, _read_plate)
                if require_plate and not plate_number:
                    continue

                self.photo_sampler.mark_captured(track_id)
                violation_label = label_from_subject(
                    self.violation_detector, sub, light_state
                )

                image_path = self.logger.log(
                    violation_type=violation_label,
                    track_id=track_id,
                    plate_number=plate_number,
                    confidence=sub.get("conf", conf_by_id.get(track_id, 0.0)),
                    frame=frame,
                )
                if image_path:
                    self.photos_saved += 1

                if self.incident_gate.allow(track_id, plate_number):
                    self.incident_gate.mark(track_id, plate_number)
                    self.videos_recorded += 1

                    if image_path and getattr(config, "VIOLATION_SAVE_VIDEO", True):
                        video_path = image_path.rsplit(".", 1)[0] + ".mp4"
                        if self.video_buffer.trigger(video_path):
                            print(f"[VIDEO] Short clip → {video_path}")

                    self.root.after(
                        0, self._add_violation, violation_label, track_id, plate_number
                    )

            # Step 4: Draw annotations
            frame = self.light_detector.draw(frame)
            frame = self.violation_detector.draw(frame, detections, violations, light_state)

            # Draw light state indicator on frame
            light_colors = {"red": (0, 0, 255), "yellow": (0, 255, 255),
                           "green": (0, 255, 0), "unknown": (128, 128, 128)}
            lc = light_colors.get(light_state, (128, 128, 128))
            cv2.circle(frame, (30, 30), 18, lc, -1)
            cv2.circle(frame, (30, 30), 18, (255, 255, 255), 2)
            cv2.putText(frame, light_state.upper(), (55, 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, lc, 2)

            # Update stats from main thread
            self.root.after(0, self._update_stats, len(detections))

            # Convert frame for tkinter
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.current_frame = frame_rgb

            # Display on canvas from main thread
            self.root.after(0, self._update_canvas, frame_rgb)

            time.sleep(0.01)

    def _update_canvas(self, frame_rgb):
        try:
            canvas_w = self.canvas.winfo_width()
            canvas_h = self.canvas.winfo_height()
            if canvas_w < 10 or canvas_h < 10:
                return

            h, w = frame_rgb.shape[:2]
            scale = min(canvas_w / w, canvas_h / h)
            new_w = int(w * scale)
            new_h = int(h * scale)

            img = Image.fromarray(frame_rgb)
            img = img.resize((new_w, new_h), Image.LANCZOS)
            self.photo = ImageTk.PhotoImage(img)

            self.canvas.delete("all")
            x_offset = (canvas_w - new_w) // 2
            y_offset = (canvas_h - new_h) // 2
            self.canvas.create_image(x_offset, y_offset, anchor=tk.NW, image=self.photo)
        except Exception:
            pass

    def _update_stats(self, num_vehicles):
        self.stats_label.config(
            text=f"Vehicles: {num_vehicles} | Photos: {self.photos_saved} | Videos: {self.videos_recorded} | Frame: {self.frame_count}"
        )

    def _on_close(self):
        self.running = False
        time.sleep(0.1)
        self.camera.release()
        self.root.destroy()

    def run(self):
        print("=" * 60)
        print("  TRAFFIC ENFORCEMENT SYSTEM - GUI")
        print("  R=Red | G=Green | Y=Yellow | T=Reset | Q=Quit")
        print("=" * 60)

        self.camera.start()
        self.running = True

        # Start video processing in background thread
        video_thread = threading.Thread(target=self._video_loop, daemon=True)
        video_thread.start()

        # Run tkinter main loop
        self.root.mainloop()

        print(f"\n[DONE] Photos: {self.photos_saved} | Videos: {self.videos_recorded}")
        print(f"[DONE] Evidence saved in: data/violations/")


if __name__ == "__main__":
    app = TrafficEnforcementGUI()
    app.run()
