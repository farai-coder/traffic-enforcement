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
import numpy as np
import threading
import time
from PIL import Image, ImageTk
from datetime import datetime

import config
from capture.camera import Camera
from capture.video_buffer import VideoBuffer
from detection.traffic_state import TrafficLightDetector
from detection.violation_detector import ViolationDetector
from anpr.plate_detector import PlateDetector
from anpr.ocr import PlateOCR
from violations.logger import ViolationLogger


class TrafficEnforcementGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Traffic Enforcement System")
        self.root.configure(bg="#1e1e2e")
        self.root.state("zoomed")  # Maximize window

        self.running = False
        self.paused = False
        self.frame_count = 0
        self.logged_violations = set()
        self.current_frame = None

        # Initialize modules
        self.camera = Camera()
        self.video_buffer = VideoBuffer(pre_seconds=1.5, post_seconds=1.5)
        # Serial mode: read live light state from the ESP32 over the COM port in
        # settings.json. Falls back to manual (R/Y/G buttons) if the port can't open.
        self.light_detector = TrafficLightDetector(mode="serial")
        self.light_detector.set_state("unknown")
        self.violation_detector = ViolationDetector()
        self.plate_detector = PlateDetector()
        self.plate_ocr = PlateOCR()
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

        # -- Serial Port (ESP32 traffic light) --
        serial_frame = tk.LabelFrame(right_frame, text=" SERIAL PORT (ESP32) ",
                                      font=("Consolas", 11, "bold"),
                                      fg="#cdd6f4", bg="#1e1e2e",
                                      labelanchor="n", bd=2, relief="groove")
        serial_frame.pack(fill=tk.X, pady=(0, 10))

        row = tk.Frame(serial_frame, bg="#1e1e2e")
        row.pack(fill=tk.X, padx=8, pady=(8, 4))

        self.port_var = tk.StringVar(value=self.light_detector.port)
        self.port_combo = ttk.Combobox(row, textvariable=self.port_var,
                                        values=self.light_detector.list_ports(),
                                        width=10, font=("Consolas", 10))
        self.port_combo.pack(side=tk.LEFT, padx=(0, 5))

        tk.Button(row, text="↻", font=("Consolas", 10, "bold"),
                  bg="#585b70", fg="white", bd=0, width=2,
                  command=self._refresh_ports).pack(side=tk.LEFT, padx=2)
        tk.Button(row, text="Connect", font=("Consolas", 10, "bold"),
                  bg="#89b4fa", fg="#1e1e2e", bd=0, width=8,
                  command=self._connect_port).pack(side=tk.LEFT, padx=2)

        self.serial_status = tk.Label(serial_frame, text="", font=("Consolas", 9),
                                      fg="#bac2de", bg="#1e1e2e", wraplength=360,
                                      justify="left")
        self.serial_status.pack(fill=tk.X, padx=8, pady=(0, 8))
        self._set_serial_status()

        # -- Camera Source --
        cam_frame = tk.LabelFrame(right_frame, text=" CAMERA SOURCE ",
                                   font=("Consolas", 11, "bold"),
                                   fg="#cdd6f4", bg="#1e1e2e",
                                   labelanchor="n", bd=2, relief="groove")
        cam_frame.pack(fill=tk.X, pady=(0, 10))

        crow = tk.Frame(cam_frame, bg="#1e1e2e")
        crow.pack(fill=tk.X, padx=8, pady=(8, 4))

        self.cam_var = tk.StringVar(value=str(self.camera.source))
        self.cam_combo = ttk.Combobox(crow, textvariable=self.cam_var,
                                      values=["0", "1", "2"], width=14,
                                      font=("Consolas", 10))
        self.cam_combo.pack(side=tk.LEFT, padx=(0, 5))

        tk.Button(crow, text="Connect", font=("Consolas", 10, "bold"),
                  bg="#89b4fa", fg="#1e1e2e", bd=0, width=8,
                  command=self._connect_camera).pack(side=tk.LEFT, padx=2)

        self.cam_status = tk.Label(cam_frame, text="", font=("Consolas", 9),
                                   fg="#bac2de", bg="#1e1e2e", wraplength=360,
                                   justify="left")
        self.cam_status.pack(fill=tk.X, padx=8, pady=(0, 8))

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

    def _refresh_ports(self):
        ports = self.light_detector.list_ports()
        self.port_combo["values"] = ports
        if ports and self.port_var.get() not in ports:
            self.port_var.set(ports[0])
        print(f"[SERIAL] Available ports: {ports or 'none found'}")

    def _connect_port(self):
        port = self.port_var.get().strip()
        if not port:
            self.serial_status.config(text="Enter or select a port (e.g. COM5).", fg="#f9e2af")
            return
        ok, msg = self.light_detector.set_serial_port(port)
        self.serial_status.config(text=msg, fg="#a6e3a1" if ok else "#f38ba8")
        print(f"[SERIAL] {msg}")

    def _set_serial_status(self):
        connected = self.light_detector._serial_conn is not None
        if connected:
            self.serial_status.config(text=f"Connected: {self.light_detector.port}", fg="#a6e3a1")
        else:
            self.serial_status.config(
                text="Not connected (manual R/Y/G). Pick a port and Connect.", fg="#f9e2af")

    def _connect_camera(self):
        source = self.cam_var.get().strip()
        if not source:
            self.cam_status.config(text="Enter a camera index (0,1) or stream URL.", fg="#f9e2af")
            return
        # Keep rotation consistent with the source (webcam upright, phone sideways).
        config.CAMERA_ROTATION = config.default_rotation_for(source)
        ok, msg = self.camera.set_source(source)
        if ok:
            # Persist so this machine remembers its camera (no more phone-IP default).
            import settings
            settings.set_camera_source(source)
            msg += "  (saved)"
        self.cam_status.config(text=msg, fg="#a6e3a1" if ok else "#f38ba8")
        print(f"[CAMERA] {msg}")

    def _set_camera_status(self):
        if self.camera.is_opened():
            self.cam_status.config(text=f"Connected: {self.camera.source}", fg="#a6e3a1")
        else:
            self.cam_status.config(
                text="No camera. Enter a source and Connect.", fg="#f9e2af")

    def _show_no_camera(self):
        """Display a placeholder on the canvas when no camera is available."""
        placeholder = np.full((360, 640, 3), 24, dtype=np.uint8)
        cv2.putText(placeholder, "NO CAMERA", (170, 170),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (180, 180, 180), 3)
        cv2.putText(placeholder, "Select a source under CAMERA SOURCE -> Connect",
                    (90, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (140, 140, 140), 1)
        self._update_canvas(placeholder)

    def _set_light(self, state):
        self.light_detector.set_state(state)
        colors = {"red": "#f38ba8", "yellow": "#f9e2af", "green": "#a6e3a1"}
        fg = colors.get(state, "#cdd6f4")
        self.light_label.config(text=state.upper(), fg=fg)
        print(f"[LIGHT] Switched to {state.upper()}")

    def _reset_tracking(self):
        self.logged_violations.clear()
        self.violation_detector.tracked_vehicles.clear()
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

            frame = self.camera.read()
            if frame is None:
                # No camera yet — show a placeholder and keep the UI responsive
                # so a source can be connected later.
                self.root.after(0, self._show_no_camera)
                time.sleep(0.2)
                continue

            saved_clip = self.video_buffer.add(frame)
            if saved_clip:
                print(f"[VIDEO] Saved {saved_clip}")
            self.frame_count += 1

            # Step 1: Detect traffic light state
            light_state = self.light_detector.detect(frame)

            # Step 2: Detect vehicles and check violations
            detections = self.violation_detector.detect_vehicles(frame)
            violations = self.violation_detector.check_violations(detections, light_state)

            # Step 3: For each new violation, run ANPR and log
            for v in violations:
                key = (v["track_id"], v["type"])
                if key not in self.logged_violations:
                    self.logged_violations.add(key)

                    x1, y1, x2, y2 = v["bbox"]
                    vehicle_crop = frame[y1:y2, x1:x2]

                    plate_text = None
                    if vehicle_crop.size > 0:
                        plate_image = self.plate_detector.detect_plate(vehicle_crop)
                        if plate_image is not None:
                            plate_text = self.plate_ocr.read_plate(plate_image)

                    image_path = self.logger.log(
                        violation_type=v["type"],
                        track_id=v["track_id"],
                        plate_number=plate_text,
                        confidence=v.get("conf", 0.0),
                        frame=frame,
                        light_state=light_state,
                    )

                    if image_path:
                        video_path = image_path.rsplit(".", 1)[0] + ".mp4"
                        if self.video_buffer.trigger(video_path):
                            print(f"[VIDEO] Recording post-roll → {video_path}")

                    # Update GUI from main thread
                    self.root.after(0, self._add_violation, v["type"], v["track_id"], plate_text)

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
            text=f"Vehicles: {num_vehicles} | Violations: {len(self.logged_violations)} | Frame: {self.frame_count}"
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
        self._set_camera_status()
        self.running = True

        # Start video processing in background thread
        video_thread = threading.Thread(target=self._video_loop, daemon=True)
        video_thread.start()

        # Run tkinter main loop
        self.root.mainloop()

        print(f"\n[DONE] Total violations logged: {len(self.logged_violations)}")
        print(f"[DONE] Evidence saved in: data/violations/")


if __name__ == "__main__":
    app = TrafficEnforcementGUI()
    app.run()
