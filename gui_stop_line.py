"""
Beautiful GUI for stop-line / red-light / lane-change detection (threaded).

Shows four live frames (clean detection, annotated stop-line view, ANPR webcam
frame, detected plate crop), plus the most recent violation as a coloured
badge, the most recent plate read in big monospace, an LED-style traffic
light widget, metric cards, and a scrolling violations log.

Reuses FrameCaptureThread + helper geometry from test_stop_line.py and runs
the same three background workers (phone capture, ANPR webcam capture,
ANPR/API worker). The main thread runs Tkinter; a dedicated detection thread
runs YOLO + violation rules.
"""

import queue
import re
import threading
import time
import tkinter as tk
from datetime import datetime
from tkinter import ttk

import cv2
from PIL import Image, ImageTk
from ultralytics import YOLO

import config
import settings
from anpr.plate_detector import PlateDetector
from anpr.ocr import PlateOCR
from api_client import post_violation
from detection.traffic_state import TrafficLightDetector
from test_stop_line import FrameCaptureThread, _crossed_line, _side_of_line


# ----- Catppuccin Mocha-inspired palette -----
BG_BASE      = "#11111b"
BG_PANEL     = "#1e1e2e"
BG_RAISED    = "#181825"
BG_SUBTLE    = "#313244"
TEXT_HI      = "#cdd6f4"
TEXT_MED     = "#a6adc8"
TEXT_LO      = "#6c7086"
ACCENT_RED   = "#f38ba8"
ACCENT_YEL   = "#f9e2af"
ACCENT_GRN   = "#a6e3a1"
ACCENT_BLU   = "#89b4fa"
ACCENT_LAV   = "#b4befe"
ACCENT_TEAL  = "#94e2d5"
LED_DIM      = "#2a2a3a"

TAG_TO_TYPE = {
    "STOPLINE": "stop_line",
    "REDLIGHT": "red_light",
    "LANECHANGE": "lane_change",
}
TAG_COLOR = {
    "REDLIGHT":   ACCENT_RED,
    "STOPLINE":   ACCENT_YEL,
    "LANECHANGE": ACCENT_BLU,
}
TAG_LABEL = {
    "REDLIGHT":   "RED-LIGHT",
    "STOPLINE":   "STOP-LINE",
    "LANECHANGE": "LANE-CHANGE",
}

# A real plate is exactly 3 letters followed by 4 digits (case-insensitive).
PLATE_PATTERN = re.compile(r"^[A-Z]{3}[0-9]{4}$")


def normalise_plate(text):
    """Return canonical plate (uppercased, no spaces) if it matches the
    AAA1234 pattern, else None."""
    if not text:
        return None
    cleaned = re.sub(r"[^A-Za-z0-9]", "", text).upper()
    if PLATE_PATTERN.match(cleaned):
        return cleaned
    return None


class StopLineGUI:
    def __init__(self):
        self.settings = settings.load()
        self.main_capture_lock = threading.Lock()

        camera_url = settings.camera_source(self.settings)
        print(f"[INFO] Starting phone capture thread on {camera_url} "
              f"(opening in background)...", flush=True)
        self.main_capture = FrameCaptureThread(camera_url, name="phone")
        threading.Thread(
            target=self._open_main_capture_async,
            args=(camera_url,),
            daemon=True,
            name="phone-opener",
        ).start()

        if config.ANPR_CAMERA_SOURCE == config.CAMERA_SOURCE:
            print("[INFO] ANPR camera = main camera; reusing frames", flush=True)
            self.anpr_capture = None
        else:
            print(f"[INFO] Starting ANPR capture thread on source "
                  f"{config.ANPR_CAMERA_SOURCE}...", flush=True)
            self.anpr_capture = FrameCaptureThread(
                config.ANPR_CAMERA_SOURCE, name="anpr",
                api_pref=cv2.CAP_DSHOW, warmup=10,
            )
            if not self.anpr_capture.start():
                print(f"[WARN] Cannot open ANPR camera source "
                      f"{config.ANPR_CAMERA_SOURCE} — ANPR will fall back to "
                      f"the main capture frame for plate reads.",
                      flush=True)
                self.anpr_capture = None

        print("[INFO] Loading YOLO + ANPR + OCR (this may take ~10s)...", flush=True)
        self.model = YOLO(config.YOLO_MODEL)
        self.plate_detector = PlateDetector()
        self.ocr = PlateOCR()
        print("[INFO] Initializing ESP32 traffic light over serial...", flush=True)
        self.light_detector = TrafficLightDetector(mode="serial")
        print("[INFO] Models ready.", flush=True)

        self.frame_slots = {"clean": None, "annotated": None,
                            "anpr": None, "plate": None}
        self.frame_lock = threading.Lock()
        self.gui_queue = queue.Queue()
        self.anpr_queue = queue.Queue(maxsize=20)
        self.stop_event = threading.Event()
        self.running = True

        self.last_capture_t = {}
        self.last_lane_t = {}
        self.track_positions = {}
        self.cooldown_s = 1.0
        self.last_seen_id = -1

        self.counts = {"REDLIGHT": 0, "STOPLINE": 0, "LANECHANGE": 0}
        self.last_anpr_t = 0.0
        self.tracking_state = {}  # track_id -> dict(cx, cy, prev, hist, sides, last_seen)
        self.tracking_lock = threading.Lock()

        self._build_ui()

        self.detection_thread = threading.Thread(
            target=self._detection_loop, daemon=True, name="detection")
        self.anpr_thread = threading.Thread(
            target=self._anpr_loop, daemon=True, name="anpr-worker")
        self.detection_thread.start()
        self.anpr_thread.start()

        self.root.after(33, self._gui_tick)

    # =================== UI ===================
    def _build_ui(self):
        self.root = tk.Tk()
        self.root.title("Traffic Enforcement System")
        self.root.configure(bg=BG_BASE)
        self.root.geometry("1600x980")
        self.root.minsize(1280, 800)

        # ---- Header bar ----
        header = tk.Frame(self.root, bg=BG_PANEL, height=64)
        header.pack(fill=tk.X, side=tk.TOP)
        header.pack_propagate(False)

        title_wrap = tk.Frame(header, bg=BG_PANEL)
        title_wrap.pack(side=tk.LEFT, padx=20, pady=10)
        tk.Label(title_wrap, text="◉ TRAFFIC ENFORCEMENT",
                 font=("Consolas", 18, "bold"),
                 fg=TEXT_HI, bg=BG_PANEL).pack(anchor="w")
        tk.Label(title_wrap, text="Stop-line • Red-light • Lane-change",
                 font=("Consolas", 9),
                 fg=TEXT_LO, bg=BG_PANEL).pack(anchor="w")

        # Camera config (centre-left)
        cam_wrap = tk.Frame(header, bg=BG_PANEL)
        cam_wrap.pack(side=tk.LEFT, padx=20, pady=10)
        tk.Label(cam_wrap, text="PHONE CAM",
                 font=("Consolas", 8, "bold"),
                 fg=TEXT_LO, bg=BG_PANEL).pack(anchor="w")
        ip_row = tk.Frame(cam_wrap, bg=BG_PANEL)
        ip_row.pack(anchor="w", pady=(2, 0))
        self.ip_var = tk.StringVar(value=str(self.settings.get("phone_ip", "")))
        self.port_var = tk.StringVar(value=str(self.settings.get("phone_port", 8080)))
        ip_entry = tk.Entry(ip_row, textvariable=self.ip_var,
                            font=("Consolas", 11), width=16,
                            bg=BG_RAISED, fg=TEXT_HI, insertbackground=TEXT_HI,
                            bd=0, highlightthickness=1,
                            highlightbackground=BG_SUBTLE,
                            highlightcolor=ACCENT_LAV)
        ip_entry.pack(side=tk.LEFT, ipady=4, padx=(0, 4))
        tk.Label(ip_row, text=":", font=("Consolas", 11),
                 fg=TEXT_MED, bg=BG_PANEL).pack(side=tk.LEFT)
        port_entry = tk.Entry(ip_row, textvariable=self.port_var,
                              font=("Consolas", 11), width=6,
                              bg=BG_RAISED, fg=TEXT_HI, insertbackground=TEXT_HI,
                              bd=0, highlightthickness=1,
                              highlightbackground=BG_SUBTLE,
                              highlightcolor=ACCENT_LAV)
        port_entry.pack(side=tk.LEFT, ipady=4, padx=(4, 6))
        tk.Button(ip_row, text="RECONNECT",
                  font=("Consolas", 9, "bold"),
                  bg=ACCENT_LAV, fg=BG_BASE,
                  activebackground=ACCENT_LAV, activeforeground=BG_BASE,
                  bd=0, padx=10, pady=3, cursor="hand2",
                  command=self._reconnect_phone).pack(side=tk.LEFT)
        ip_entry.bind("<Return>", lambda _e: self._reconnect_phone())
        port_entry.bind("<Return>", lambda _e: self._reconnect_phone())

        # ESP32 / serial config
        ser_wrap = tk.Frame(header, bg=BG_PANEL)
        ser_wrap.pack(side=tk.LEFT, padx=20, pady=10)
        tk.Label(ser_wrap, text="ESP32 SERIAL",
                 font=("Consolas", 8, "bold"),
                 fg=TEXT_LO, bg=BG_PANEL).pack(anchor="w")
        ser_row = tk.Frame(ser_wrap, bg=BG_PANEL)
        ser_row.pack(anchor="w", pady=(2, 0))
        self.com_var = tk.StringVar(value=str(self.settings.get("serial_port", "COM5")))
        self.baud_var = tk.StringVar(value=str(self.settings.get("serial_baud", 9600)))
        com_entry = tk.Entry(ser_row, textvariable=self.com_var,
                             font=("Consolas", 11), width=8,
                             bg=BG_RAISED, fg=TEXT_HI, insertbackground=TEXT_HI,
                             bd=0, highlightthickness=1,
                             highlightbackground=BG_SUBTLE,
                             highlightcolor=ACCENT_TEAL)
        com_entry.pack(side=tk.LEFT, ipady=4, padx=(0, 4))
        tk.Label(ser_row, text="@", font=("Consolas", 11),
                 fg=TEXT_MED, bg=BG_PANEL).pack(side=tk.LEFT)
        baud_entry = tk.Entry(ser_row, textvariable=self.baud_var,
                              font=("Consolas", 11), width=7,
                              bg=BG_RAISED, fg=TEXT_HI, insertbackground=TEXT_HI,
                              bd=0, highlightthickness=1,
                              highlightbackground=BG_SUBTLE,
                              highlightcolor=ACCENT_TEAL)
        baud_entry.pack(side=tk.LEFT, ipady=4, padx=(4, 6))
        tk.Button(ser_row, text="RECONNECT",
                  font=("Consolas", 9, "bold"),
                  bg=ACCENT_TEAL, fg=BG_BASE,
                  activebackground=ACCENT_TEAL, activeforeground=BG_BASE,
                  bd=0, padx=10, pady=3, cursor="hand2",
                  command=self._reconnect_serial).pack(side=tk.LEFT)
        com_entry.bind("<Return>", lambda _e: self._reconnect_serial())
        baud_entry.bind("<Return>", lambda _e: self._reconnect_serial())

        right_hdr = tk.Frame(header, bg=BG_PANEL)
        right_hdr.pack(side=tk.RIGHT, padx=20, pady=10)
        self.status_dot = tk.Label(right_hdr, text="●",
                                   font=("Consolas", 18),
                                   fg=ACCENT_GRN, bg=BG_PANEL)
        self.status_dot.pack(side=tk.LEFT, padx=(0, 6))
        self.status_text = tk.Label(right_hdr, text="LIVE",
                                    font=("Consolas", 11, "bold"),
                                    fg=TEXT_HI, bg=BG_PANEL)
        self.status_text.pack(side=tk.LEFT, padx=(0, 18))
        self.clock_label = tk.Label(right_hdr, text="--:--:--",
                                    font=("Consolas", 11),
                                    fg=TEXT_MED, bg=BG_PANEL)
        self.clock_label.pack(side=tk.LEFT)

        # ---- Body ----
        body = tk.Frame(self.root, bg=BG_BASE)
        body.pack(fill=tk.BOTH, expand=True, padx=14, pady=14)

        # Left: 2x2 frames
        left = tk.Frame(body, bg=BG_BASE)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        top_row = tk.Frame(left, bg=BG_BASE)
        top_row.pack(fill=tk.BOTH, expand=True, pady=(0, 7))
        self.canvas_clean = self._make_panel(top_row, "DETECTION (clean)", ACCENT_TEAL)
        self.canvas_annotated = self._make_panel(top_row, "STOP LINE TEST", ACCENT_YEL)

        bot_row = tk.Frame(left, bg=BG_BASE)
        bot_row.pack(fill=tk.BOTH, expand=True, pady=(7, 0))
        self.canvas_anpr = self._make_panel(bot_row, "ANPR FRAME", ACCENT_LAV)
        self.canvas_plate = self._make_panel(bot_row, "DETECTED PLATE", ACCENT_GRN)

        # Right: sidebar
        right = tk.Frame(body, bg=BG_BASE, width=420)
        right.pack(side=tk.RIGHT, fill=tk.Y, padx=(14, 0))
        right.pack_propagate(False)

        self._build_violation_card(right)
        self._build_traffic_light(right)
        self._build_metrics(right)
        self._build_tracking(right)
        self._build_log(right)
        self._build_buttons(right)

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.bind("<KeyPress>", self._on_key)

    def _make_panel(self, parent, title, accent):
        # Outer frame acts as accent border
        outer = tk.Frame(parent, bg=accent, padx=1, pady=1)
        outer.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4)
        inner = tk.Frame(outer, bg=BG_PANEL)
        inner.pack(fill=tk.BOTH, expand=True)

        header = tk.Frame(inner, bg=BG_PANEL, height=28)
        header.pack(fill=tk.X)
        header.pack_propagate(False)
        tk.Label(header, text="●", font=("Consolas", 11),
                 fg=accent, bg=BG_PANEL).pack(side=tk.LEFT, padx=(10, 4))
        tk.Label(header, text=title, font=("Consolas", 10, "bold"),
                 fg=TEXT_HI, bg=BG_PANEL).pack(side=tk.LEFT)

        canvas = tk.Canvas(inner, bg=BG_BASE, highlightthickness=0)
        canvas.pack(fill=tk.BOTH, expand=True, padx=1, pady=(0, 1))
        return canvas

    def _build_violation_card(self, parent):
        outer = tk.Frame(parent, bg=BG_SUBTLE, padx=1, pady=1)
        outer.pack(fill=tk.X, pady=(0, 12))
        self.viol_card = tk.Frame(outer, bg=BG_PANEL, padx=18, pady=14)
        self.viol_card.pack(fill=tk.X)

        tk.Label(self.viol_card, text="LAST VIOLATION",
                 font=("Consolas", 9, "bold"),
                 fg=TEXT_LO, bg=BG_PANEL).pack(anchor="w")

        # Type badge (bg color matches the violation type)
        self.type_badge_outer = tk.Frame(self.viol_card, bg=BG_PANEL)
        self.type_badge_outer.pack(fill=tk.X, pady=(8, 8))
        self.type_badge = tk.Label(self.type_badge_outer,
                                   text="—",
                                   font=("Consolas", 22, "bold"),
                                   fg=BG_BASE, bg=BG_SUBTLE,
                                   padx=14, pady=8)
        self.type_badge.pack(fill=tk.X)

        # Plate (very prominent monospace)
        self.plate_label = tk.Label(self.viol_card, text="—",
                                    font=("Consolas", 26, "bold"),
                                    fg=TEXT_HI, bg=BG_PANEL)
        self.plate_label.pack(anchor="w", pady=(2, 4))

        # Vehicle id + time
        self.vehicle_label = tk.Label(self.viol_card, text="vehicle —   time —",
                                      font=("Consolas", 10),
                                      fg=TEXT_MED, bg=BG_PANEL)
        self.vehicle_label.pack(anchor="w")

    def _build_traffic_light(self, parent):
        outer = tk.Frame(parent, bg=BG_SUBTLE, padx=1, pady=1)
        outer.pack(fill=tk.X, pady=(0, 12))
        wrap = tk.Frame(outer, bg=BG_PANEL, padx=18, pady=14)
        wrap.pack(fill=tk.X)

        tk.Label(wrap, text="TRAFFIC LIGHT",
                 font=("Consolas", 9, "bold"),
                 fg=TEXT_LO, bg=BG_PANEL).pack(anchor="w")

        row = tk.Frame(wrap, bg=BG_PANEL)
        row.pack(fill=tk.X, pady=(10, 6))

        # Three stacked LED dots inside a "housing"
        housing = tk.Frame(row, bg=BG_RAISED, padx=10, pady=10,
                           highlightbackground=BG_SUBTLE, highlightthickness=1)
        housing.pack(side=tk.LEFT)
        self.led_red = self._make_led(housing, ACCENT_RED)
        self.led_red.pack(pady=3)
        self.led_yel = self._make_led(housing, ACCENT_YEL)
        self.led_yel.pack(pady=3)
        self.led_grn = self._make_led(housing, ACCENT_GRN)
        self.led_grn.pack(pady=3)

        # State name beside the LEDs
        right = tk.Frame(row, bg=BG_PANEL)
        right.pack(side=tk.LEFT, padx=(16, 0), fill=tk.BOTH, expand=True)
        self.light_name = tk.Label(right, text="UNKNOWN",
                                   font=("Consolas", 22, "bold"),
                                   fg=TEXT_MED, bg=BG_PANEL)
        self.light_name.pack(anchor="w", pady=(6, 0))
        tk.Label(right, text="ESP32 • COM5", font=("Consolas", 9),
                 fg=TEXT_LO, bg=BG_PANEL).pack(anchor="w")

    def _make_led(self, parent, color):
        c = tk.Canvas(parent, width=24, height=24, bg=BG_RAISED, highlightthickness=0)
        # store the active color so we can recolor later
        c._active_color = color
        c._dim_color = LED_DIM
        c.create_oval(2, 2, 22, 22, fill=LED_DIM, outline="", tags="dot")
        return c

    def _set_led(self, canvas, on):
        canvas.itemconfig("dot", fill=canvas._active_color if on else canvas._dim_color)

    def _build_metrics(self, parent):
        outer = tk.Frame(parent, bg=BG_SUBTLE, padx=1, pady=1)
        outer.pack(fill=tk.X, pady=(0, 12))
        wrap = tk.Frame(outer, bg=BG_PANEL, padx=14, pady=12)
        wrap.pack(fill=tk.X)

        tk.Label(wrap, text="STATUS",
                 font=("Consolas", 9, "bold"),
                 fg=TEXT_LO, bg=BG_PANEL).pack(anchor="w")

        grid = tk.Frame(wrap, bg=BG_PANEL)
        grid.pack(fill=tk.X, pady=(8, 2))

        self.metric_red = self._make_metric(grid, "RED-LIGHT", ACCENT_RED, 0)
        self.metric_stop = self._make_metric(grid, "STOP-LINE", ACCENT_YEL, 1)
        self.metric_lane = self._make_metric(grid, "LANE-CHG", ACCENT_BLU, 2)
        for i in range(3):
            grid.columnconfigure(i, weight=1, uniform="metric")

        # second row of stats
        row2 = tk.Frame(wrap, bg=BG_PANEL)
        row2.pack(fill=tk.X, pady=(8, 0))
        self.queue_label = tk.Label(row2, text="queue 0/20",
                                    font=("Consolas", 9),
                                    fg=TEXT_MED, bg=BG_PANEL)
        self.queue_label.pack(side=tk.LEFT)
        self.fps_label = tk.Label(row2, text="anpr idle",
                                  font=("Consolas", 9),
                                  fg=TEXT_LO, bg=BG_PANEL)
        self.fps_label.pack(side=tk.RIGHT)

    def _make_metric(self, parent, label, color, col):
        cell = tk.Frame(parent, bg=BG_RAISED,
                        highlightbackground=BG_SUBTLE, highlightthickness=1)
        cell.grid(row=0, column=col, padx=4, sticky="nsew")
        tk.Label(cell, text=label,
                 font=("Consolas", 8, "bold"),
                 fg=color, bg=BG_RAISED).pack(pady=(8, 0))
        val = tk.Label(cell, text="0",
                       font=("Consolas", 20, "bold"),
                       fg=TEXT_HI, bg=BG_RAISED)
        val.pack(pady=(2, 8))
        return val

    def _build_tracking(self, parent):
        outer = tk.Frame(parent, bg=BG_SUBTLE, padx=1, pady=1)
        outer.pack(fill=tk.X, pady=(0, 12))
        wrap = tk.Frame(outer, bg=BG_PANEL, padx=14, pady=12)
        wrap.pack(fill=tk.X)

        header_row = tk.Frame(wrap, bg=BG_PANEL)
        header_row.pack(fill=tk.X)
        tk.Label(header_row, text="TRACKING",
                 font=("Consolas", 9, "bold"),
                 fg=TEXT_LO, bg=BG_PANEL).pack(side=tk.LEFT)
        self.track_count_label = tk.Label(header_row, text="0 vehicles",
                                          font=("Consolas", 9),
                                          fg=TEXT_MED, bg=BG_PANEL)
        self.track_count_label.pack(side=tk.RIGHT)

        self.tracking_text = tk.Text(
            wrap, height=7, bg=BG_RAISED, fg=TEXT_HI,
            font=("Consolas", 9), bd=0, padx=8, pady=6,
            highlightthickness=1, highlightbackground=BG_SUBTLE,
            wrap=tk.NONE,
        )
        self.tracking_text.pack(fill=tk.X, pady=(8, 0))
        self.tracking_text.tag_configure("id",   foreground=ACCENT_LAV)
        self.tracking_text.tag_configure("flip", foreground=ACCENT_GRN)
        self.tracking_text.tag_configure("dim",  foreground=TEXT_LO)
        self.tracking_text.config(state="disabled")

    def _build_log(self, parent):
        outer = tk.Frame(parent, bg=BG_SUBTLE, padx=1, pady=1)
        outer.pack(fill=tk.BOTH, expand=True, pady=(0, 12))
        wrap = tk.Frame(outer, bg=BG_PANEL, padx=12, pady=10)
        wrap.pack(fill=tk.BOTH, expand=True)

        tk.Label(wrap, text="VIOLATIONS LOG",
                 font=("Consolas", 9, "bold"),
                 fg=TEXT_LO, bg=BG_PANEL).pack(anchor="w", pady=(0, 6))

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("V.Treeview",
                        background=BG_RAISED,
                        foreground=TEXT_HI,
                        fieldbackground=BG_RAISED,
                        bordercolor=BG_PANEL,
                        font=("Consolas", 9), rowheight=26)
        style.configure("V.Treeview.Heading",
                        background=BG_SUBTLE,
                        foreground=TEXT_HI,
                        font=("Consolas", 9, "bold"),
                        relief="flat")
        style.map("V.Treeview",
                  background=[("selected", BG_SUBTLE)])

        cols = ("time", "type", "vid", "plate")
        self.tree = ttk.Treeview(wrap, columns=cols, show="headings",
                                 style="V.Treeview", height=14)
        for cid, text, width in (
            ("time", "Time", 70),
            ("type", "Violation", 110),
            ("vid", "ID", 50),
            ("plate", "Plate", 130),
        ):
            self.tree.heading(cid, text=text)
            self.tree.column(cid, width=width, anchor="center")
        # row tag colours per violation type
        self.tree.tag_configure("REDLIGHT",   foreground=ACCENT_RED)
        self.tree.tag_configure("STOPLINE",   foreground=ACCENT_YEL)
        self.tree.tag_configure("LANECHANGE", foreground=ACCENT_BLU)

        sb = ttk.Scrollbar(wrap, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)

    def _build_buttons(self, parent):
        row = tk.Frame(parent, bg=BG_BASE)
        row.pack(fill=tk.X)
        for label, color, cmd in (
            ("RESET",  BG_SUBTLE,   self._reset),
            ("QUIT",   ACCENT_RED,  self._on_close),
        ):
            btn = tk.Button(row, text=label, font=("Consolas", 10, "bold"),
                            bg=color, fg=BG_BASE if color != BG_SUBTLE else TEXT_HI,
                            activebackground=color, activeforeground=BG_BASE,
                            bd=0, padx=18, pady=10,
                            command=cmd, cursor="hand2")
            btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=3)

    def _on_key(self, event):
        k = event.char.lower()
        if k == "r":
            self.light_detector.set_state("red")
        elif k == "y":
            self.light_detector.set_state("yellow")
        elif k == "g":
            self.light_detector.set_state("green")
        elif k == "t":
            self._reset()
        elif k == "q":
            self._on_close()

    def _open_main_capture_async(self, camera_url):
        """Open the phone capture in the background so __init__ doesn't block
        on FFmpeg's long open-timeout when the phone is unreachable."""
        ok = self.main_capture.start()
        if ok:
            print(f"[INFO] Phone capture connected on {camera_url}", flush=True)
        else:
            print(f"[WARN] Cannot open {camera_url} — use Reconnect when "
                  f"phone is ready.", flush=True)

    def _reconnect_serial(self):
        """Apply COM port / baud from the entries: close existing serial,
        persist the new values, rebuild TrafficLightDetector. The detection
        thread reads light_detector.current_state directly so swapping the
        instance is safe."""
        port = self.com_var.get().strip()
        baud_s = self.baud_var.get().strip()
        if not port:
            print("[RECONNECT] empty COM port, ignored", flush=True)
            return
        try:
            baud = int(baud_s)
        except ValueError:
            print(f"[RECONNECT] invalid baud '{baud_s}'", flush=True)
            return

        self.settings["serial_port"] = port
        self.settings["serial_baud"] = baud
        settings.save(self.settings)
        # config is read at import; mutate it so TrafficLightDetector picks
        # up the new values when it constructs.
        config.SERIAL_PORT = port
        config.SERIAL_BAUD = baud
        print(f"[RECONNECT] switching ESP32 serial -> {port}@{baud}", flush=True)

        try:
            old = self.light_detector
            if getattr(old, "_serial_conn", None) is not None:
                try:
                    old._serial_conn.close()
                except Exception:
                    pass
        except Exception:
            pass

        self.light_detector = TrafficLightDetector(mode="serial")

    def _reconnect_phone(self):
        """Apply IP/port from the entries: stop the current phone capture
        thread, persist the new values, and start a fresh capture thread."""
        ip = self.ip_var.get().strip()
        port_s = self.port_var.get().strip()
        if not ip:
            print("[RECONNECT] empty IP, ignored", flush=True)
            return
        try:
            port = int(port_s)
        except ValueError:
            print(f"[RECONNECT] invalid port '{port_s}'", flush=True)
            return

        self.settings["phone_ip"] = ip
        self.settings["phone_port"] = port
        settings.save(self.settings)
        new_url = settings.camera_source(self.settings)
        print(f"[RECONNECT] switching phone source -> {new_url}", flush=True)

        with self.main_capture_lock:
            try:
                self.main_capture.stop()
            except Exception as e:
                print(f"[RECONNECT] error stopping old capture: {e}", flush=True)
            self.main_capture = FrameCaptureThread(new_url, name="phone")
            ok = self.main_capture.start()
        if ok:
            print(f"[RECONNECT] new phone capture thread started", flush=True)
            self.last_seen_id = -1
        else:
            print(f"[RECONNECT] failed to open {new_url} — phone offline?",
                  flush=True)

    def _reset(self):
        self.last_capture_t.clear()
        self.last_lane_t.clear()
        self.track_positions.clear()
        for item in self.tree.get_children():
            self.tree.delete(item)
        for k in self.counts:
            self.counts[k] = 0
        self.metric_red.config(text="0")
        self.metric_stop.config(text="0")
        self.metric_lane.config(text="0")
        self.type_badge.config(text="—", bg=BG_SUBTLE, fg=TEXT_MED)
        self.plate_label.config(text="—")
        self.vehicle_label.config(text="vehicle —   time —")
        print("[RESET] Cleared cooldowns, history, and log.", flush=True)

    # =================== detection thread ===================
    def _detection_loop(self):
        p1 = tuple(config.STOP_LINE_POINTS[0])
        p2 = tuple(config.STOP_LINE_POINTS[1])
        lane_lines = [
            (tuple(line[0]), tuple(line[1]))
            for line in getattr(config, "LANE_LINES", [])
        ]
        frame_idx = 0
        skip = max(1, getattr(config, "FRAME_SKIP", 1))

        while not self.stop_event.is_set():
            with self.main_capture_lock:
                cap = self.main_capture
            frame, frame_id = cap.read()
            if frame is None or frame_id == self.last_seen_id:
                time.sleep(0.01)
                continue
            self.last_seen_id = frame_id

            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            h, w = frame.shape[:2]
            scale = 500 / h
            frame = cv2.resize(frame, (int(w * scale), 500))
            clean = frame.copy()

            cv2.line(frame, p1, p2, (0, 0, 255), 3)
            cv2.putText(frame, "STOP LINE", (p1[0], max(15, p1[1] - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            for ll_a, ll_b in lane_lines:
                cv2.line(frame, ll_a, ll_b, (0, 255, 255), 2)

            run_yolo = (frame_idx % skip == 0)
            results = None
            if run_yolo:
                results = self.model.track(
                    clean, persist=True,
                    conf=config.CONFIDENCE_THRESHOLD,
                    classes=config.VEHICLE_CLASSES,
                    verbose=False,
                )

            now = time.time()
            n_det = 0
            n_touch = 0
            if (run_yolo and results
                    and results[0].boxes is not None
                    and results[0].boxes.id is not None):
                boxes = results[0].boxes
                n_det = len(boxes)
                for i in range(n_det):
                    x1, y1, x2, y2 = [int(v) for v in boxes.xyxy[i].tolist()]
                    track_id = int(boxes.id[i])
                    conf = float(boxes.conf[i])
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2

                    history = self.track_positions.setdefault(track_id, [])
                    history.append((cx, cy))
                    if len(history) > 30:
                        del history[:-30]

                    rect = (x1, y1, x2 - x1, y2 - y1)
                    touches, _, _ = cv2.clipLine(rect, p1, p2)
                    lane_crossed = False
                    if touches:
                        n_touch += 1
                        last_t = self.last_capture_t.get(track_id, 0.0)
                        if (now - last_t) >= self.cooldown_s:
                            self.last_capture_t[track_id] = now
                            light_state = self.light_detector.current_state
                            tag = "REDLIGHT" if light_state == "red" else "STOPLINE"
                            print(
                                f"[{tag}] Vehicle #{track_id} (bbox={x1},{y1},{x2},{y2}, "
                                f"conf={conf:.2f}, light={light_state})",
                                flush=True,
                            )
                            self._enqueue_anpr(track_id, tag, conf, frame)

                    sides_str = []
                    sides_flipped = []
                    if lane_lines and len(history) >= 2:
                        prev = history[-2]
                        curr = history[-1]
                        for j, (ll_a, ll_b) in enumerate(lane_lines):
                            s_prev = _side_of_line(prev, ll_a, ll_b)
                            s_curr = _side_of_line(curr, ll_a, ll_b)
                            sgn_prev = "+" if s_prev > 0 else "-" if s_prev < 0 else "0"
                            sgn_curr = "+" if s_curr > 0 else "-" if s_curr < 0 else "0"
                            flipped = (s_prev > 0 and s_curr < 0) or (s_prev < 0 and s_curr > 0)
                            sides_str.append(f"L{j}:{sgn_prev}->{sgn_curr}")
                            sides_flipped.append(flipped)
                        print(
                            f"[TRACK] #{track_id} c=({cx},{cy}) prev=({prev[0]},{prev[1]}) "
                            f"hist={len(history)} {' '.join(sides_str)}",
                            flush=True,
                        )
                        last_lane = self.last_lane_t.get(track_id, 0.0)
                        if (now - last_lane) >= self.cooldown_s:
                            for ll_a, ll_b in lane_lines:
                                if _crossed_line(prev, curr, ll_a, ll_b):
                                    lane_crossed = True
                                    self.last_lane_t[track_id] = now
                                    print(
                                        f"[LANECHANGE] Vehicle #{track_id} "
                                        f"(bbox={x1},{y1},{x2},{y2}, conf={conf:.2f})",
                                        flush=True,
                                    )
                                    self._enqueue_anpr(track_id, "LANECHANGE", conf, frame)
                                    break

                    with self.tracking_lock:
                        self.tracking_state[track_id] = {
                            "cx": cx, "cy": cy,
                            "prev": history[-2] if len(history) >= 2 else None,
                            "hist": len(history),
                            "sides": sides_str,
                            "flipped": sides_flipped,
                            "last_seen": now,
                            "conf": conf,
                        }

                    color = (0, 0, 255) if (touches or lane_crossed) else (0, 255, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"ID:{track_id} {conf:.2f}",
                                (x1, max(15, y1 - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            light_state = self.light_detector.current_state
            light_color = {
                "red": (0, 0, 255), "yellow": (0, 255, 255), "green": (0, 255, 0),
            }.get(light_state, (180, 180, 180))
            cv2.putText(frame, f"LIGHT: {light_state.upper()}",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, light_color, 2)
            cv2.putText(frame,
                        f"dets={n_det} touching={n_touch} q={self.anpr_queue.qsize()}",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

            with self.frame_lock:
                self.frame_slots["clean"] = clean
                self.frame_slots["annotated"] = frame

            frame_idx += 1

    def _enqueue_anpr(self, track_id, tag, conf, frame):
        try:
            self.anpr_queue.put_nowait((track_id, tag, conf, frame.copy()))
        except queue.Full:
            print(f"[ANPR] queue full, dropping {tag} #{track_id}", flush=True)

    # =================== ANPR/API thread ===================
    def _anpr_loop(self):
        while not self.stop_event.is_set():
            try:
                event = self.anpr_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if event is None:
                break
            track_id, tag, conf, fallback = event

            if self.anpr_capture is not None:
                anpr_frame, _ = self.anpr_capture.read()
            else:
                anpr_frame = fallback
            if anpr_frame is None:
                print(f"[ANPR] {tag} #{track_id} -> ANPR camera frame unavailable",
                      flush=True)
                self.anpr_queue.task_done()
                continue

            cv2.imwrite(
                f"data/violations/{tag.lower()}_vehicle_{track_id}.jpg", anpr_frame)
            anpr_preview = anpr_frame.copy()
            cv2.putText(anpr_preview, f"{tag} #{track_id}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

            plate_text = None
            plate_img = self.plate_detector.detect_plate(anpr_frame)
            plate_preview = None
            if plate_img is None:
                print(f"[ANPR] {tag} #{track_id} -> no plate detected", flush=True)
            else:
                cv2.imwrite(
                    f"data/violations/{tag.lower()}_plate_{track_id}.jpg", plate_img)
                raw_text = self.ocr.read_plate(plate_img)
                normalised = normalise_plate(raw_text)
                if normalised:
                    plate_text = normalised
                    display_text = normalised
                    print(f"[ANPR] {tag} #{track_id} -> plate: {normalised}",
                          flush=True)
                else:
                    display_text = "(invalid)"
                    if raw_text:
                        print(f"[ANPR] {tag} #{track_id} -> rejected OCR "
                              f"'{raw_text}' (not AAA1234 format)", flush=True)
                    else:
                        print(f"[ANPR] {tag} #{track_id} -> plate unreadable",
                              flush=True)
                ph, pw = plate_img.shape[:2]
                target_w = max(360, pw)
                scale_p = target_w / pw
                preview = cv2.resize(
                    plate_img, (target_w, int(ph * scale_p)),
                    interpolation=cv2.INTER_CUBIC)
                cv2.putText(preview, f"{tag} #{track_id}: {display_text}", (8, 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                plate_preview = preview

            with self.frame_lock:
                self.frame_slots["anpr"] = anpr_preview
                if plate_preview is not None:
                    self.frame_slots["plate"] = plate_preview

            self.last_anpr_t = time.time()
            self.gui_queue.put({
                "track_id": track_id,
                "tag": tag,
                "plate": plate_text or "—",
            })

            ok, resp = post_violation(
                violation_type=TAG_TO_TYPE.get(tag, tag.lower()),
                plate_number=plate_text,
                track_id=track_id,
                confidence=conf,
                light_state=self.light_detector.current_state,
                image_bgr=anpr_frame,
            )
            if ok:
                print(f"[API] {tag} #{track_id} posted OK (HTTP {resp.status_code})",
                      flush=True)
            else:
                print(f"[API] {tag} #{track_id} POST failed: {resp}", flush=True)

            self.anpr_queue.task_done()

    # =================== GUI tick ===================
    def _gui_tick(self):
        with self.frame_lock:
            clean = self.frame_slots["clean"]
            ann = self.frame_slots["annotated"]
            anpr = self.frame_slots["anpr"]
            plate = self.frame_slots["plate"]

        if clean is not None:
            self._show(self.canvas_clean, clean)
        if ann is not None:
            self._show(self.canvas_annotated, ann)
        if anpr is not None:
            self._show(self.canvas_anpr, anpr)
        if plate is not None:
            self._show(self.canvas_plate, plate)

        # Traffic light LEDs + name
        ls = self.light_detector.current_state
        self._set_led(self.led_red, ls == "red")
        self._set_led(self.led_yel, ls == "yellow")
        self._set_led(self.led_grn, ls == "green")
        name_color = {
            "red": ACCENT_RED, "yellow": ACCENT_YEL, "green": ACCENT_GRN,
        }.get(ls, TEXT_MED)
        self.light_name.config(text=ls.upper(), fg=name_color)

        # Status dot — green if a frame was rendered recently, otherwise dim
        live = clean is not None
        self.status_dot.config(fg=ACCENT_GRN if live else ACCENT_RED)
        self.status_text.config(text="LIVE" if live else "WAITING",
                                fg=TEXT_HI if live else TEXT_MED)
        self.clock_label.config(text=datetime.now().strftime("%H:%M:%S"))

        # Drain GUI queue
        while True:
            try:
                msg = self.gui_queue.get_nowait()
            except queue.Empty:
                break
            self._add_violation(msg["tag"], msg["track_id"], msg["plate"])

        # Metrics
        self.metric_red.config(text=str(self.counts["REDLIGHT"]))
        self.metric_stop.config(text=str(self.counts["STOPLINE"]))
        self.metric_lane.config(text=str(self.counts["LANECHANGE"]))
        self.queue_label.config(text=f"queue {self.anpr_queue.qsize()}/20")
        self._render_tracking()
        if self.last_anpr_t > 0:
            ago = time.time() - self.last_anpr_t
            self.fps_label.config(text=f"anpr {ago:.1f}s ago")

        if self.running:
            self.root.after(33, self._gui_tick)

    def _render_tracking(self):
        """Refresh the tracking text panel from self.tracking_state."""
        now = time.time()
        with self.tracking_lock:
            # drop entries not seen in the last 2 seconds
            stale = [tid for tid, s in self.tracking_state.items()
                     if now - s.get("last_seen", 0) > 2.0]
            for tid in stale:
                del self.tracking_state[tid]
            snapshot = sorted(self.tracking_state.items())

        self.track_count_label.config(
            text=f"{len(snapshot)} vehicle{'s' if len(snapshot) != 1 else ''}")

        self.tracking_text.config(state="normal")
        self.tracking_text.delete("1.0", tk.END)
        if not snapshot:
            self.tracking_text.insert(tk.END, "no vehicles tracked\n", "dim")
        else:
            for tid, s in snapshot:
                self.tracking_text.insert(tk.END, f"#{tid:>3}", "id")
                prev = s.get("prev")
                prev_str = f"({prev[0]},{prev[1]})" if prev else "(—,—)"
                self.tracking_text.insert(
                    tk.END,
                    f"  c=({s['cx']},{s['cy']})  prev={prev_str}  "
                    f"hist={s['hist']:>2}  ",
                )
                sides = s.get("sides", [])
                flipped = s.get("flipped", [])
                if not sides:
                    self.tracking_text.insert(tk.END, "L?:n/a", "dim")
                else:
                    for i, side_text in enumerate(sides):
                        tag = "flip" if (i < len(flipped) and flipped[i]) else "dim"
                        self.tracking_text.insert(tk.END, side_text, tag)
                        if i < len(sides) - 1:
                            self.tracking_text.insert(tk.END, " ")
                self.tracking_text.insert(tk.END, "\n")
        self.tracking_text.config(state="disabled")

    def _show(self, canvas, frame_bgr):
        cw = canvas.winfo_width()
        ch = canvas.winfo_height()
        if cw < 10 or ch < 10:
            return
        h, w = frame_bgr.shape[:2]
        s = min(cw / w, ch / h)
        nw = max(1, int(w * s))
        nh = max(1, int(h * s))
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb).resize((nw, nh), Image.LANCZOS)
        photo = ImageTk.PhotoImage(img)
        canvas._photo = photo
        canvas.delete("all")
        canvas.create_image(
            (cw - nw) // 2, (ch - nh) // 2, anchor=tk.NW, image=photo)

    def _add_violation(self, tag, track_id, plate):
        ts = datetime.now().strftime("%H:%M:%S")
        col = TAG_COLOR.get(tag, TEXT_HI)
        label = TAG_LABEL.get(tag, tag)

        # Big violation card update
        self.type_badge.config(text=label, bg=col, fg=BG_BASE)
        self.plate_label.config(text=plate)
        self.vehicle_label.config(text=f"vehicle #{track_id}   time {ts}")

        # Log row
        self.tree.insert("", 0, values=(ts, label, f"#{track_id}", plate),
                         tags=(tag,))

        if tag in self.counts:
            self.counts[tag] += 1

    # =================== shutdown ===================
    def _on_close(self):
        if not self.running:
            return
        print("[INFO] Shutting down...", flush=True)
        self.running = False
        self.stop_event.set()
        try:
            self.anpr_queue.put_nowait(None)
        except queue.Full:
            pass
        self.main_capture.stop()
        if self.anpr_capture is not None:
            self.anpr_capture.stop()
        try:
            self.root.destroy()
        except tk.TclError:
            pass

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = StopLineGUI()
    app.run()
