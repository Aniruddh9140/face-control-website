"""EyeSync desktop: Face Controlled OS navigation + EyeSync speech."""
from __future__ import annotations

import argparse
import logging
from logging.handlers import RotatingFileHandler
import os
os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")
import queue
from pathlib import Path
import sys
import threading
import tkinter as tk
from tkinter import ttk

from PIL import Image, ImageTk

import desktop
import native_input
import voice_models as vm
import voice_typing as vt
import voice_commands as vc
from control_engine import Controller, ROOT
from dashboard_bridge import DashboardBridge
from voice_overlay import VoiceOverlay
import app_paths
from bundle_signals import EngineExit

BG, CARD, TEXT, MUTED, ACCENT = "#0b1020", "#151e32", "#edf3ff", "#a3b3ce", "#5ee3bf"
PHRASES = [
    "Open the on-screen keyboard", "Close the on-screen keyboard",
    "Increase the size of the on-screen keyboard", "Decrease the size of the on-screen keyboard",
    "Let me adjust the position of the on-screen keyboard",
    "Done adjusting the position of the on-screen keyboard",
]


class App:
    def __init__(self, root, camera=0, smoke=False, dashboard_ports=(8765, 8080), dashboard_test=False):
        self.root = root
        self.smoke = smoke
        self.dashboard_test = dashboard_test
        self.closing = False
        self.overlay = None
        self.pending = None
        self.dashboard_owned = False
        self.bridge = None
        self.engine_exit = None if smoke or dashboard_test else EngineExit()
        root.title("EyeSync (dashboard test)" if dashboard_test else "EyeSync")
        root.configure(bg=BG)
        root.geometry("1120x850")
        root.minsize(960, 740)
        icon = ROOT / "models" / "eyesync.ico"
        if icon.exists():
            root.iconbitmap(str(icon))
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.update_idletasks()
        self.controller = Controller((root.winfo_screenwidth(), root.winfo_screenheight()), camera)
        self.voice_overlay = VoiceOverlay(root, lambda: vt.request(self.controller.state, False, "voice bar"))
        self.photo = None
        self.hotkeys = None
        self._style()
        self._build()
        if not smoke and not dashboard_test:
            self.controller.start_voice_service()
            self.hotkeys = desktop.GlobalHotkeys({
                "F9": self.controller.pause,
                "F8": lambda: vt.request(self.controller.state, None, "global shortcut"),
                "F5": lambda: vt.request(self.controller.state, None, "global shortcut"),
            })
            self.hotkeys.start()
        if not smoke:
            self.bridge = DashboardBridge(self.controller, ports=dashboard_ports, test_mode=dashboard_test).start()
        self.tick()

    def _style(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background=BG)
        style.configure("Card.TFrame", background=CARD)
        style.configure("TLabel", background=CARD, foreground=TEXT, font=("Segoe UI", 10))
        style.configure("Muted.TLabel", foreground=MUTED, background=CARD)
        style.configure("TButton", font=("Segoe UI", 10), padding=(12, 8), background="#263650", foreground=TEXT, borderwidth=0)
        style.map("TButton", background=[("active", "#364e6e"), ("disabled", "#20293b")], foreground=[("disabled", "#8391a8")])
        style.configure("Accent.TButton", background=ACCENT, foreground="#082a25", font=("Segoe UI", 10, "bold"))
        style.map("Accent.TButton", background=[("active", "#a4f4de")])
        style.configure("TCheckbutton", background=CARD, foreground=MUTED, font=("Segoe UI", 10))
        style.map("TCheckbutton", background=[("active", CARD)], foreground=[("active", TEXT)])
        style.configure("TEntry", fieldbackground="#0b1425", foreground=TEXT, padding=9, insertcolor=TEXT)
        style.configure("TProgressbar", background=ACCENT, troughcolor="#253149", borderwidth=0, thickness=5)

    def card(self, parent, title, subtitle):
        frame = ttk.Frame(parent, style="Card.TFrame", padding=18)
        ttk.Label(frame, text=title, font=("Segoe UI", 14, "bold")).pack(anchor="w")
        ttk.Label(frame, text=subtitle, style="Muted.TLabel", wraplength=460).pack(anchor="w", pady=(5, 12))
        return frame

    def _build(self):
        host = tk.Canvas(self.root, bg=BG, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=host.yview)
        scrollbar.pack(side="right", fill="y")
        host.pack(side="left", fill="both", expand=True)
        host.configure(yscrollcommand=scrollbar.set)
        shell = ttk.Frame(host, padding=24)
        content = host.create_window((0, 0), window=shell, anchor="nw")
        shell.bind("<Configure>", lambda _e: host.configure(scrollregion=host.bbox("all")))
        host.bind("<Configure>", lambda e: host.itemconfigure(content, width=e.width))
        self.root.bind("<MouseWheel>", lambda e: host.yview_scroll(int(-e.delta / 120), "units"))
        header = tk.Frame(shell, bg=BG)
        header.pack(fill="x", pady=(0, 18))
        tk.Label(header, text="EYESYNC", bg=BG, fg=TEXT, font=("Segoe UI", 28, "bold")).pack(side="left")
        tk.Label(header, text="FACE  /  VOICE  /  CONTROL", bg=BG, fg=ACCENT, font=("Segoe UI", 10, "bold")).pack(side="right")
        body = ttk.Frame(shell)
        body.pack(fill="both", expand=True)
        body.columnconfigure(0, weight=1, uniform="columns")
        body.columnconfigure(1, weight=1, uniform="columns")
        body.rowconfigure(0, weight=1)
        left, right = ttk.Frame(body), ttk.Frame(body)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 9))
        right.grid(row=0, column=1, sticky="nsew", padx=(9, 0))

        camera = self.card(left, "Face navigation", "Head movement steers. Facial gestures click.")
        camera.pack(fill="both", expand=True)
        preview_box = tk.Frame(camera, bg="#080d18", height=180)
        preview_box.pack(fill="x")
        preview_box.pack_propagate(False)
        self.preview = tk.Label(preview_box, text="Camera is off\n\nStart when you are ready", bg="#080d18", fg=MUTED,
                                font=("Segoe UI", 13))
        self.preview.pack(fill="both", expand=True)
        self.camera_status = tk.StringVar(value="CAMERA OFF   ·   POINTER PAUSED")
        ttk.Label(camera, textvariable=self.camera_status, style="Muted.TLabel").pack(anchor="w", pady=10)
        row = ttk.Frame(camera, style="Card.TFrame")
        row.pack(fill="x")
        self.start_button = ttk.Button(row, text="Start camera", style="Accent.TButton", command=self.start_camera)
        self.start_button.pack(side="left")
        ttk.Button(row, text="Calibrate pointer", command=self.calibrate).pack(side="left", padx=8)
        ttk.Button(row, text="Stop", command=self.controller.stop_camera).pack(side="left")
        row2 = ttk.Frame(camera, style="Card.TFrame")
        row2.pack(fill="x", pady=(8, 0))
        self.enable_button = ttk.Button(row2, text="Train & enable pointer", command=self.enable_pointer)
        self.enable_button.pack(side="left")
        ttk.Button(row2, text="Pause pointer", command=self.controller.pause).pack(side="left", padx=8)
        ttk.Label(camera, text="Emergency pause: Ctrl + Alt + E", foreground=ACCENT).pack(anchor="w", pady=(12, 0))
        self.hint = tk.StringVar()
        ttk.Label(camera, textvariable=self.hint, wraplength=450, style="Muted.TLabel").pack(anchor="w", pady=(7, 0))
        self.quality = tk.StringVar()
        ttk.Label(camera, textvariable=self.quality, wraplength=450, style="Muted.TLabel").pack(anchor="w", pady=(5, 0))
        gestures = self.card(left, "Your gestures", "Relax your face between actions to rearm.")
        gestures.pack(fill="x", pady=(16, 0))
        self.bars = []
        for text in ["Big smile · 1.5 s → pause / resume", "Mouth wide open · 3 s → quit (close to cancel)", "Both eyes closed · 2.5 s → voice on / off"]:
            ttk.Label(gestures, text=text).pack(anchor="w", pady=(4, 2))
            bar = ttk.Progressbar(gestures, maximum=1)
            bar.pack(fill="x", pady=(0, 4))
            self.bars.append(bar)
        ttk.Label(gestures, text="Blink ~0.3 s: left click · ~1.3 s: right click\nWink left / right: left / right click\nMouth left: recenter · mouth right 3 s: calibrate\nTilt left / right: scroll up / down", wraplength=450,
                  style="Muted.TLabel").pack(anchor="w", pady=(8, 0))
        self.scroll_status = tk.StringVar(value="Scroll: tilt an ear toward its shoulder, hold; straighten to stop.")
        ttk.Label(gestures, textvariable=self.scroll_status, wraplength=450,
                  foreground=ACCENT).pack(anchor="w", pady=(8, 0))

        voice = self.card(right, "Voice typing", "EyeSync's offline recognition, punctuation and text editing.")
        voice.pack(fill="x")
        self.voice_status = tk.StringVar(value="Microphone off")
        ttk.Label(voice, textvariable=self.voice_status, font=("Segoe UI", 13, "bold"), foreground=ACCENT).pack(anchor="w")
        self.mic_bar = ttk.Progressbar(voice, maximum=1)
        self.mic_bar.pack(fill="x", pady=(8, 12))
        ttk.Label(voice, text="New line · new paragraph · delete that · comma\nVoice shortcut: Ctrl + Alt + Shift + F5", wraplength=450,
                  style="Muted.TLabel").pack(anchor="w", pady=(0, 8))
        self.accurate = tk.BooleanVar(value=vm.enabled())
        ttk.Checkbutton(voice, text="Accurate mode: fewer mistakes, but slow (7–30 s a phrase). "
                                    "Downloads 190 MB once. Speech still stays on this PC.",
                        variable=self.accurate, command=self.toggle_accurate).pack(anchor="w", pady=(0, 10))
        row = ttk.Frame(voice, style="Card.TFrame")
        row.pack(fill="x")
        self.voice_button = ttk.Button(row, text="Start voice typing", style="Accent.TButton", command=self.toggle_voice)
        self.voice_button.pack(side="left")
        ttk.Button(row, text="Stop listening", command=lambda: vt.request(self.controller.state, False, "stop button")).pack(side="left", padx=8)
        self.transcript = tk.StringVar(value="Click a text field in another app, then speak.\nSay “new line”, “delete that”, or “stop listening”.")
        ttk.Label(voice, textvariable=self.transcript, wraplength=450, style="Muted.TLabel").pack(anchor="w", pady=(12, 0))
        commands = self.card(right, "Voice command panel", "Speak or type an exact phrase. Commands use the same offline speech engine.")
        commands.pack(fill="both", expand=True, pady=(16, 0))
        self.command_text = tk.StringVar()
        entry = ttk.Entry(commands, textvariable=self.command_text, font=("Segoe UI", 10))
        entry.pack(fill="x")
        entry.bind("<Return>", lambda _e: self.run_command())
        self.command_text.trace_add("write", self.schedule_command)
        row = ttk.Frame(commands, style="Card.TFrame")
        row.pack(fill="x", pady=8)
        ttk.Button(row, text="Listen for command", command=self.listen_command).pack(side="left")
        ttk.Button(row, text="Run phrase", command=self.run_command).pack(side="left", padx=8)
        for phrase in PHRASES:
            button = tk.Button(commands, text=phrase, anchor="w", relief="flat", bg=CARD, fg=MUTED,
                               activebackground="#263650", activeforeground=TEXT, wraplength=450,
                               font=("Segoe UI", 9), cursor="hand2", command=lambda p=phrase: self.command_text.set(p))
            button.pack(fill="x", pady=2)
        self.command_status = tk.StringVar()
        ttk.Label(commands, textvariable=self.command_status, foreground=ACCENT, wraplength=450).pack(anchor="w", pady=(10, 0))
        tk.Label(shell, text="LOCAL PROCESSING    ·    Camera and microphone start on request    ·    No external mouse hardware",
                 bg=BG, fg=MUTED, font=("Segoe UI", 9)).pack(anchor="w", pady=(16, 0))

    def start_camera(self):
        if not self.smoke and not self.dashboard_test:
            self.dashboard_owned = False
            self.start_button.configure(state="disabled")
            self.controller.start()

    def calibrate(self):
        if not self.controller.state.read().running:
            self.controller.state.update(hint="Start the camera before calibrating.")
            return
        if self.overlay:
            return
        self.controller.request("calibrate")

    def enable_pointer(self):
        if self.smoke or self.dashboard_test:
            return
        self.controller.request_enable()
        if self.controller.state.read().pointer_mode == "untrained":
            self.calibrate()

    def show_dot(self, index, point):
        if self.overlay is None:
            self.overlay = tk.Toplevel(self.root)
            self.overlay.attributes("-fullscreen", True)
            self.overlay.attributes("-topmost", True)
            self.canvas = tk.Canvas(self.overlay, bg=BG, highlightthickness=0)
            self.canvas.pack(fill="both", expand=True)
            self.overlay.bind("<Escape>", lambda _e: self.cancel_calibration())
            self.overlay.protocol("WM_DELETE_WINDOW", self.cancel_calibration)
            self.overlay.update_idletasks()
        width, height = self.overlay.winfo_width(), self.overlay.winfo_height()
        x, y = point[0] * width, point[1] * height
        self.canvas.delete("all")
        self.canvas.create_text(width / 2, height * 0.28, text=f"POINTER CALIBRATION  {index + 1} / 9", fill=TEXT, font=("Segoe UI", 22, "bold"))
        self.canvas.create_text(width / 2, height * 0.72, text="Gently move your head toward the dot. Hold still until it moves.\nKeep eyes open and mouth relaxed. Esc cancels.",
                                fill=MUTED, font=("Segoe UI", 14), justify="center")
        self.canvas.create_oval(x - 24, y - 24, x + 24, y + 24, outline=ACCENT, width=2)
        self.canvas.create_oval(x - 8, y - 8, x + 8, y + 8, fill=ACCENT, outline=ACCENT)
        self.canvas.create_arc(x - 31, y - 31, x + 31, y + 31, start=90, extent=0,
                               style="arc", outline=ACCENT, width=5, tags="capture")
        self.canvas.create_text(width / 2, height * .63, text="Finding your face…", fill=ACCENT,
                                font=("Segoe UI", 14, "bold"), width=width * .8, tags="feedback")
        self.canvas.create_image(width * .80, height * .30, tags="camera")

    def cancel_calibration(self):
        self.controller.request("cancel_calibration")
        self.hide_calibration()

    def hide_calibration(self):
        if self.overlay is not None:
            self.overlay.destroy()
            self.overlay = None

    def toggle_accurate(self):
        vm.set_enabled(self.accurate.get())
        if self.accurate.get():  # also repairs a damaged model
            threading.Thread(target=self._fetch_accurate, name="voice-model", daemon=True).start()

    def _fetch_accurate(self):
        state = self.controller.state
        try:
            finished = vm.fetch(lambda percent: state.update(voice_hint=f"Downloading the Accurate voice model: {percent}%"))
            if finished and vm.enabled():
                state.update(voice_hint="Accurate mode is ready. It starts with your next voice typing.")
        except (OSError, ValueError) as exc:
            vm.set_enabled(False)  # the refresh loop unticks the box
            state.update(voice_hint=f"Accurate mode stayed off: {exc}")

    def toggle_voice(self):
        if not self.smoke and not self.dashboard_test:
            if self.controller.voice.active():
                vt.request(self.controller.state, False, "voice button")
            else:
                vt.request(self.controller.state, True, "voice button")

    def listen_command(self):
        if self.smoke or self.dashboard_test:
            return
        if self.controller.voice.active():
            self.controller.state.update(command_result="Stop the current listening session first, then listen for a command.")
            return
        vt.request(self.controller.state, True, "command panel")

    def schedule_command(self, *_args):
        if self.pending is not None:
            self.root.after_cancel(self.pending)
            self.pending = None
        if vc.wispr_command_action(self.command_text.get()) is not None:
            self.pending = self.root.after(1200, self.run_command)

    def run_command(self):
        if self.pending is not None:
            self.root.after_cancel(self.pending)
            self.pending = None
        phrase = self.command_text.get().strip()
        if self.smoke or self.dashboard_test:
            return
        if vc.wispr_command_action(phrase) is None:
            self.controller.state.update(command_result="No exact supported phrase matched.")
            return
        self.controller.request("command", phrase)
        self.command_text.set("")

    def tick(self):
        if self.closing:
            return
        if self.engine_exit and self.engine_exit.is_set():
            self.close()
            return
        self.controller.pump_idle_commands()
        snap = self.controller.state.read()
        if not self.smoke and not self.dashboard_test:
            self.voice_overlay.update(snap)
        self.hint.set(snap.hint)
        self.scroll_status.set({1: "Scrolling up — straighten your head to stop.",
                               -1: "Scrolling down — straighten your head to stop."}.get(
                                   snap.scrolling, "Scroll: tilt left / right about 22°, hold briefly; straighten to stop."))
        self.quality.set(snap.calibration_quality)
        self.command_status.set(snap.command_result)
        camera = f"{snap.fps:.0f} FPS" if snap.running else "CAMERA OFF"
        self.camera_status.set(f"{camera}   ·   {'FACE FOUND' if snap.face else 'NO FACE'}   ·   POINTER {'ON' if snap.enabled else 'PAUSED'}   ·   WEB {snap.dashboard_clients}")
        self.start_button.configure(state="disabled" if self.controller.thread and self.controller.thread.is_alive() else "normal")
        self.enable_button.configure(text="Enable pointer" if snap.calibrated or snap.pointer_mode == "public" else "Train & enable pointer",
                                     state="normal" if snap.running and self.overlay is None else "disabled")
        self.voice_status.set({"": "Microphone off", "starting": "Starting microphone…", "listening": "Listening…", "hearing": "Hearing speech…", "working": "Recognizing locally…", "error": "Voice needs attention"}.get(snap.voice, snap.voice))
        self.voice_button.configure(text="Stop voice typing" if snap.voice not in ("", "error") else "Start voice typing")
        self.mic_bar["value"] = snap.voice_level
        self.accurate.set(vm.enabled())
        if snap.voice_hint or snap.voice_text:
            self.transcript.set(snap.voice_hint or snap.voice_text)
        for bar, value in zip(self.bars, (snap.smile, snap.mouth, snap.eyes)):
            bar["value"] = value
        with self.controller.state.lock:
            frame = self.controller.state.preview
        if frame is not None:
            image = Image.fromarray(frame)
            image.thumbnail((440, 180))
            self.photo = ImageTk.PhotoImage(image)
            self.preview.configure(image=self.photo, text="", width=0, height=0)
            if self.overlay is not None:
                image.thumbnail((192, 108))
                self.calibration_photo = ImageTk.PhotoImage(image)
                self.canvas.itemconfigure("camera", image=self.calibration_photo)
        elif self.photo is not None:
            self.preview.configure(image="", text="Camera is off\n\nStart when you are ready", width=32, height=9)
            self.photo = None
        if self.overlay is not None:
            self.canvas.itemconfigure("feedback", text=snap.calibration_text,
                                      fill=ACCENT if snap.face else "#ffbc78")
            self.canvas.itemconfigure("capture", extent=-359 * snap.capture_progress)
        while True:
            try:
                kind, value = self.controller.events.get_nowait()
            except queue.Empty:
                break
            if kind == "calibration_point":
                self.show_dot(*value)
            elif kind in ("calibration_done", "stopped"):
                self.hide_calibration()
                current = self.controller.state.read()
                if kind == "calibration_done" and current.calib_step == "done" and current.mouse_requested and not current.paused:
                    self.controller.request_enable()
            elif kind.startswith("dashboard_"):
                self.dashboard_event(kind, value)
        self.root.after(100, self.tick)

    def dashboard_event(self, kind, value):
        if kind == "dashboard_activate":
            if self.dashboard_test:
                self.controller.state.update(hint="Dashboard test: connected; camera, microphone and input disabled.")
                return
            if self.controller.thread and self.controller.thread.is_alive() and self.controller.worker_stop.is_set():
                self.root.after(300, lambda: self.controller.events.put(("dashboard_activate", None))
                                if self.controller.state.read().dashboard_clients else None)
                return
            self.dashboard_owned = True
            self.controller.request_enable()
            self.controller.start()
        elif kind == "dashboard_disconnect" and self.dashboard_owned:
            self.controller.stop_camera()
            vt.request(self.controller.state, False, "dashboard disconnected")
            self.dashboard_owned = False
        elif kind == "dashboard_calibrate":
            if value == "cancel":
                self.cancel_calibration()
            else:
                self.calibrate()
        elif kind == "dashboard_show":
            self.root.deiconify()
            self.root.lift()
        elif kind == "dashboard_quit":
            self.close()

    def close(self):
        if self.closing:
            return
        self.closing = True
        self.hide_calibration()
        self.voice_overlay._hide()
        if self.hotkeys:
            self.hotkeys.stop()
        self.controller.pause()
        def shutdown():
            if self.bridge:
                self.bridge.close()
            self.controller.close()
        threading.Thread(target=shutdown, daemon=True).start()
        self.root.after(250, self._finish_close)

    def _finish_close(self):
        thread = self.controller.thread
        voice = self.controller.voice
        bridge = self.bridge
        if (bridge and bridge.thread and bridge.thread.is_alive()) or (thread and thread.is_alive()) or (voice and (voice.thread.is_alive() or voice.active())):
            self.root.after(100, self._finish_close)
        else:
            self.root.destroy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--ui-smoke", action="store_true", help="Show UI without camera, microphone, hotkeys or input")
    parser.add_argument("--smoke-seconds", type=float, default=0)
    parser.add_argument("--minimized", action="store_true")
    parser.add_argument("--ws-ports", type=int, nargs="+", default=[8765, 8080])
    parser.add_argument("--dashboard-test-mode", action="store_true", help="Test WebSocket/UI integration without camera, microphone, hotkeys or native input")
    args = parser.parse_args()
    (app_paths.DATA_DIR / "data").mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(Path(os.environ.get("EYESYNC_LOG", app_paths.DATA_DIR / "data" / "eyesync.log")), maxBytes=2_000_000, backupCount=2, encoding="utf-8")
    handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s [%(threadName)s] %(name)s: %(message)s"))
    logging.basicConfig(level=logging.INFO, handlers=[handler])
    instance = "EyeSync.Brahmastra.Desktop" + (".Test" if args.dashboard_test_mode else "")
    if not args.ui_smoke and not desktop.acquire_single_instance(instance):
        desktop.bring_window_to_front("EyeSync", "TkTopLevel")
        return 0
    native_input.make_dpi_aware()
    root = tk.Tk()
    app = App(root, args.camera, args.ui_smoke, args.ws_ports, args.dashboard_test_mode)
    if args.minimized:
        root.iconify()
    if args.smoke_seconds > 0 and (args.ui_smoke or args.dashboard_test_mode):
        root.after(int(args.smoke_seconds * 1000), app.close)
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
