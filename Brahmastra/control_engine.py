"""One camera, Face Controlled OS gestures, and EyeSync offline speech."""
from __future__ import annotations

import copy
import json
import logging
import queue
import threading
import time
from collections import deque
from datetime import date
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

import desktop
import face_control as fc
import face_engine as fe
import native_input
import voice_commands as vc
import voice_typing as vt
import app_paths
from improved_voice import ImprovedVoiceTyping
from gesture_adapter import EyeSyncGestures
from calibration import StableCapture
from public_pose import PoseRegressor, PublicPointerModel, pose_features, MODEL_PATH

ROOT = app_paths.APP_DIR
log = logging.getLogger("brahmastra")
POINTS = [(0.5, 0.5)] + [(x, y) for y in (0.12, 0.5, 0.88)
                         for x in (0.12, 0.5, 0.88) if (x, y) != (0.5, 0.5)]


@dataclass
class Snapshot:
    running: bool = False
    face: bool = False
    enabled: bool = False
    click_count: int = 0
    right_click_count: int = 0
    calib_step: str = ""
    voice: str = ""
    voice_hint: str = ""
    voice_text: str = ""
    voice_level: float = 0.0
    voice_left: float = 0.0
    hint: str = "Ready. Start the camera, then calibrate the pointer."
    command_result: str = "Six exact keyboard commands supported."
    fps: float = 0.0
    smile: float = 0.0
    mouth: float = 0.0
    eyes: float = 0.0
    baseline_ready: bool = False
    calibrated: bool = False
    calibrated_date: str = ""
    calibration_text: str = ""
    calibration_progress: float = 0.0
    capture_progress: float = 0.0
    calibration_quality: str = ""
    mouse_requested: bool = False
    paused: bool = False
    speed: int = 3
    click_mode: str = "blink"
    gestures: bool = True
    dashboard_clients: int = 0
    x: float = 0.0
    y: float = 0.0
    eyes_closed: bool = False
    pointer_mode: str = "untrained"
    scrolling: int = 0
    voice_mode: str = "dictation"
    voice_latency: float = 0.
    voice_device: str = ""


class State:
    """The small state contract required by the unchanged EyeSync voice engine."""
    def __init__(self):
        self.lock = threading.RLock()
        self.snap = Snapshot()
        self.stop = threading.Event()
        self.voice_requests = queue.SimpleQueue()
        self.preview = None

    def update(self, **fields):
        with self.lock:
            for name, value in fields.items():
                setattr(self.snap, name, value)

    def read(self):
        with self.lock:
            return copy.copy(self.snap)


class PointerModel:
    """Source project's whole-face features, ridge map and smooth correction.

    NumPy replaces scikit-learn/pickle so this works with EyeSync's Python.
    Only numeric coefficients and camera/screen metadata are persisted.
    """
    def __init__(self, mean, scale, coef, centers, residuals, camera, screen):
        self.mean, self.scale, self.coef = mean, scale, coef
        self.centers, self.residuals = centers, residuals
        self.camera, self.screen = tuple(camera), tuple(screen)
        self.validation_error = None

    @staticmethod
    def _ridge(x, y, strength):
        mean, scale = x.mean(axis=0), np.maximum(x.std(axis=0), 0.003)
        z = (x - mean) / scale
        design = np.c_[np.ones(len(x)), z]
        penalty = np.eye(5) * strength
        penalty[0, 0] = 0
        coef = np.linalg.solve(design.T @ design + penalty, design.T @ y)
        return mean, scale, z, design, coef

    @classmethod
    def fit(cls, features, targets, camera, screen):
        x, y = np.asarray(features, float), np.asarray(targets, float)
        if (x.shape != (9, 4) or y.shape != (9, 2)
                or not np.isfinite(x).all() or not np.isfinite(y).all()):
            raise ValueError("Calibration needs nine valid face positions.")
        if np.ptp(x[:, [0, 2]].mean(axis=1)) < 0.025 or np.ptp(x[:, [1, 3]].mean(axis=1)) < 0.025:
            raise ValueError("Move your head toward each dot; eye movement alone is not enough.")
        # Leave each target out, including its normalization statistics. Select
        # regularization on predictions of unseen dots, not training residuals.
        candidates = []
        for strength in (.02, .2, 2.0):
            errors = []
            for index in range(len(x)):
                keep = np.arange(len(x)) != index
                mean, scale, _, _, coef = cls._ridge(x[keep], y[keep], strength)
                predicted = np.r_[1, (x[index] - mean) / scale] @ coef
                errors.append(np.linalg.norm((predicted - y[index]) * screen) / min(screen))
            candidates.append((float(np.mean(errors)), strength, np.asarray(errors)))
        _, strength, errors = min(candidates, key=lambda item: item[0])
        if np.median(errors) > .16 or np.quantile(errors, .9) > .28:
            raise ValueError("Training positions were inconsistent. Retry with small, steady head movements toward each dot.")
        mean, scale, z, design, coef = cls._ridge(x, y, strength)
        residuals = y - design @ coef
        # A small smooth residual correction follows the source's hybrid design.
        model = cls(mean, scale, coef, z, residuals, camera, screen)
        model.validation_error = float(np.median(errors))
        err = np.linalg.norm((np.array([model.normalized(v) for v in x]) - y) * screen, axis=1)
        if float(np.median(err)) > min(screen) * 0.16:
            raise ValueError("Head positions did not match the dots. Please calibrate again slowly.")
        return model

    def normalized(self, features):
        z = np.clip((np.asarray(features) - self.mean) / self.scale, -4, 4)
        w = np.exp(-np.sum((self.centers - z) ** 2, axis=1) / 2)
        correction = (w @ self.residuals) / (w.sum() + 0.5)
        return np.clip(np.r_[1, z] @ self.coef + correction, 0, 1)

    def predict(self, features):
        return self.normalized(features) * (np.asarray(self.screen) - 1)

    def save(self, path):
        temporary = path.with_suffix(".tmp.npz")
        np.savez(temporary, version=1, mean=self.mean, scale=self.scale, coef=self.coef,
                 centers=self.centers, residuals=self.residuals, camera=self.camera, screen=self.screen,
                 validation_error=-1 if self.validation_error is None else self.validation_error)
        temporary.replace(path)

    @classmethod
    def load(cls, path, camera, screen):
        with np.load(path, allow_pickle=False) as d:
            if int(d["version"]) != 1 or tuple(d["camera"]) != tuple(camera) or tuple(d["screen"]) != tuple(screen):
                raise ValueError("Camera or display changed; recalibrate the pointer.")
            expected = {"mean": (4,), "scale": (4,), "coef": (5, 2), "centers": (9, 4), "residuals": (9, 2)}
            for key, shape in expected.items():
                if d[key].shape != shape or not np.isfinite(d[key]).all():
                    raise ValueError("Invalid calibration; please calibrate again.")
            if np.any(d["scale"] <= 0):
                raise ValueError("Invalid calibration scale.")
            model = cls(*(d[k].copy() for k in expected), camera, screen)
            if "validation_error" in d:
                error = float(d["validation_error"])
                if np.isfinite(error) and error >= 0:
                    model.validation_error = error
            return model


def measurements(xyz, width, height):
    """Exact Face Controlled OS smile, inner-mouth and 3D eye geometry."""
    xyz = np.asarray(xyz, dtype=np.float32)[:468]
    if xyz.shape != (468, 3) or not np.isfinite(xyz).all():
        raise ValueError("Invalid face landmarks")
    points = xyz * np.array([width, height, width], np.float32)
    eye_l = (points[33, :2] + points[133, :2]) * 0.5
    eye_r = (points[362, :2] + points[263, :2]) * 0.5
    iod = float(np.linalg.norm(eye_l - eye_r)) + 1e-6
    smile = float(abs(points[61, 0] - points[291, 0])) / iod
    mouth = float(abs(points[13, 1] - points[14, 1])) / iod
    low, high = xyz[:, :2].min(axis=0), xyz[:, :2].max(axis=0)
    return (np.clip(np.r_[low, high], 0, 1), smile, mouth,
            fc._eye_aspect_ratio(points, "left"), fc._eye_aspect_ratio(points, "right"))


class RoutedVoiceTyping(ImprovedVoiceTyping):
    """Full EyeSync speech with cancellation guards and separate panel routing."""
    def __init__(self, state, commands, **kwargs):
        super().__init__(state, **kwargs)
        self.commands = commands
        self.command_mode = False

    def _start(self, source):
        self.command_mode = source == "command panel"
        self.state.update(voice_mode="command" if self.command_mode else "dictation")
        super()._start(source)

    def _deliver_phrase(self, text, target, started):
        if not self.command_mode:
            return super()._deliver_phrase(text, target, started)
        phrase = vt.clean(text)
        if not phrase:
            return False
        self.state.update(voice_text=phrase)
        if vc.wispr_command_action(phrase) is not None:
            self.commands.put(("command", phrase))
            return True
        if vt.normalise(phrase) in vt.STOP_PHRASES:
            return True
        self.state.update(command_result="No exact command matched: " + phrase)
        return False


class Controller:
    def __init__(self, screen, camera=0, mouse_factory=native_input.LocalMouseController, gesture_profile="eyesync"):
        self.gesture_profile = gesture_profile
        self.state = State()
        self.screen = tuple(screen)
        self.state.update(x=screen[0] / 2, y=screen[1] / 2)
        self.camera = camera
        self.mouse_factory = mouse_factory
        self.commands = queue.Queue()
        self.events = queue.Queue()
        self.thread = None
        self.worker_stop = threading.Event()
        self.emergency = threading.Event()
        self.enable_requested = threading.Event()
        self.voice = None
        self.model_path = app_paths.DATA_DIR / "data" / "pointer.npz"
        self.mouse = None
        self.model = None
        self.dragging = False
        self.clicks = True
        self._reset_detectors()

    def _reset_detectors(self):
        self.ai_gestures = EyeSyncGestures()
        self.smile = fc.LandmarkSmileClickDetector()
        self.mouth = fc.MouthOpenClickDetector()
        self.eyes = fc.BilateralEyeClosureDetector()
        self.must_relax = True
        self.relax_since = None
        self.ema = fc.VectorEma(0.35)
        self.smoother = fc.OneEuroFilter2D(1.2, 0.015)
        self.last_xy = None
        self.was_frozen = False
        self.pointer_offset = np.zeros(2)
        self.last_features = None
        self.dwell_anchor = None
        self.dwell_since = None
        self.dwell_fired = False
        self.position_history = deque(maxlen=120)
        self.tilt_held = False

    def start_voice_service(self):
        if self.voice is None:
            self.voice = RoutedVoiceTyping(self.state, self.commands).start()

    def start(self):
        if self.thread is not None and self.thread.is_alive():
            return
        self.worker_stop.clear()
        self.emergency.clear()
        self.thread = threading.Thread(target=self._run, name="face-control", daemon=True)
        self.thread.start()

    def stop_camera(self):
        self.enable_requested.clear()
        self.emergency.set()
        self.worker_stop.set()
        self.state.update(enabled=False, mouse_requested=False)

    def pause(self):
        self.enable_requested.clear()
        self.emergency.set()
        self.dwell_anchor = self.dwell_since = None
        self.dwell_fired = False
        self.was_frozen = True
        self.state.update(enabled=False, paused=True, scrolling=0, hint="Pointer paused. Resume explicitly when ready.")

    def request_enable(self):
        self.state.update(mouse_requested=True, paused=False,
                          hint="Calibrate your head pointer before control can start." if self.model is None
                          else "Pointer requested. Keep your face visible while baselines settle.")
        self.emergency.clear()
        self.enable_requested.set()

    def request(self, name, value=None):
        if name == "enable":
            self.request_enable()
        self.commands.put((name, value))

    def _release_drag(self):
        if self.dragging and self.mouse is not None:
            self.mouse.button_up("left")
            self.dragging = False

    def _execute_command(self, phrase):
        action = vc.wispr_command_action(phrase)
        if action is None:
            self.state.update(command_result="No supported command matched; nothing ran.")
            return
        try:
            if action == "open_keyboard":
                vc.open_onscreen_keyboard()
                result = "On-screen keyboard opened."
            elif action == "close_keyboard":
                self._release_drag()
                result = "Keyboard closed." if vc.close_onscreen_keyboard() else "Keyboard is already closed."
            elif action in ("increase_keyboard", "reduce_keyboard"):
                changed = vc.resize_onscreen_keyboard(1.2 if action == "increase_keyboard" else 1 / 1.2)
                result = "Keyboard resized." if changed else "Open the on-screen keyboard first."
            elif action == "start_adjusting":
                if not self.state.read().enabled or self.mouse is None:
                    raise ValueError("Enable the calibrated pointer first.")
                if not vc.is_keyboard_titlebar_cursor_position():
                    raise ValueError("Move the pointer onto the keyboard title bar first.")
                if not self.dragging:
                    self.mouse.button_down("left")
                    self.dragging = True
                result = "Dragging keyboard. Say Done adjusting the position of the on-screen keyboard to release."
            else:
                self._release_drag()
                result = "Keyboard drag released."
            self.state.update(command_result=result)
        except (OSError, ValueError) as exc:
            self.state.update(command_result=str(exc))

    def pump_idle_commands(self):
        """Process panel commands even when the camera is off; called by UI."""
        if self.thread is None or not self.thread.is_alive():
            while True:
                try:
                    name, value = self.commands.get_nowait()
                except queue.Empty:
                    break
                if name == "command":
                    self._execute_command(value)
                elif name == "enable":
                    self.state.update(hint="Start the camera and calibrate before enabling the pointer.")

    def _run(self):
        cap = mesh = None
        try:
            self._reset_detectors()
            self.mouse = self.mouse_factory()
            cap = fe.open_camera(self.camera, width=320, height=240)
            camera_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            mesh = fe.FaceMeshBackend()
            self.model = None
            if self.model_path.exists():
                try:
                    self.model = PointerModel.load(self.model_path, camera_size, self.screen)
                except (ValueError, OSError, KeyError) as exc:
                    self.state.update(hint=str(exc))
            personal = self.model is not None
            if self.model is None and MODEL_PATH.exists():
                try:
                    self.model = PublicPointerModel(PoseRegressor.load(), self.screen)
                except (ValueError, OSError, KeyError) as exc:
                    log.warning("Public model could not load: %s", exc)
            self.state.update(running=True, enabled=False, calibrated=personal,
                              pointer_mode="personal" if personal else "public" if self.model is not None else "untrained",
                              calibrated_date=date.fromtimestamp(self.model_path.stat().st_mtime).isoformat() if personal else "",
                              calibration_quality=(f"Withheld-dot median error: {self.model.validation_error * 100:.1f}% of shorter screen edge (training check)."
                                                   if self.model and self.model.validation_error is not None else ""),
                              hint="Keep eyes open and mouth relaxed while baselines settle.")
            self.events.put(("started", None))
            calibration = None
            features = []
            point_index = 0
            last_frame = time.monotonic()
            fps_frames, fps_time = 0, last_frame
            while not self.worker_stop.is_set() and not self.state.stop.is_set():
                if self.emergency.is_set() or native_input.hotkey_kill_pressed():
                    self.emergency.clear()
                    self.enable_requested.clear()
                    self.state.update(enabled=False, paused=True)
                    self._release_drag()
                    self.must_relax = True
                while True:
                    try:
                        name, value = self.commands.get_nowait()
                    except queue.Empty:
                        break
                    if name == "enable":
                        # request_enable owns the pending intent; _track enables
                        # only after baselines settle. A queued command must not
                        # undo an emergency pause or bypass that readiness gate.
                        if self.enable_requested.is_set() and self.model is None:
                            self.state.update(hint="Choose Train & enable pointer to learn your head movements.")
                    elif name == "recenter":
                        if self.model is not None and self.last_features is not None:
                            if isinstance(self.model, PublicPointerModel):
                                try:
                                    self.model.recenter(self.last_features)
                                except ValueError as exc:
                                    self.state.update(hint=str(exc))
                                    continue
                            self.pointer_offset = np.asarray(self.screen) / 2 - self.model.predict(self.last_features)
                            self.ema = fc.VectorEma(0.35)
                            self.smoother = fc.OneEuroFilter2D(1.2, 0.015)
                            self.last_xy = None
                            self.state.update(hint="Pointer centered on your current head position.")
                    elif name == "calibrate":
                        self._release_drag()
                        self.enable_requested.clear()
                        self.state.update(enabled=False, calib_step="starting", calibration_text="Follow each dot with your head.",
                                          calibration_progress=0, capture_progress=0, hint="Follow each dot with your head.")
                        vt.request(self.state, False, "calibration")
                        calibration, features = StableCapture(time.monotonic()), []
                        point_index = 0
                        self.events.put(("calibration_point", (0, POINTS[0])))
                    elif name == "cancel_calibration":
                        calibration = None
                        self.state.update(calib_step="", enabled=False, calibration_text="Calibration cancelled.",
                                          calibration_progress=0, capture_progress=0, hint="Calibration cancelled. Previous model kept.")
                        self.events.put(("calibration_done", None))
                    elif name == "command":
                        self._execute_command(value)
                ok, frame = cap.read(timeout=0.3)
                now = time.monotonic()
                if calibration and calibration.expired(now):
                    calibration = None
                    self.state.update(calib_step="", enabled=False, capture_progress=0,
                                      calibration_text="Training timed out. No new profile saved.",
                                      hint="Training timed out waiting for a steady, visible face. Check camera and lighting, then retry. Previous model kept.")
                    self.events.put(("calibration_done", None))
                if not ok or frame is None:
                    self._lost_face()
                    if calibration:
                        calibration.missing(now)
                        self.state.update(calibration_text=calibration.message, capture_progress=0)
                    if now - last_frame > 4:
                        raise RuntimeError("Camera stopped delivering frames. Stop and restart the camera.")
                    continue
                last_frame = now
                mesh.process(frame)
                xyz = mesh.last_xyz
                if xyz is None:
                    self._lost_face()
                    if calibration:
                        calibration.missing(now)
                        self.state.update(calibration_text=calibration.message, capture_progress=0)
                else:
                    values = measurements(xyz, frame.shape[1], frame.shape[0])
                    self.state.update(face=True)
                    if calibration:
                        captured = calibration.update(values[0], now)
                        self.state.update(calib_step="centre" if point_index == 0 else "right",
                                          calibration_text=f"Position {point_index + 1}/9: {calibration.message}",
                                          capture_progress=calibration.progress,
                                          calibration_progress=(point_index + calibration.progress) / 9)
                        if captured is not None:
                            features.append(captured)
                            point_index += 1
                            if point_index == len(POINTS):
                                saved = False
                                try:
                                    new_model = PointerModel.fit(features, POINTS, camera_size, self.screen)
                                    new_model.save(self.model_path)
                                    self.model = new_model
                                    saved = True
                                    self.state.update(calibrated=True, pointer_mode="personal", calibrated_date=date.today().isoformat(),
                                                      calibration_text="Head calibration saved.", calibration_progress=1,
                                                      calibration_quality=f"Withheld-dot median error: {new_model.validation_error * 100:.1f}% of shorter screen edge (training check).",
                                                      hint="Training saved. Enable pointer when ready.")
                                except (ValueError, OSError) as exc:
                                    self.state.update(hint=str(exc))
                                calibration = None
                                self.state.update(calib_step="done" if saved else "")
                                self.events.put(("calibration_done", None))
                                self._reset_detectors()
                            else:
                                calibration = StableCapture(now)
                                self.events.put(("calibration_point", (point_index, POINTS[point_index])))
                    else:
                        if isinstance(self.model, PublicPointerModel):
                            try:
                                values = (pose_features(xyz, frame.shape[1], frame.shape[0]), *values[1:])
                            except ValueError:
                                self._lost_face()
                                continue
                        landmarks = xyz[:, :2]
                        nose, face_px = fe.nose_pointer(landmarks, frame.shape[1], frame.shape[0])
                        gesture_frame = dict(ears=fe.eye_aspect_ratios(landmarks, frame.shape[1], frame.shape[0]),
                            scores=mesh.last_blendshapes, pose=mesh.last_pose,
                            mar=fe.mouth_aspect_ratio(landmarks, frame.shape[1], frame.shape[0], mesh.last_pose),
                            mouth_shift=fe.mouth_shift(landmarks, frame.shape[1], frame.shape[0]),
                            roll=fe.head_roll(landmarks, frame.shape[1], frame.shape[0]),
                            winks_ok=fe.eyes_level_ratio(landmarks, frame.shape[1], frame.shape[0]) > .7,
                            head=nose / face_px)
                        self._track(values, now, gesture_frame)
                # Latest frame only; the GUI never blocks the camera thread.
                with self.state.lock:
                    self.state.preview = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
                fps_frames += 1
                if now - fps_time > 1:
                    self.state.update(fps=fps_frames / (now - fps_time))
                    fps_frames, fps_time = 0, now
        except Exception as exc:
            log.exception("Tracking stopped")
            self.state.update(hint=str(exc))
        finally:
            try:
                self._release_drag()
            except OSError:
                log.exception("Could not release dragged mouse button")
            # Driver/backend cleanup can fail independently. Always attempt
            # both resources and publish stopped state even when one fails.
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    log.exception("Could not release camera")
            if mesh is not None:
                try:
                    mesh.close()
                except Exception:
                    log.exception("Could not close face tracking backend")
            self.mouse = None
            self.state.update(running=False, enabled=False, face=False, calib_step="", baseline_ready=False, scrolling=0)
            with self.state.lock:
                self.state.preview = None
            self.events.put(("stopped", None))

    def _lost_face(self):
        self.ai_gestures.lost_face()
        if isinstance(self.model, PublicPointerModel):
            self.model.cancel_hold()
        self.smile.reset()
        self.mouth.reset()
        self.eyes.cancel_hold()
        self.eyes._open_since = None
        self.must_relax = True
        self.relax_since = None
        self.was_frozen = True
        self.dwell_anchor = self.dwell_since = None
        self.dwell_fired = False
        self.position_history.clear()
        self.tilt_held = False
        self.state.update(face=False, smile=0, mouth=0, eyes=0, eyes_closed=False, scrolling=0)
        self._release_drag()

    def _track(self, values, now, gesture_frame=None):
        bbox, width, gap, left, right = values
        self.last_features = bbox
        pointer_ready = self.model is not None
        if isinstance(self.model, PublicPointerModel):
            pointer_ready = self.model.observe(bbox, now)
            self.state.update(calibrated=self.model.ready)
            if not pointer_ready:
                self._release_drag()
                current = self.state.read()
                self.state.update(enabled=False)
                if current.mouse_requested and not current.paused and not self.emergency.is_set():
                    self.enable_requested.set()
                self.state.update(hint=f"Public model: face the screen and hold your head steady for 2 seconds ({self.model.progress * 100:.0f}%).")
            elif not self.state.read().calibrated or self.enable_requested.is_set():
                self.state.update(hint="Public-data head control ready. Recenter resets your neutral position; nine-dot training is optional.")
        gesture_ready = self.ai_gestures.ready if self.gesture_profile == "eyesync" else self.eyes.ready and self.smile.ready
        if self.enable_requested.is_set() and pointer_ready and gesture_ready:
            self.state.update(enabled=True, paused=False, hint="Pointer enabled. Ctrl+Alt+E pauses immediately.")
            self.enable_requested.clear()
        snap = self.state.read()
        if self.gesture_profile == "eyesync":
            speaking = snap.voice not in ("", "error")
            g = self.ai_gestures.update(left, right, now, gesture_frame,
                    gestures=snap.gestures, active=snap.enabled and pointer_ready and not snap.paused and not self.dragging and not self.emergency.is_set(),
                    speaking=speaking)
            tilted = g['tilted']
            if tilted and not self.tilt_held and snap.enabled and not self.emergency.is_set():
                # Match the old AI: undo sideways motion while the user leaned
                # into scrolling, using the last position with a straight head.
                calm_at = self.ai_gestures.scroll.calm_at
                if self.position_history and now-calm_at <= 1.:
                    # The five-frame tilt median lags the actual lean. Include
                    # the old AI's 130 ms click-rewind interval to avoid keeping
                    # cursor drift from those first frames in the anchor.
                    earlier = [p for p in self.position_history if p[0] <= calm_at - .13]
                    _, x, y = earlier[-1] if earlier else self.position_history[0]
                    self.mouse.move(int(x), int(y))
                    self.last_xy = np.array([x, y])
                    self.state.update(x=float(x), y=float(y))
                self._release_drag()
            self.tilt_held = tilted
            self.state.update(scrolling=g['scroll_direction'])
            if g['action'] == 'smile' and snap.calibrated:
                if snap.paused or not snap.enabled:
                    self.request_enable()
                else:
                    self.pause()
            elif g['action'] == 'mouth_left':
                self.request('recenter')
            elif g['action'] == 'mouth_right':
                self.events.put(('dashboard_calibrate', 'head'))
            if g['quit']:
                self.pause()
                self.events.put(('dashboard_quit', None))
            if g['voice'] and snap.calibrated:
                vt.request(self.state, None, 'eyes closed')
            if g['wheel'] and self.mouse is not None and snap.enabled and not self.emergency.is_set():
                self.mouse.scroll(g['wheel'])
            if g['hint']:
                self.state.update(hint=g['hint'])
            left_click = g['click'] == 'left' and snap.click_mode == 'blink' and snap.gestures
            right_click = g['click'] == 'right' and snap.click_mode == 'blink' and snap.gestures
            frozen = g['frozen']
            ready = self.ai_gestures.ready
            self.must_relax = not self.ai_gestures.expressions.relaxed
            self.state.update(smile=g['progress'] if self.ai_gestures.expressions.active else 0.,
                              mouth=g['progress'] if self.ai_gestures.close.gesture_since is not None else 0.,
                              eyes=g['progress'] if g['closed'] else 0., eyes_closed=g['closed'], baseline_ready=ready)
        else:
            eye_toggle = self.eyes.update(left, right, now)
            speaking = snap.voice not in ("", "error")
            # Speaking must never become a held-mouth right click.
            if speaking or not snap.enabled or self.must_relax:
                self.smile.reset()
                self.mouth.reset()
                if not self.smile.ready and not speaking:
                    self.smile.update(width, gap, self.mouth.release_threshold, now)
                if gap < self.mouth.release_threshold and not self.eyes.active:
                    if self.relax_since is None:
                        self.relax_since = now
                    if now - self.relax_since >= 0.4:
                        self.must_relax = False
                else:
                    self.relax_since = None
                left_click = right_click = False
            else:
                right_click = self.mouth.update(gap, now) if self.clicks and snap.gestures else False
                left_click = self.smile.update(width, gap, self.mouth.release_threshold, now) if self.clicks and snap.gestures and snap.click_mode == "blink" else False
                if not snap.gestures:
                    self.smile.reset()
                    self.mouth.reset()
                if self.mouth.active or self.eyes.active:
                    self.smile.reset()
                    left_click = False
                if self.eyes.active:
                    self.mouth.reset()
                    right_click = False
            if eye_toggle and snap.gestures and snap.calibrated:
                vt.request(self.state, None, "eyes closed")
            frozen = self.eyes.active or (not speaking and (self.smile.active or self.mouth.active))
            ready = self.eyes.ready and self.smile.ready
            self.state.update(smile=self.smile.hold_progress, mouth=self.mouth.hold_progress,
                              eyes=self.eyes.hold_progress, eyes_closed=self.eyes.active,
                              baseline_ready=ready)
        if ready and not snap.baseline_ready and self.model is None:
            self.state.update(hint="Face baseline ready. Calibrate your head pointer from the dashboard or EyeSync window.")
        if not pointer_ready or not snap.enabled or self.emergency.is_set():
            self.dwell_anchor = self.dwell_since = None
            self.dwell_fired = False
            self.was_frozen = True
            return
        if not frozen:
            if self.was_frozen:
                self.ema = fc.VectorEma(0.35)
                self.smoother = fc.OneEuroFilter2D(1.2, 0.015)
                if self.last_xy is not None:
                    self.smoother.update(*self.last_xy, now)
            raw = self.model.predict(self.ema.update(bbox)) + self.pointer_offset
            center = np.asarray(self.screen) / 2
            raw = center + (raw - center) * {1: 0.65, 2: 0.8, 3: 1., 4: 1.2, 5: 1.4}[snap.speed]
            raw = np.clip(raw, 0, np.asarray(self.screen) - 1)
            xy = self.smoother.update(*raw, now)
            self.state.update(x=float(xy[0]), y=float(xy[1]))
            # Small targets need finer motion than the source's fixed 10 pixels.
            deadzone = float(np.clip(min(self.screen) * .003, 2, 5))
            if self.last_xy is None or np.linalg.norm(np.asarray(xy) - self.last_xy) >= deadzone:
                self.mouse.move(int(xy[0]), int(xy[1]))
                self.last_xy = np.asarray(xy)
            actual = self.last_xy if self.last_xy is not None else np.asarray(xy)
            self.position_history.append((now, float(actual[0]), float(actual[1])))
        self.was_frozen = frozen
        if snap.click_mode == "dwell" and not frozen and not speaking and not self.must_relax:
            pos = np.array([self.state.read().x, self.state.read().y])
            if self.dwell_anchor is None or np.linalg.norm(pos - self.dwell_anchor) > 28:
                self.dwell_anchor, self.dwell_since, self.dwell_fired = pos, now, False
            elif not self.dwell_fired and now - self.dwell_since >= 1.2:
                left_click = True
                self.dwell_fired = True
        else:
            self.dwell_anchor = self.dwell_since = None
            self.dwell_fired = False
        if not self.dragging and (right_click or left_click):
            self.mouse.click("right" if right_click else "left")
            field = "right_click_count" if right_click else "click_count"
            self.state.update(**{field: getattr(snap, field) + 1})

    def close(self):
        self.pause()
        self.state.stop.set()
        self.worker_stop.set()
        if self.thread is not None:
            self.thread.join(5)
        if self.voice is not None:
            self.voice.close()

