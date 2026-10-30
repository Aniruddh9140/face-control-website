"""Face Controlled OS algorithms transplanted into Brahmastra.
See docs/INTEGRATION.md for provenance and compatibility changes.
"""
from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Optional
import numpy as np
MOUTH_OPEN_THRESHOLD = 0.142
MOUTH_OPEN_RELEASE_THRESHOLD = 0.109
EYE_MESH = {
    "right": ((33, 133), ((160, 144), (159, 145), (158, 153))),
    "left": ((362, 263), ((385, 380), (386, 374), (387, 373))),
}


@dataclass
class FrameFeatures:
    features: np.ndarray      # (4,) float32 [min_x, min_y, max_x, max_y], each in [0, 1]
    ear: float                # average Eye Aspect Ratio of both eyes
    ear_right: float = 0.0    # subject's right eye
    ear_left: float = 0.0     # subject's left eye
    smile: float = 0.0        # mouth-corner width / inter-ocular distance
    smile_features: Optional[np.ndarray] = None
    mouth: float = 0.0        # 3D outer-lip gap / 3D inter-ocular distance
    mouth_patch: Optional[np.ndarray] = None
    lm: Optional[np.ndarray] = None   # (468,3) landmarks centred on the nose, scaled by iod


def _eye_aspect_ratio(pts3d: np.ndarray, eye: str) -> float:
    # pts3d is in PIXEL units (x*w, y*h, z*w); normalised coords would distort the ratio.
    (c1, c2), pairs = EYE_MESH[eye]
    horizontal = np.linalg.norm(pts3d[c1] - pts3d[c2])
    vertical = np.mean([np.linalg.norm(pts3d[u] - pts3d[l]) for u, l in pairs])
    return float(vertical / (horizontal + 1e-9))


class VectorEma:
    """EMA over a feature vector; smooths landmark noise before the network sees it."""

    def __init__(self, alpha: float = 0.3) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = alpha
        self._state: Optional[np.ndarray] = None

    def update(self, v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, dtype=np.float64)
        self._state = v if self._state is None else self.alpha * v + (1 - self.alpha) * self._state
        return self._state.astype(np.float32)


class OneEuroFilter2D:
    """One-Euro filter (Casiez et al.): heavy smoothing when the point is nearly
    still (kills jitter), lighter smoothing when it moves fast (low lag).
    min_cutoff: lower = steadier at rest.  beta: higher = less lag when moving."""

    def __init__(self, min_cutoff: float = 0.5, beta: float = 0.004,
                 d_cutoff: float = 1.0) -> None:
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self._x: Optional[np.ndarray] = None
        self._dx = np.zeros(2)
        self._t: Optional[float] = None

    @staticmethod
    def _alpha(dt: float, cutoff: float) -> float:
        tau = 1.0 / (2.0 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def update(self, x: float, y: float, t: float) -> tuple[float, float]:
        p = np.array([x, y], dtype=np.float64)
        if self._x is None or self._t is None:
            self._x, self._t = p, t
            return float(p[0]), float(p[1])
        dt = max(t - self._t, 1e-3)
        dx = (p - self._x) / dt
        a_d = self._alpha(dt, self.d_cutoff)
        self._dx = a_d * dx + (1 - a_d) * self._dx
        cutoff = self.min_cutoff + self.beta * float(np.linalg.norm(self._dx))
        a = self._alpha(dt, cutoff)
        self._x = a * p + (1 - a) * self._x
        self._t = t
        return float(self._x[0]), float(self._x[1])


class LandmarkSmileClickDetector:
    """A sustained closed-mouth smile, measured from face landmarks, clicks once."""

    HOLD_S = 1.2
    BASELINE_SAMPLES = 24
    SMILE_RATIO = 1.15
    RELEASE_RATIO = 1.08

    def __init__(self) -> None:
        self._baseline_samples: list[float] = []
        self._baseline: Optional[float] = None
        self._since: Optional[float] = None
        self._fired = False
        self.last_ratio = 1.0

    @property
    def active(self) -> bool:
        return self._since is not None

    @property
    def ready(self) -> bool:
        return self._baseline is not None

    @property
    def hold_progress(self) -> float:
        if self._since is None:
            return 0.0
        return float(np.clip(
            (time.monotonic() - self._since) / self.HOLD_S, 0.0, 1.0
        ))

    def reset(self) -> None:
        self._since = None
        self._fired = False

    def update(self, mouth_width: float, mouth_open: float,
               max_mouth_open: float, now: Optional[float] = None) -> bool:
        now = time.monotonic() if now is None else now
        if (not np.isfinite(mouth_width) or mouth_width <= 0
                or not np.isfinite(mouth_open) or not np.isfinite(max_mouth_open)):
            self.reset()
            return False

        if self._baseline is None:
            if mouth_open < max_mouth_open:
                self._baseline_samples.append(float(mouth_width))
                if len(self._baseline_samples) >= self.BASELINE_SAMPLES:
                    self._baseline = float(np.median(self._baseline_samples))
                    self._baseline_samples.clear()
            return False

        self.last_ratio = float(mouth_width / self._baseline)
        if mouth_open >= max_mouth_open:
            self.reset()
            return False

        if self._since is None:
            if self.last_ratio >= self.SMILE_RATIO:
                self._since = now
                self._fired = False
            elif self.last_ratio <= self.RELEASE_RATIO:
                self._baseline = (
                    0.998 * self._baseline + 0.002 * float(mouth_width)
                )
            return False

        if self.last_ratio < self.RELEASE_RATIO:
            self.reset()
            return False

        if not self._fired and now - self._since >= self.HOLD_S:
            self._fired = True
            return True
        return False


class BilateralEyeClosureDetector:
    """Recognize a sustained bilateral eye closure using personal EAR baselines."""

    HOLD_S = 1.0
    RELEASE_S = 0.25
    AMBIGUOUS_GAP_S = 0.20
    BASELINE_SAMPLES = 30
    BASELINE_MIN_EAR = 0.05
    CLOSE_RATIO = 0.84
    OPEN_RATIO = 0.90
    BASELINE_ALPHA = 0.03

    def __init__(self, profile: Optional[dict] = None) -> None:
        self._baseline_samples_l: list[float] = []
        self._baseline_samples_r: list[float] = []
        self._baseline_l: Optional[float] = None
        self._baseline_r: Optional[float] = None
        self.close_ratio_l = float(
            profile["close_ratio_left"] if profile else self.CLOSE_RATIO
        )
        self.open_ratio_l = float(
            profile["open_ratio_left"] if profile else self.OPEN_RATIO
        )
        self.close_ratio_r = float(
            profile["close_ratio_right"] if profile else self.CLOSE_RATIO
        )
        self.open_ratio_r = float(
            profile["open_ratio_right"] if profile else self.OPEN_RATIO
        )
        self._since: Optional[float] = None
        self._open_since: Optional[float] = None
        self._gap_since: Optional[float] = None
        self._latched = False
        self.last_left = 1.0
        self.last_right = 1.0

    @property
    def ready(self) -> bool:
        return self._baseline_l is not None and self._baseline_r is not None

    @property
    def baseline_ears(self) -> Optional[tuple[float, float]]:
        if not self.ready:
            return None
        return self._baseline_l, self._baseline_r

    @property
    def active(self) -> bool:
        return self._since is not None or self._latched

    @property
    def awaiting_reopen(self) -> bool:
        return self._latched

    @property
    def hold_progress(self) -> float:
        if self._since is None:
            return 0.0
        return float(np.clip(
            (time.monotonic() - self._since) / self.HOLD_S, 0.0, 1.0
        ))

    def cancel_hold(self) -> None:
        self._since = None
        self._gap_since = None

    def update(self, ear_left: float, ear_right: float,
               now: Optional[float] = None) -> bool:
        now = time.monotonic() if now is None else now
        if (not np.isfinite(ear_left) or ear_left <= 0
                or not np.isfinite(ear_right) or ear_right <= 0):
            self.cancel_hold()
            return False

        if not self.ready:
            if (ear_left >= self.BASELINE_MIN_EAR
                    and ear_right >= self.BASELINE_MIN_EAR):
                self._baseline_samples_l.append(float(ear_left))
                self._baseline_samples_r.append(float(ear_right))
                if len(self._baseline_samples_l) >= self.BASELINE_SAMPLES:
                    self._baseline_l = float(np.median(self._baseline_samples_l))
                    self._baseline_r = float(np.median(self._baseline_samples_r))
                    self._baseline_samples_l.clear()
                    self._baseline_samples_r.clear()
            return False

        self.last_left = float(ear_left / self._baseline_l)
        self.last_right = float(ear_right / self._baseline_r)
        both_closed = (
            self.last_left <= self.close_ratio_l
            and self.last_right <= self.close_ratio_r
        )
        both_open = (
            self.last_left >= self.open_ratio_l
            and self.last_right >= self.open_ratio_r
        )

        if both_open:
            self._since = None
            self._gap_since = None
            if self._latched:
                if self._open_since is None:
                    self._open_since = now
                elif now - self._open_since >= self.RELEASE_S:
                    self._latched = False
                    self._open_since = None
            else:
                self._baseline_l = (
                    (1.0 - self.BASELINE_ALPHA) * self._baseline_l
                    + self.BASELINE_ALPHA * float(ear_left)
                )
                self._baseline_r = (
                    (1.0 - self.BASELINE_ALPHA) * self._baseline_r
                    + self.BASELINE_ALPHA * float(ear_right)
                )
            return False

        self._open_since = None
        if self._latched:
            self._since = None
            return False

        if not both_closed:
            if self._since is not None:
                if self._gap_since is None:
                    self._gap_since = now
                elif now - self._gap_since > self.AMBIGUOUS_GAP_S:
                    self.cancel_hold()
            return False

        if self._since is None:
            self._since = now
            self._gap_since = None
            return False
        self._gap_since = None
        if now - self._since >= self.HOLD_S:
            self._since = None
            self._latched = True
            return True
        return False


class MouthOpenClickDetector:
    """A sustained, calibrated mouth opening produces one right click."""

    HOLD_S = 1.0
    COOLDOWN_S = 0.8
    AMBIGUOUS_GAP_S = 0.15

    def __init__(self, profile: Optional[dict] = None) -> None:
        self.open_threshold = (
            float(profile["open_threshold"]) if profile
            else MOUTH_OPEN_THRESHOLD
        )
        self.release_threshold = (
            float(profile["release_threshold"]) if profile
            else MOUTH_OPEN_RELEASE_THRESHOLD
        )
        self._since: Optional[float] = None
        self._gap_since: Optional[float] = None
        self._latched = False
        self._last_click = 0.0

    @property
    def active(self) -> bool:
        return self._since is not None

    @property
    def hold_progress(self) -> float:
        if self._since is None:
            return 0.0
        return float(np.clip(
            (time.monotonic() - self._since) / self.HOLD_S, 0.0, 1.0
        ))

    def reset(self) -> None:
        self._since = None
        self._gap_since = None
        self._latched = False

    def update(self, mouth_open: float, now: Optional[float] = None) -> bool:
        now = time.monotonic() if now is None else now
        if not np.isfinite(mouth_open):
            self.reset()
            return False

        if self._since is None:
            if mouth_open >= self.open_threshold:
                self._since = now
            return False

        if mouth_open >= self.release_threshold:
            self._gap_since = None
        elif not self._latched:
            if self._gap_since is None:
                self._gap_since = now
            elif now - self._gap_since > self.AMBIGUOUS_GAP_S:
                self.reset()
                return False
        else:
            self.reset()
            return False

        if (not self._latched and now - self._since >= self.HOLD_S
                and now - self._last_click >= self.COOLDOWN_S):
            self._latched = True
            self._last_click = now
            return True
        return False
