"""Stable, bounded sampling for personal head-pointer training (no camera I/O)."""
from collections import deque

import numpy as np


class StableCapture:
    """Wait for a settled head, then collect a consistent 0.8-second window.

    Coordinates are normalized camera coordinates. These tolerances are starting
    values to validate with real users, not a claim of measured human accuracy.
    """
    SETTLE_SECONDS = 1.3
    HOLD_SECONDS = 0.8
    MIN_SAMPLES = 6
    TIMEOUT_SECONDS = 25.0

    def __init__(self, now):
        self.started = self.settle_started = now
        self.samples = deque(maxlen=120)
        self.progress = 0.0
        self.message = "Move your head toward the dot."

    def expired(self, now):
        return now - self.started >= self.TIMEOUT_SECONDS

    def missing(self, now):
        self.samples.clear()
        self.settle_started = now
        self.progress = 0.0
        self.message = "Face not visible. Face the camera in good light."

    def update(self, feature, now):
        feature = np.asarray(feature, dtype=float)
        if feature.shape != (4,) or not np.isfinite(feature).all():
            self.missing(now)
            return None
        if now - self.settle_started < self.SETTLE_SECONDS:
            self.message = "Move toward the dot, then hold your head steady."
            return None
        # Gaps must not count as time spent holding still.
        if self.samples and now - self.samples[-1][0] > 0.25:
            self.samples.clear()
        self.samples.append((now, feature.copy()))
        while self.samples and now - self.samples[0][0] > 1.0:
            self.samples.popleft()
        duration = now - self.samples[0][0]
        values = np.array([value for _, value in self.samples])
        self.progress = min(duration / self.HOLD_SECONDS, 1.0)
        if len(values) >= self.MIN_SAMPLES:
            spread = np.quantile(values, .9, axis=0) - np.quantile(values, .1, axis=0)
            split = len(values) // 2
            drift = np.abs(np.median(values[:split], axis=0) - np.median(values[split:], axis=0))
            if np.max(spread) > .010 or np.max(drift) > .006:
                # Restart at the newest sample, so a moving head cannot train a dot.
                self.samples.clear()
                self.samples.append((now, feature.copy()))
                self.progress = 0.0
                self.message = "Still moving. Hold gently; no need to lean farther."
                return None
        self.message = f"Hold steady — {int(self.progress * 100)}% captured."
        # Use the same sample minimum as the stability check. Requiring twelve
        # frames inside this one-second buffer prevents a 10 FPS camera from
        # ever completing, even with a perfectly steady head.
        if duration >= self.HOLD_SECONDS and len(values) >= self.MIN_SAMPLES:
            return np.median(values, axis=0)
        return None
