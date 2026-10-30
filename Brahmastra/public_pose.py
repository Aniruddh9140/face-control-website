"""NumPy-only head orientation regressor trained on public BIWI images."""
from collections import deque
from pathlib import Path

import numpy as np

LANDMARKS = (1, 4, 6, 10, 33, 133, 263, 362, 168, 127, 356, 234, 454, 9, 151, 109, 338, 197)
FEATURE_COUNT = len(LANDMARKS) * 3
MODEL_PATH = Path(__file__).resolve().parent / "models" / "biwi_head_pose.npz"


def pose_features(xyz, width, height):
    points = np.asarray(xyz, dtype=float)
    if points.ndim != 2 or points.shape[0] < 468 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("Invalid pose landmarks")
    points = points * np.array([width, height, width])
    center = (points[33] + points[263]) / 2
    scale = np.linalg.norm(points[33] - points[263])
    if scale < 10:
        raise ValueError("Move closer to the camera")
    return ((points[list(LANDMARKS)] - center) / scale).ravel()


def design_matrix(z, quadratic):
    z = np.atleast_2d(z)
    return np.c_[np.ones(len(z)), z, z * z] if quadratic else np.c_[np.ones(len(z)), z]


class PoseRegressor:
    def __init__(self, mean, scale, coef, quadratic=False, signs=(-1., -1.)):
        self.mean, self.scale, self.coef = mean, scale, coef
        self.quadratic, self.signs = bool(quadratic), np.asarray(signs, float)

    @classmethod
    def fit(cls, x, y, strength=1., quadratic=False):
        x, y = np.asarray(x, float), np.asarray(y, float)
        mean, scale = x.mean(axis=0), np.maximum(x.std(axis=0), .01)
        z = np.clip((x - mean) / scale, -8, 8)
        a = design_matrix(z, quadratic)
        penalty = np.eye(a.shape[1]) * strength
        penalty[0, 0] = 0
        coef = np.linalg.solve(a.T @ a + penalty, a.T @ y)
        # Output follows the mirrored camera horizontally and image-y vertically.
        signs = [-np.sign(np.corrcoef(x[:, 0], y[:, 0])[0, 1]),
                  np.sign(np.corrcoef(x[:, 1], y[:, 1])[0, 1])]
        return cls(mean, scale, coef, quadratic, signs)

    def predict(self, x):
        x = np.asarray(x, float)
        if x.shape[-1] != FEATURE_COUNT or not np.isfinite(x).all():
            raise ValueError("Invalid pose input")
        z = np.clip((x - self.mean) / self.scale, -8, 8)
        result = design_matrix(z, self.quadratic) @ self.coef
        return result[0] if x.ndim == 1 else result

    def save(self, path):
        path = Path(path)
        temporary = path.with_suffix(".tmp.npz")
        np.savez(temporary, version=1, mean=self.mean, scale=self.scale,
                 coef=self.coef, quadratic=int(self.quadratic), signs=self.signs,
                 landmarks=np.asarray(LANDMARKS))
        temporary.replace(path)

    @classmethod
    def load(cls, path=MODEL_PATH):
        with np.load(path, allow_pickle=False) as d:
            quadratic = bool(int(d["quadratic"]))
            if int(d["version"]) != 1 or not np.array_equal(d["landmarks"], LANDMARKS):
                raise ValueError("Unsupported public head model")
            shapes = {"mean": (FEATURE_COUNT,), "scale": (FEATURE_COUNT,),
                      "coef": (1 + FEATURE_COUNT * (2 if quadratic else 1), 2), "signs": (2,)}
            for key, shape in shapes.items():
                if d[key].shape != shape or not np.isfinite(d[key]).all():
                    raise ValueError("Invalid public head model")
            if np.any(d["scale"] <= 0) or not np.all(np.abs(d["signs"]) == 1):
                raise ValueError("Invalid public model scales or direction")
            return cls(d["mean"].copy(), d["scale"].copy(), d["coef"].copy(), quadratic, d["signs"].copy())


class PublicPointerModel:
    """Relative head rotation, centered once per session; no nine-dot prerequisite."""
    def __init__(self, regressor, screen):
        self.regressor, self.screen = regressor, tuple(screen)
        self.neutral = None
        self.samples = deque(maxlen=90)
        self.validation_error = None
        self.progress = 0.

    @property
    def ready(self):
        return self.neutral is not None

    def cancel_hold(self):
        self.samples.clear()
        self.progress = 0.

    def observe(self, features, now):
        angles = self.regressor.predict(features)
        if not np.isfinite(angles).all() or np.any(np.abs(angles) > 50):
            self.cancel_hold()
            return False
        if self.ready:
            return True
        if np.any(np.abs(angles) > 25):
            self.cancel_hold()
            return False
        if self.samples and now - self.samples[-1][0] > .25:
            self.cancel_hold()
        self.samples.append((now, angles))
        while self.samples and now - self.samples[0][0] > 2.3:
            self.samples.popleft()
        values = np.array([a for _, a in self.samples])
        if len(values) >= 6 and np.max(np.ptp(values, axis=0)) > 4:
            self.cancel_hold()
            self.samples.append((now, angles))
            return False
        self.progress = min((now - self.samples[0][0]) / 2, 1.)
        if self.progress >= 1 and len(values) >= 20:
            self.neutral = np.median(values, axis=0)
        return self.ready

    def recenter(self, features):
        angles = self.regressor.predict(features)
        if np.any(np.abs(angles) > 40):
            raise ValueError("Face the screen before recentering")
        self.neutral = angles
        self.cancel_hold()

    def predict(self, features):
        if not self.ready:
            return (np.asarray(self.screen) - 1) / 2
        angles = self.regressor.predict(features)
        # +/- 22 yaw and +/- 16 pitch span the display at default speed.
        xy = .5 + (angles - self.neutral) * self.regressor.signs / np.array([44., 32.])
        return np.clip(xy, 0, 1) * (np.asarray(self.screen) - 1)
