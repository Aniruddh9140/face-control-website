# Personal pointer training improvements

This build learns a small personal head-to-screen mapping locally. It uses the
existing pretrained face detector. It does not claim to have retrained MediaPipe
or learned this user's movements before a real calibration has been completed.

The first target is screen center. At each of nine targets the sampler allows
1.3 seconds to move, then requires a stable 0.8-second window with at least 12
observations. It checks robust coordinate spread and drift, captures the median,
and starts over after lost face detection or a camera gap. The overlay shows a
mirrored camera preview, capture ring, and live instructions. A 25-second timeout
per target closes an unsuccessful session instead of waiting indefinitely.

The mapper compares three ridge strengths with leave-one-target-out predictions.
Each fold computes its own normalization without seeing the omitted point.
Inconsistent positions are rejected using median and 90th-percentile prediction
error. A failed or cancelled session preserves the saved profile. Numeric model
data and validation error are stored locally in `data/pointer.npz`; recordings
are not saved. Old version-1 profiles still load.

The displayed withheld-dot error is a training diagnostic, not measured real-world
cursor accuracy. Selecting a regularization value using these same folds makes it
an optimistic estimate. Real target selection, varying lighting, fatigue and
different seating positions still need user trials. Sampling and rejection
thresholds are engineering starting values, not clinically validated thresholds.

The enabled **Train & enable pointer** button replaces a disabled first-run Enable
button. After successful training, pending control starts once the face baselines
settle. Explicit pause cancels that request. Source click timing, original offline
voice typing and the dashboard activation contract are retained.

## Research and design rationale

- [MediaPipe Face Landmarker documentation](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker)
  describes the pretrained landmark pipeline used by the existing backend.
- [Casiez, Roussel and Vogel: One Euro Filter](https://gery.casiez.net/1euro/)
  describes adapting smoothing to signal speed to balance jitter and lag. The
  source project's filter is retained; this update reduces the separate cursor
  deadzone to 2–5 pixels rather than claiming a new universal filter tuning.
- [scikit-learn cross-validation guide](https://scikit-learn.org/stable/modules/cross_validation.html)
  describes evaluating models on withheld samples. The implementation here uses
  NumPy only, with whole targets omitted, avoiding a scikit-learn dependency on
  Python 3.15. The thresholds and capture algorithm are this project's design.

## Repeat verification

Use this folder's Python 3.15 environment:

```
.venv\Scripts\python.exe -m unittest discover -s tests -v
.venv\Scripts\python.exe tests\check_training_ui.py
.venv\Scripts\python.exe tests\check_dashboard_lifecycle.py
```

Tests use synthetic features and fake mouse output, including full nine-target
training, invalid training preserving an existing file, unseen-position prediction,
fine pointer movement, and cancellation by emergency pause. They cannot replace
a personal calibration or an actual usability session.
