# EyeSync integration notes

## Current gesture source

The default gesture profile now uses the AI from `E:\EyeSync\live_inference.py`.
`eyesync_gestures.py` contains its unchanged BlinkDetector, PoseBins,
FaceGestures, TiltScroll and close-gesture method/constants. `gesture_adapter.py`
routes these to this application's native input, voice service and UI events.
Original AI learned expression scores, per-pose eye baselines, wink guards and
long-closure rules are supplied from the face backend. The former Face Controlled
OS gesture mapping below is available only as an explicit internal `face_os`
profile and is not active concurrently. Speech suppresses mouth actions.

The UI and dashboard descriptions reflect the original AI meanings. Eye gaze
jump is a separate feature and remains unsupported by this combined build.

## Source selection

The original projects remain unchanged:

- `E:\EyeSync` provides the offline speech engine, Windows foreground tracking,
  current face backend and validated camera reader.
- `E:\rudra12345\Face-Controlled-Operating-System-main\Face-Controlled-Operating-System-main`
  provides the whole-face bounding-box cursor features, feature EMA, One Euro
  smoothing, smile click, mouth-open click, bilateral eye-closure detector,
  native mouse input and exact on-screen-keyboard commands.

`SOURCE_MANIFEST.json` records hashes of preserved modules. The three gesture
detector classes and the two filter classes were transplanted verbatim into
`face_control.py`. The feature geometry in `measurements` follows the source's
image-plane mouth-width/eye-distance ratio, inner-lip gap/eye-distance ratio,
3D eye-aspect ratios and first-468-landmark face bounds.

## Compatibility and changes

The existing EyeSync runtime is Python 3.15, MediaPipe 1.1.1, OpenCV 5.0.0 and
NumPy 2.5.4. The face-control project's old `mp.solutions.face_mesh` API is absent
from this runtime. EyeSync's Tasks FaceLandmarker supplies the same landmark
indices, allowing the original geometry and detectors to work with this runtime.
No old sklearn/joblib calibration is loaded into the new app.

The new nine-point mapping uses the source project's four face-bounds features
and ridge-plus-residual concept. A regularized NumPy ridge map with a restrained
smooth radial residual correction replaces its sklearn MLP. This avoids an
incompatible older dependency stack and unsafe pickle model loading. It is an
adaptation of the source design, not its identical trained model. Coefficients
are saved as numeric NPZ, validated on load, and rejected when camera or display
dimensions change. A calibration with inadequate head movement is rejected.

In the optional legacy `face_os` profile, the source smile detector's 1.2-second hold and 15% width increase, the mouth-open
detector's 1-second hold and release thresholds, and the 1-second eye closure /
0.25-second reopen timing are preserved. Eyes toggle native EyeSync dictation
instead of sending Tab to an external speech program. Cursor smoothing uses the
source defaults: feature alpha 0.35, minimum cutoff 1.2, beta 0.015. The movement
deadzone now scales to the shorter display edge, bounded to 2–5 pixels, so small
head adjustments are not discarded by the original 10-pixel threshold.
Gestures freeze the pointer, while ordinary speech suppresses mouth clicks.

The local command-panel behavior is integrated directly instead of starting a
second application and WebSocket server. It retains the source's six exact
phrases, 1.2-second text debounce and OSK functions. Spoken panel commands use a
subclass of EyeSync's unchanged voice module. `improved_voice.py` retains its
recognizer, segmentation and composer while adding lifecycle and output guards.
Command mode bypasses text insertion entirely. The original no-focus microphone
bar and F5 voice shortcut are restored; F8 remains an alias.

## Research consulted

- Google MediaPipe Face Landmarker guide:
  https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker
  Documents 3D landmarks, VIDEO mode, one-face smoothing and optional blendshapes.
- Google FaceLandmarkerResult API:
  https://ai.google.dev/edge/api/mediapipe/python/mp/tasks/vision/FaceLandmarkerResult
  Documents landmark, blendshape and transformation-matrix results.
- Microsoft SendInput:
  https://learn.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-sendinput
  Documents accepted input counts, correct structure size and Windows integrity
  restrictions. Native input checks the accepted count; partial clicks attempt
  to release a button before reporting failure.
- python-sounddevice usage:
  https://github.com/spatialaudio/python-sounddevice/blob/master/doc/usage.rst
  Supports retaining EyeSync's callback/queue approach for microphone capture,
  with recognition handled outside the audio callback.

## Practical limits

Webcam head control requires personal calibration and usable lighting. Software
checks do not establish real-user accuracy or guarantee universal accessibility.
Facial ratios vary across users; this build uses source defaults and learns relaxed
smile/open-eye baselines at startup. It does not import either project's old
personal calibration. Only the primary display is mapped. Display/camera changes
require a new calibration. The bundled speech model is English. Windows can block
input into elevated applications; voice typing retains EyeSync's warning for that.

## Dashboard integration

`dashboard_bridge.py` implements the existing live website's protocol without
editing or publishing the website. `dashboard_launcher.py` reuses EyeSync's
lightweight `site_access.wait_for_visit` standby protocol: probe responses use
close code 4003, start responses use 4002, explicit Quit uses 4000. The supervisor
returns to standby after the desktop app exits. The URL handler and existing
EyeSync per-user login entry now start this supervisor from Brahmastra; old
registry values are saved before installation for rollback.

Commands and status follow the contract in the original EyeSync `WEBSITE.md`
and the live page retrieved on 29 October 2026. Browser requests to both default
ports are supported on IPv4 and IPv6 loopback. Other website origins are refused.
The native camera and input are activated only by a normal dashboard connection
or local user action, never by probe discovery. First-run personal head calibration
is still required. The original voice-typing module remains byte-for-byte unchanged.

The dashboard's historical `blink` click-mode name maps to the requested facial
click system in this build. `dwell` supports one left click per stationary hold.
Eye-gaze jump was not carried over to this head-control integration, so its
commands are explicitly rejected instead of reporting success without functionality.

Camera activation and live face detection were verified from the signed-in dashboard.
Personal calibration and live pointer/voice accuracy remain user-operated checks.
No image, speech recording, API key, user log or previous calibration was copied from EyeSync.

## Scroll stability update

The original TiltScroll detector remains unchanged: five-frame median,
22-degree start, 0.35-second hold, hysteresis and speed increasing with tilt.
The controller now restores the position from before the lean (including a
130 ms allowance for median delay), holds movement and clicks during the tilt,
and holds for 0.2 seconds after release as in the original AI pipeline.
Camera gaps over 0.25 seconds cancel unfinished gestures instead of counting
the missing interval as a hold. Nonfinite landmark/expression inputs suspend
input and reset gesture state. Scrolling is disabled while paused, gestures
are disabled, or a keyboard drag is active. Speech and tilt scrolling can coexist.
Dwell timers reset on pause or unavailable tracking, preventing an immediate
click on resume. Quit emits one UI event per completed hold. Scroll direction
appears in the local window and dashboard status (`scrolling`: -1/0/1).
