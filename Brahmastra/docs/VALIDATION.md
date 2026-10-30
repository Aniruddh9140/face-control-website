# EyeSync validation — 29 October 2026

Completed on the local Windows environment in Desktop/Brahmastra.

Python compatibility: **CPython 3.15.0, Windows x64**. The launcher selects
`Brahmastra\.venv\Scripts\pythonw.exe` explicitly. App imports, the test suite,
UI smoke launch and native model checks were run using this Python 3.15
environment. `requirements-python315.txt` records the verified dependency versions.

- **201 automated tests passed**, zero failures or skips. Tests cover source
  gesture timing and latching, natural-blink rejection, one-eye rejection,
  face-loss cancellation, native voice-toggle routing, speech click suppression,
  fake mouse click output, drag guards/release, camera backend lifecycle,
  calibration interpolation/save/load/mismatch rejection, exact command routing,
  normal dictation delegation and the retained EyeSync speech regressions.
- Live WebSocket tests also cover origin rejection, idle probes, start activation,
  status/commands, opt-in JPEG preview, 800×600 coordinate mapping, transcript
  privacy, and disconnect cleanup. They use ephemeral loopback ports and synthetic data.
- A separate real-process lifecycle test on port 8799 passed: standby probe ->
  start request -> combined UI -> Quit/4000 -> standby -> start again. Its explicit
  test mode disables camera, microphone, hotkeys and native input throughout.
- The installed Windows `eyesync://` and login Run entries were verified to
  target `Desktop/Brahmastra/dashboard_launcher.py` with the local Python 3.15
  environment. A real probe to port 8765 confirmed the registered standby was ready.
- The public live website's inline JavaScript was inspected to match its exact
  discovery and command contract. After the user signed in, the real dashboard
  Connect button launched the Brahmastra copy and showed Connected, Camera On,
  Face found, approximately 22–24 updates/second, and zero rejected updates.
  Its Pause command visibly changed to Resume; Quit returned the application to
  standby; Connect relaunched it. A live probe confirmed implementation
  `brahmastra` and Python `3.15`. The pointer remained inactive pending personal
  head calibration. The optional gaze object is omitted so unsupported eye-gaze
  controls stay hidden on the page. Current gesture instructions appear in its status hint.
- The preserved `voice_typing.py` matches its original SHA-256.
- All combined-app modules imported successfully in the independent local `.venv`.
- Main app and integration modules passed Python byte-compilation.
- A UI smoke launch and timed clean shutdown succeeded without opening camera
  or microphone or injecting input. The interface was visually inspected.
- Real MediaPipe Tasks FaceLandmarker initialized from the copied local model
  and processed a synthetic black frame, returning no face as expected.
- Real Silero VAD loaded and classified synthetic silence as non-speech
  (probability approximately 0.0017).
- Real Whisper loaded the copied English model and returned an empty
  transcription for synthetic silence.

Public-data update: a BIWI head-pose regressor was trained locally using the
Python 3.15 environment. Of 3,145 sampled frames, 2,215 had usable detected faces
inside the declared pose range. Person-separated splits contained 1,584 training,
299 validation and 332 test samples. Hyperparameters were selected on validation;
the final model fit training plus validation and scored 3.08 degrees mean absolute
head-angle error on the untouched four-person test set (yaw 2.56, pitch 3.60).
The constant-prediction baseline scored 15.63 degrees. This is an internal pose
benchmark, not a measured screen-coordinate accuracy result. Exact metrics,
split subjects, candidate settings and model hash are recorded in
`PUBLIC_TRAINING_RESULTS.json`.

Held-out public features replayed through the actual controller with fake mouse
output: the model learned neutral, enabled, moved 148 simulated pixels on a head
turn, and stopped on pause. No real input was injected. All nine training overlay
positions, live feedback and cleanup also passed a UI check. Dashboard supervisor
start/quit/relaunch passed again after the update.

Personal pointer accuracy, live gesture comfort, microphone recognition and real
Windows keyboard/mouse/OSK output still need a calibrated user-operated session.
Actual camera startup and live face detection were verified through the dashboard.
Automated tests use
fake input and synthetic data; they do not move the user's pointer, type into
applications or open the camera/microphone. The app starts with input disabled;
dashboard activation or local action starts the camera. Public mode requires a
two-second steady neutral hold; personal nine-dot calibration is optional and
takes precedence when a valid profile exists. Eye-closure voice toggles wait for
successful public neutral setup or personal calibration
to avoid enabling the microphone while first-run baselines are being learned.

Original AI gesture update: 59 source detector regressions plus eight controller routing tests passed. AI blink/wink clicks, smile pause/resume, mouth-side recenter/calibrate, tilt scrolling, 2.5-second voice closure and three-second quit are now the default. UI smoke, public-feature replay and dashboard lifecycle checks passed again. Source detector code is extracted unchanged; integration adapts routing to this app. Real user gesture comfort remains unverified.

Scroll stability update: 11 new regressions passed for both scroll directions,
straighten-to-stop, pause, gesture disable, restoring the pre-lean cursor,
release settling, camera gaps during blink/quit, invalid roll rejection,
face-loss reset, scrolling while dictating, single quit dispatch and resetting
dwell on pause. Full suite: 146 passed. UI training smoke, public-data controller
replay and dashboard supervisor start/quit/relaunch passed after these changes.
Checks used fake mouse output; live scrolling comfort remains a user check.

Voice system update: original offline speech module hash still matches E:/EyeSync.
Restored the original no-focus voice bar and Ctrl+Alt+Shift+F5 shortcut. Added
bounded/fault-reporting capture, reliable device cleanup, cancellation of pending
text/commands, recognition-time target guards and partial keyboard delivery handling.
Fourteen fake-capture checks, 17 enhanced-session checks, five keyboard-batch checks
and a threaded control/session lifecycle check passed. The lifecycle check confirms
Stop releases a fake microphone before blocked inference completes and the cancelled
result is not typed. Voice UI smoke passed for dictation/command modes, level,
countdown, error/retry, Stop visibility and cleanup. Training UI and dashboard
start/quit/relaunch checks passed again. Full test suite: 183 tests on Python 3.15.
No live microphone/keyboard injection was used to validate the new behavior;
recognition accuracy with this user's speech remains a live check.

Single-file release 1.6.0: packaged with PyInstaller 6.22.3 and Python 3.15 x64.
The complete frozen EXE passes 201 tests, including real face model initialization
on a synthetic blank frame and Silero inference on silence. The bundled Whisper
model is exercised by the retained speech-model regression. A no-registration
installation into a temporary directory preserves the EXE hash. The installed EXE
passes two dashboard start/quit/standby cycles on isolated port 8799 with all
hardware/input disabled. A direct test-mode launch verifies that persistent data
is written beside the installed executable, outside PyInstaller's extraction
folder. Source and frozen versions use the same version metadata (1.6.0).
Release workflow YAML and embedded Python syntax were validated locally. The
existing GitHub Pages configuration was inspected read-only: main branch, root.
Publishing and real HKCU registration were not performed. GitHub Actions/Pages
execution will be checked when the user uploads the source and pushes its tag.

Reliability and voice update, 30 October 2026 (source only, not built): 232 automated
tests and the UI smoke launch passed. A real voice session on recorded EdAcc speech
(real microphone class, session loop, Whisper and Silero; fake sound card and typing)
confirmed that a phrase taking over five seconds to recognise ended the session; the
capture queue now holds one minute. On a busy PC the Accurate model can still reach
that limit. Accurate mode's accuracy figures come from recorded clips, including only
three speakers labelled Indian English. It has not been tried on a live microphone or
the user's own voice, and pointer accuracy and gesture comfort remain unmeasured.
