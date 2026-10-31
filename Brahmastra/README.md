# EyeSync

Face Controlled Operating System navigation combined with EyeSync's offline voice typing.
The project is named **EyeSync**; its separate working folder is **Brahmastra** on the Desktop.

## Single-file Windows release

The packaged application is `release/EyeSync.exe`. It includes Python 3.15 and
all runtime models. Users download that one executable, install it for their
account, and press Connect on the existing dashboard. Build and GitHub publishing
instructions are in `release.md`; `.github/workflows/release.yml` builds, tests,
publishes version tags and updates the Pages download metadata. GitHub Pages serves
the website; tracking and speech run locally. Source builds continue to work below.

## Start

Double-click **Start EyeSync.bat** in this folder. The included `.venv` uses the
**Windows x64 Python 3.15.0** installation already on this computer and contains its own dependencies.
The launcher explicitly selects this folder's `.venv\Scripts\pythonw.exe`, so it
does not depend on which Python version is selected on PATH.
Neither original project folder is needed to run this copy.

The bundled public-data head model supports a quick start: click **Start camera**,
face the screen and hold your head steady for two seconds, then click **Enable pointer**.
Dashboard **Connect** also requests enable automatically. Turn your head gently
to steer. **Recenter** on the dashboard resets the neutral head position.
The public model is a general starting point; the personal nine-dot setup below
can improve the fit for your seating position and display.

1. Click **Start camera**. Keep both eyes open and your mouth relaxed for a few seconds.
2. Click **Calibrate pointer** for personal training (or **Train & enable pointer**
   if no public model is installed). Gently move your head toward each of the nine dots;
   hold still until the dot advances. Eye movement alone does not calibrate head control.
   The center dot comes first. Watch the camera preview, instructions and capture ring.
   Press **Esc** to cancel. Allow around 30–45 seconds; this depends on how quickly
   you settle at each dot. Each position times out after 25 seconds without a valid hold.
3. Successful training enables the requested pointer after facial baselines settle.
   **Calibrate pointer** also retrains an existing profile. Pause cancels pending enable.
4. Move your head to steer. **Blink and reopen after ~0.3 seconds** for a left click,
   or **~1.3 seconds** for a right click. Left/right winks also click. A big smile
   held for 1.5 seconds pauses or resumes. See the gesture table below.
5. Click a text field in another application, then use **Start voice typing**, or
   close both eyes for **2.5 seconds** to toggle voice typing. Reopen before repeating.
  The eye shortcut becomes available after head calibration; the voice button
  also works before calibration.

**Emergency pointer pause: Ctrl+Alt+E.** You can also press Ctrl+Alt+Shift+F9.
Ctrl+Alt+Shift+F5 toggles voice typing (the original EyeSync shortcut); F8 remains
an alias. Hotkeys already owned by other programs are left alone.
The camera and microphone stay off until requested. Closing the app releases both;
a lightweight dashboard listener remains ready for the next Connect request.

## Activate from the web dashboard

Open [the EyeSync dashboard](https://aniruddh9140.github.io/face-control-website/#dashboard),
sign in, and press **Connect**. The dashboard starts **this EyeSync copy in
Desktop/Brahmastra**, including its face-control mouse and original offline speech.
The dashboard's **Open EyeSync** link also targets this copy through `eyesync://start`.

The local listener starts at Windows login and is already running after setup.
If it has been stopped, run **Start EyeSync.bat** once. When the desktop window
is closed, the listener stays in standby with no camera or AI model loaded;
Connect can reopen the app. Merely probing whether EyeSync is installed never
opens the camera. A normal authenticated-site connection starts the camera.

Use **Calibrate EyeSync** on the dashboard the first time, moving your head toward
each dot. Dashboard activation requests pointer control; it becomes active only
after the public model's two-second neutral hold (or personal calibration),
open-eye baselines and a visible face. Successful
dashboard calibration resumes that requested control. **Disable Mouse Control**
and **Pause** stop it; Ctrl+Alt+E remains the emergency pause.

Dashboard controls support mouse enable/disable, pause/resume, recenter, speed,
face gestures, voice typing, camera preview, calibration/cancel, show window and
quit. The site's **Blink** click setting uses the original EyeSync blink/wink
clicks; **Dwell** adds a left click after holding still for
1.2 seconds. Eye-gaze jump and its separate eye calibration are not part of this
head-control build; those commands return an explicit unsupported-feature message.

The bridge listens on `ws://localhost:8765` and `ws://localhost:8080/eye-tracker`,
bound to loopback only. It accepts the exact live site's HTTPS origin, reports
positions in the dashboard's 800×600 coordinate frame, and sends voice state
without sending dictated words. Camera preview is sent only when requested.
Disconnecting the last dashboard stops its camera and microphone after a two-second
reconnection grace; manually started camera sessions remain under local control.

## Original EyeSync AI gestures

The default gesture controls now come from `E:\EyeSync\live_inference.py`.

| Gesture | Action |
|---|---|
| Blink, reopen after tick (~0.3 s) | Left click |
| Blink, reopen after double tick (~1.3 s, before 2.4 s) | Right click |
| Left / right wink | Left / right click |
| Big smile, 1.5 s | Pause / resume |
| Mouth pulled to your left, 1.5 s | Recenter |
| Mouth pulled to your right, 3 s | Personal pointer calibration |
| Left / right ear toward shoulder, held | Scroll up / down |
| Both eyes closed, 2.5 s | Voice typing on / off |
| Mouth wide open, 3 s | Quit (close your mouth to cancel) |

Relax between gestures. For scrolling, tilt your left ear toward your left
shoulder to scroll up, or your right ear toward your right shoulder to scroll
down. Hold about 22° beyond your resting tilt briefly (~0.35 seconds). A larger
tilt scrolls faster; straighten your head to stop. The window shows the active
scroll direction. The pointer returns to its position before the lean, stays
still during scrolling, and settles for 0.2 seconds afterward. Camera gaps,
face loss, pause, and disabling gestures cancel scrolling.
Mouth gestures are suppressed while voice typing is listening or recognizing.
The former Face Controlled OS smile-left / mouth-right / one-second voice
mapping is retained only as an explicit internal legacy profile and is not active
alongside these controls. Eye-gaze jump remains a separate unsupported feature.

## Voice typing

The original `E:\EyeSync\voice_typing.py` is preserved byte for byte. It includes
offline Whisper recognition, Silero speech detection, punctuation, phrase joining,
text editing commands, target-window selection, an audio meter, and automatic
stop after eight seconds without speech. It uses the locally copied English model.

The full original system is used through `improved_voice.py`: original Whisper,
Silero, phrase splitting, punctuation, editing, Unicode typing and window tracking.
The original floating microphone bar is restored, including level, idle countdown,
last phrase, and a Stop button that does not steal typing focus. The bar distinguishes
dictation from command listening and restores Stop after an error/retry.

Speech models load before the microphone opens. Capture has a five-second bounded
buffer; an overflow or interrupted stream stops the session with a restart message.
An explicit Stop discards pending speech and releases the microphone. If a native
recognition call is already running, it may finish, but its result is discarded.
Calibration and application shutdown also cancel pending speech.

If the foreground text window or an EyeSync click changes during recognition,
voice typing stops so old queued speech cannot be sent to the new target. Start
again in the intended field. An interrupted Windows keyboard write stops dictation,
clears its text-joining state and asks you to check any partially inserted text.

**Accurate mode** (a tick box under Voice typing) uses a larger speech model: fewer
mistakes, especially on Indian English accents, but each phrase takes 7–30 seconds
depending on how busy the PC is. It downloads 190 MB once; speech itself still stays
on this PC. Say a short phrase and wait for the text before the next one.

During normal voice typing, say **enter** to press Enter in the focused app—no command
box is needed. Say **hello enter** to type “hello” and press Enter, or dictate first
and say **enter** after a pause. In Chrome this submits the search, message, or form
when that field uses Enter to submit. **New line** inserts a line break instead.
Say **delete that** or **stop listening** as separate phrases.
When the EyeSync window has focus, normal dictation returns to the previously
used application. Mouth click gestures are suppressed throughout listening and
recognition so speaking does not trigger right clicks. Head steering remains available.
Head-tilt scrolling also remains available while dictating.

## Voice command panel

The Face Controlled OS command field and its six exact phrases are integrated
into the same window. Type a phrase and pause for 1.2 seconds, press Enter, or
click **Listen for command** and speak it. The panel uses EyeSync's offline
recognizer; a successful spoken command ends that listening session.
Stop ordinary dictation before switching to command listening.

- Open the on-screen keyboard
- Close the on-screen keyboard
- Increase the size of the on-screen keyboard
- Decrease the size of the on-screen keyboard
- Let me adjust the position of the on-screen keyboard
- Done adjusting the position of the on-screen keyboard

You can click a listed phrase to put it in the command field. Only an exact
supported phrase runs. Unrecognized speech in command mode is never typed into
another application and never executed as a shell command.

To move the keyboard, position the pointer over its title bar before starting
adjustment. Finish adjustment to release the drag. Pausing, face loss, camera
stop and app shutdown also release an active drag. Windows may request elevation
when opening its protected on-screen keyboard; handle any Windows prompt yourself.

## Files and independence

| File | Purpose |
|---|---|
| `eyesync_app.py` | Combined EyeSync desktop window and command field |
| `control_engine.py` | Camera lifecycle, calibration, pointer control and voice routing |
| `calibration.py` | Stable head-position capture and training timeout handling |
| `public_pose.py` | Public-data head orientation and neutral setup |
| `models/biwi_head_pose.npz` | Locally trained public-data model |
| `face_control.py` | Original Face Controlled OS smoothing and gesture detectors |
| `native_input.py` | Face Controlled OS native Windows mouse control |
| `voice_typing.py` | Unmodified EyeSync offline dictation engine |
| `improved_voice.py` | Cancellation, target guards and reliable keyboard delivery |
| `voice_capture.py` | Bounded microphone capture and recovery |
| `voice_overlay.py` | Original floating microphone bar with retry/mode fixes |
| `voice_commands.py` | Original Face Controlled OS command allowlist and OSK controls |
| `face_engine.py` | EyeSync's current MediaPipe backend and latest-frame camera |
| `models/` | Local face, English speech and speech-detection model files |
| `data/pointer.npz` | Calibration coefficients, created after successful calibration |
| `data/eyesync.log` | Bounded local diagnostic log |
| `docs/INTEGRATION.md` | Research, design decisions and practical limitations |
| `docs/VALIDATION.md` | What was verified and what needs real-user testing |

Windows' EyeSync login listener and `eyesync://` handler are registered to this
folder. Their previous values are backed up in `data/dashboard-registration-backup.json`.
The original project and installed program files remain unchanged. This copy
does not use the old EyeSync API key or need an external USB mouse board. Calibration
stores numbers, not camera images or recordings. The folder is on your OneDrive
Desktop, so OneDrive may sync these local files according to your existing settings.

## Developer checks

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe eyesync_app.py --ui-smoke --smoke-seconds 3
.\.venv\Scripts\python.exe tests\check_dashboard_lifecycle.py
```

The smoke option disables camera, microphone, real input, hotkeys and commands.
For another camera, run `Start EyeSync.bat --camera 1`.
The lifecycle check uses port 8799 and disables camera, microphone and mouse input.
To restore the previous Windows launch registration, run
`.\.venv\Scripts\python.exe register_dashboard.py restore`, then restart Windows.

For a different Windows computer, install 64-bit Python 3.15, create
a fresh `.venv`, install `requirements-python315.txt`, and retain the `models` folder.
The included environment is specific to this computer; it is not a portable executable.

## Camera selection

In the EyeSync window, stop the camera, choose a camera from the dropdown above
the preview, then click **Start camera**. Check the preview to identify your
built-in or external USB camera; Windows assigns their numbers. The selection is
saved for the next launch. Each camera has its own personal calibration. After
switching, calibrate that camera before enabling the personal pointer. If a device
is unplugged or does not open, select another camera; the app does not silently
switch devices. Recalibrate if Windows changes device numbering after reconnecting.
