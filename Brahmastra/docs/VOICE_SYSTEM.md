# Complete original voice system with integration improvements

The source is `E:/EyeSync/voice_typing.py`, preserved byte for byte in Brahmastra.
The original VoiceOverlay and its no-focus/work-area helpers came from
`E:/EyeSync/main_window.py`. Original dictation commands, English Whisper model,
Silero detection, pause-based phrases, punctuation, phrase joining, Unicode input,
foreground selection, start/stop sounds, eight-second silence timeout and ten-minute
session limit remain. Voice can start from the two-eye gesture, window, dashboard,
or original Ctrl+Alt+Shift+F5 shortcut. F8 remains an additional shortcut.

## Changes

- `improved_voice.py` subclasses the original session engine. Models load before
  recording starts. Explicit stop, calibration and shutdown discard unfinished
  speech; cancellation is checked again after native recognition returns.
- `voice_models.py` adds the optional Accurate mode: Whisper small.en, fetched once on
  request and checked against its SHA-1 before use. The fast model stays the default.
  On 66 South Asian EdAcc clips it got 23.5% of words wrong against 28.2%, but took
  6.9 s against 1.6 s per phrase when idle, and 11–31 s on a busy PC.
- `voice_capture.py` subclasses the original mono microphone. Its queue is bounded
  to 1,200 callback blocks (nominally one minute, because recognition runs on the reading thread). PortAudio status faults, malformed
  audio and queue overflow become explicit session errors. No damaged stream is
  silently spliced back together. Unsupported sample rates fall back to the native
  device rate with the original resampler; failed streams are closed.
- A post-recognition capture-health check catches audio faults that occur while
  inference is running. Stop closes capture while an in-flight recognition call
  finishes; that call's cancelled output is not delivered.
- Target window and EyeSync click context are captured at speech start, checked
  after recognition and guarded between keyboard batches. A changed target ends
  the session and requires a fresh start, preventing old queued speech from being
  redirected into the new target. This does not detect every caret move within
  an application or physical mouse click; the old tracker is still window based.
- Partial SendInput delivery stops later batches, releases accepted keys still
  held by EyeSync, clears text-composition history, and reports partial insertion.
  It never automatically retries, which could duplicate existing text.
- `voice_overlay.py` restores the old floating microphone bar without its old
  main window/server. Dictation and command-listening labels differ. A retry after
  an error restores the Stop button. The command panel uses all original stop
  aliases and never inserts command-recognition text into another application.

The upstream core source remains unchanged. Improvements live in subclasses and
integration modules, allowing its SHA256 and original speech regression tests to
remain valid. All files are in Desktop/Brahmastra; E:/EyeSync is untouched.

## Spoken Enter

Normal dictation uses `DictationComposer` in `improved_voice.py`. A final “enter”,
“press enter”, “hit enter”, or “press return” types preceding text first, then sends
a Return key press/release. A separate “enter” also works. It needs no command panel.
The existing target-change, cancellation, and partial-write guards still apply.
Submitting clears composition history so undo cannot erase a previously submitted
message. “Please enter your name” remains literal text; “new line” remains Shift+Enter.

## Verification

On the included Windows x64 Python 3.15 environment:

```
.venv\Scripts\python.exe -m unittest discover -s tests -q
.venv\Scripts\python.exe tests\check_voice_ui.py
.venv\Scripts\python.exe tests\check_dashboard_lifecycle.py
```

The full suite includes 14 capture fault/lifecycle tests, 17 improved-session tests,
five keyboard-batch tests, original speech/gesture regressions and dashboard checks.
Microphone/keyboard tests use fake devices and output; the UI check uses synthetic
states. Original model checks load the actual installed speech model using silence.
The updated system has not been scored for word accuracy on a new human speech
dataset or tried with this user's live microphone. No accuracy improvement over
the original recognizer is claimed. Its model and decoding settings are retained.
The improvements concern complete controls, reliable capture, cancellation,
cleanup and delivery behavior. Dictation remains English and fully local.
