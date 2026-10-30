"""Original EyeSync offline speech with bounded capture and cancellable delivery.

The upstream voice_typing.py stays intact; its recognizer, VAD, composer,
Unicode keyboard conversion and foreground tracker remain the implementation.
"""
from __future__ import annotations

import logging
import math
import queue
import threading
import time

import numpy as np

import desktop
import voice_models
import voice_typing as vt
from voice_capture import Microphone

log = logging.getLogger('eyesync.voice')


class ReliableTypist(vt.Typist):
    """Stop on a partial Windows injection and release keys still held by us."""
    def run(self, ops):
        import ctypes
        events = vt.key_events(ops)
        sent, held = 0, {}
        try:
            for start in range(0, len(events), self.BATCH):
                if getattr(self, 'cancelled', lambda: False)():
                    break
                chunk = events[start:start+self.BATCH]
                inputs = (self._INPUT * len(chunk))()
                for i, (vk, scan, flags) in enumerate(chunk):
                    inputs[i].type = 1
                    inputs[i].ki = self._KEYBDINPUT(vk, scan, flags, 0, 0)
                accepted = int(self._user32.SendInput(len(chunk), ctypes.byref(inputs), ctypes.sizeof(self._INPUT)))
                accepted = max(0, min(accepted, len(chunk)))
                sent += accepted
                for vk, scan, flags in chunk[:accepted]:
                    key = (vk, scan, flags & ~vt.KEYEVENTF_KEYUP)
                    if flags & vt.KEYEVENTF_KEYUP:
                        held.pop(key, None)
                    else:
                        held[key] = (vk, scan, flags | vt.KEYEVENTF_KEYUP)
                if accepted != len(chunk):
                    break
                if start+self.BATCH < len(events):
                    time.sleep(.01)
        finally:
            if held:
                releases = (self._INPUT * len(held))()
                for i, (vk, scan, flags) in enumerate(held.values()):
                    releases[i].type = 1
                    releases[i].ki = self._KEYBDINPUT(vk, scan, flags, 0, 0)
                self._user32.SendInput(len(held), ctypes.byref(releases), ctypes.sizeof(self._INPUT))
        return sent


class ImprovedVoiceTyping(vt.VoiceTyping):
    def __init__(self, state, **kwargs):
        self._choose_model = kwargs.get('recognizer') is None
        kwargs.setdefault('mic_factory', Microphone)
        # Injected fakes continue to work; production uses the original Win32
        # typist with partial-write handling. Construction never opens a mic.
        if kwargs.get('typist') is None and vt.sys.platform == 'win32':
            kwargs['typist'] = ReliableTypist()
        super().__init__(state, **kwargs)
        self.shutdown = threading.Event()
        self._live_mic = None
        self._phrase_context = None
        self._mic_lock = threading.Lock()
        if isinstance(self.typist, ReliableTypist):
            self.typist.cancelled = self._cancelled

    def _cancelled(self):
        return self.shutdown.is_set() or self.state.stop.is_set() or self.stop_session.is_set()

    def _close_mic(self):
        with self._mic_lock:
            mic = self._live_mic
            self._live_mic = None
        if mic is not None:
            try:
                mic.close()
            except Exception:
                log.warning('Microphone cleanup failed', exc_info=True)

    def close(self, timeout=5.):
        self.shutdown.set()
        self.stop_session.set()
        self._close_mic()
        deadline = time.monotonic()+timeout
        for thread in (self.session, self.thread):
            if thread is not None and thread.ident is not None and thread is not threading.current_thread():
                thread.join(max(0., deadline-time.monotonic()))

    def _control(self):
        try:
            while not self.shutdown.is_set() and not self.state.stop.is_set():
                try:
                    on, source = self.state.voice_requests.get(timeout=.1)
                except queue.Empty:
                    on = source = None
                if self.shutdown.is_set() or self.state.stop.is_set():
                    break
                if self.tracker is not None:
                    try:
                        self.tracker.poll()
                    except OSError:
                        pass
                if source is not None:
                    want = not self.active() if on is None else bool(on)
                    if want and not self.active():
                        if self.state.read().calib_step not in ('', 'done'):
                            self._show_problem('Finish or cancel calibration before starting voice typing.')
                        else:
                            self._start(source)
                    elif not want:
                        self.stop_session.set()
                        self._close_mic()
                if self.active() and self.state.read().calib_step not in ('', 'done'):
                    self.stop_session.set()
                    self._close_mic()
                if self.clear_at and time.monotonic() > self.clear_at and not self.active():
                    self.clear_at = 0.
                    self.state.update(voice='', voice_hint='')
        finally:
            self.stop_session.set()
            self._close_mic()

    def _run_session(self, source):
        vad = mic = None
        published = False
        problem = ''
        try:
            if self._choose_model:
                self.recognizer = voice_models.recognizer()
            # Load before recording so model startup cannot build stale audio.
            self.recognizer.load()
            if self._cancelled():
                return
            vad = self.vad_factory()
            if self._cancelled():
                return
            mic = self.mic_factory()
            mic.open()
            with self._mic_lock:
                self._live_mic = mic
                published = True
            if self._cancelled():
                return
            self.state.update(voice_device=getattr(mic, 'name', 'Default microphone'))
            if self.sounds:
                desktop.play_voice_sound(True)
            self._listen(mic, vad)
        except Exception as exc:
            self.composer.reset()
            self.context = (None,-1,-math.inf)
            if not self._cancelled():
                # These are EyeSync's and Windows' own error texts, never dictated words.
                log.warning('Voice session failed: %s: %s', type(exc).__name__, str(exc)[:140])
                problem = f'Voice stopped: {str(exc)[:140]}. Start again when ready.'
        finally:
            self._close_mic()
            if mic is not None and not published:
                try:
                    mic.close()
                except Exception:
                    log.warning('Failed microphone startup cleanup')
            for resource in (vad, self.recognizer):
                if resource is not None and hasattr(resource, 'close'):
                    try:
                        resource.close()
                    except Exception:
                        log.warning('Voice resource cleanup failed: %s', type(resource).__name__)
            self._phrase_context = None
            if self.sounds:
                desktop.play_voice_sound(False)
            if problem:
                self._show_problem(problem)
            else:
                self.state.update(voice='', voice_level=0., voice_left=0.)

    def _target_context(self):
        snap = self.state.read()
        return (self.tracker.peek() if self.tracker is not None else None,
                snap.click_count+snap.right_click_count)

    def _listen(self, mic, vad):
        seg = vt.SpeechSegmenter(vad)
        started = last_audio = last_publish = time.monotonic()
        zeros, skip = 0, int(vt.RATE*vt.START_SOUND_S)
        self._phrase_context = None
        while not self._cancelled():
            block = mic.read(.1)
            now = time.monotonic()
            if self._cancelled():
                break
            if block is None:
                if now-last_audio > 3.:
                    raise OSError('Microphone disconnected or stopped sending audio')
                continue
            block = np.asarray(block, np.float32).ravel()
            if not np.isfinite(block).all():
                raise OSError('Microphone supplied invalid audio')
            last_audio = now
            if skip:
                block, skip = block[skip:], max(0, skip-block.size)
            zeros = zeros+block.size if not np.any(block) else 0
            if zeros > vt.MIC_SILENT_S*vt.RATE and seg.t < vt.MIC_SILENT_S+1:
                self.state.update(voice_hint='No microphone signal. Check the selected Windows input and microphone access.')
            if not seg.speaking:
                self._phrase_context = self._target_context()
            for phrase in seg.feed(block):
                if self._phrase(phrase):
                    return 'stopped'
                self._phrase_context = self._target_context()
            left = vt.STOP_AFTER_S-seg.quiet_s
            if left <= 0 or now-started > vt.MAX_SESSION_S:
                return 'idle or session limit'
            if now-last_publish > .08:
                last_publish = now
                self.state.update(voice='hearing' if seg.speaking else 'listening',
                                  voice_level=round(seg.level,2), voice_left=round(left,1) if left<vt.COUNTDOWN_S else 0.)
        # Explicit stop, calibration and shutdown discard pending speech.
        return 'cancelled'

    def _phrase(self, audio):
        if self._cancelled():
            return True
        target = self._phrase_context or self._target_context()
        self.state.update(voice='working', voice_level=0.)
        started = time.monotonic()
        text = self.recognizer(audio)
        if self._cancelled():
            self.composer.reset()
            return True
        with self._mic_lock:
            mic = self._live_mic
        if mic is not None and hasattr(mic, 'check_health'):
            mic.check_health()
        took = time.monotonic()-started
        self.state.update(voice_latency=round(took,2))
        if not vt.clean(text):
            reason = vt.dropped_because(text)
            log.info('[voice] %.1fs of sound - no speech in it (%s; %.2fs)',
                     audio.size/vt.RATE, reason, took)
            if 'another language' in reason:
                self.state.update(voice_hint="That didn't sound like English, so nothing was typed. "
                                             "Voice typing understands English only.")
            return False
        log.info('[voice] %d characters heard (%.1fs of speech, recognised in %.2fs)',
                 len(text.strip()), audio.size/vt.RATE, took)
        return self._deliver_phrase(text, target, started)

    def _deliver_phrase(self, text, target, started):
        window, clicks = target
        if self._target_context() != target:
            self.composer.reset()
            self.stop_session.set()
            self.state.update(voice_hint='Text target changed. Voice stopped; start again in the intended field.')
            return True
        last_window, last_clicks, last_time = self.context
        if window != last_window or clicks != last_clicks or started-last_time > vt.CONTEXT_S:
            self.composer.reset()
        ops = self.composer.plan(text)
        stop = ('stop','') in ops
        ops = [op for op in ops if op[0]!='stop']
        typed = ''.join(str(a) for o,a in ops if o=='text')
        if ops and self.tracker is not None:
            destination = self.tracker.target()
            if destination is None or destination != window:
                self.composer.reset()
                self.stop_session.set()
                self.state.update(voice_hint='Voice stopped. Click the intended text field and start again.')
                return True
            if self.tracker.elevated(destination):
                self.composer.reset()
                self.state.update(voice_hint="Windows does not allow EyeSync to type into administrator apps.")
                return stop
        if self._cancelled():
            self.composer.reset()
            return True
        if ops:
            expected = len(vt.key_events(ops))
            if isinstance(self.typist, ReliableTypist):
                self.typist.cancelled = lambda: self._cancelled() or self._target_context() != target
            try:
                sent = self.typist.run(ops)
            finally:
                if isinstance(self.typist, ReliableTypist):
                    self.typist.cancelled = self._cancelled
            if sent != expected:
                self.composer.reset()
                self.context = (None,-1,-math.inf)
                self.state.update(voice_hint='Typing was interrupted. Some text may be present; check the field before continuing.',voice_text='')
                self.stop_session.set()
                return True
            self.context = (window, clicks, time.monotonic())
            self.state.update(voice_text=typed.strip() or vt.describe(ops),voice_hint='')
            log.info('Typed %d keyboard events', sent)
        return stop
