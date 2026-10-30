"""Offline voice regressions with fake audio devices, recognizers and keyboard output."""
import queue
import unittest
from unittest.mock import Mock

import numpy as np

import control_engine as ce
from improved_voice import ImprovedVoiceTyping
import voice_typing as vt


class Recognizer:
    def __init__(self, text="Private sentence.", events=None):
        self.text, self.events = text, events if events is not None else []
        self.on_recognize = self.on_load = None
        self.calls = self.closed = 0

    def load(self):
        self.events.append("model loaded")
        if self.on_load:
            self.on_load()

    def __call__(self, audio):
        self.calls += 1
        if self.on_recognize:
            self.on_recognize()
        return self.text

    def close(self):
        self.closed += 1


class Tracker:
    def __init__(self):
        self.window = 101

    def peek(self):
        return self.window

    def target(self):
        return self.window

    def elevated(self, window):
        return False

    def poll(self):
        pass


class Typist:
    def __init__(self):
        self.ops = []

    def run(self, ops):
        self.ops.extend(ops)
        return len(vt.key_events(ops))


class Mic:
    name = "Fake microphone"

    def __init__(self, events=None, open_error=False):
        self.events = events if events is not None else []
        self.open_error = open_error
        self.closed = 0
        self.read = Mock(return_value=None)

    def open(self):
        self.events.append("microphone opened")
        if self.open_error:
            raise OSError("Fake startup error")

    def close(self):
        self.closed += 1


class Vad:
    def __init__(self):
        self.closed = 0

    def __call__(self, block):
        return 1.0 if np.any(block) else 0.0

    def close(self):
        self.closed += 1


class ImprovedVoiceTests(unittest.TestCase):
    audio = np.ones(vt.RATE, np.float32) * .1

    def voice(self, cls=ImprovedVoiceTyping, text="Private sentence.", **kwargs):
        state = ce.State()
        rec, typist, tracker = Recognizer(text), Typist(), Tracker()
        defaults = dict(recognizer=rec, typist=typist, tracker=tracker,
                        sounds=False, check=lambda: "")
        defaults.update(kwargs)
        if cls is ce.RoutedVoiceTyping:
            voice = cls(state, queue.Queue(), **defaults)
        else:
            voice = cls(state, **defaults)
        return voice, state, rec, typist, tracker

    def test_foreign_speech_is_logged_and_explained_without_any_words(self):
        voice, state, _, typist, _ = self.voice(text="(speaking in foreign language)")
        with self.assertLogs("eyesync.voice", "INFO") as logs:
            self.assertFalse(voice._phrase(self.audio))
        self.assertEqual(typist.ops, [])
        self.assertIn("English", state.read().voice_hint)
        self.assertIn("speech in another language", logs.output[0])

    def test_other_sounds_are_logged_but_do_not_nag_the_user(self):
        for text in ("[BLANK_AUDIO]", "", "(buzzing)"):
            with self.subTest(text=text):
                voice, state, _, _, _ = self.voice(text=text)
                with self.assertLogs("eyesync.voice", "INFO") as logs:
                    self.assertFalse(voice._phrase(self.audio))
                self.assertEqual(state.read().voice_hint, "")
                self.assertIn("no speech in it", logs.output[0])

    def test_recognition_time_is_logged_without_the_words(self):
        voice, _, _, _, _ = self.voice(text="Private sentence.")
        with self.assertLogs("eyesync.voice", "INFO") as logs:
            voice._phrase(self.audio)
        self.assertIn("recognised in", logs.output[0])
        self.assertNotIn("Private", " ".join(logs.output))

    def test_a_failed_session_logs_its_reason(self):
        mic = Mic(open_error=True)
        voice, _, _, _, _ = self.voice(mic_factory=lambda: mic, vad_factory=Vad)
        with self.assertLogs("eyesync.voice", "WARNING") as logs:
            voice._run_session("test")
        self.assertIn("OSError: Fake startup error", " ".join(logs.output))

    def test_cancelled_recognition_never_types(self):
        voice, _, rec, typist, _ = self.voice()
        rec.on_recognize = voice.stop_session.set
        voice.composer.plan("Earlier text")
        self.assertTrue(voice._phrase(self.audio))
        self.assertEqual(typist.ops, [])
        self.assertEqual(voice.composer.typed, [])

    def test_cancelled_command_recognition_never_dispatches(self):
        voice, _, rec, typist, _ = self.voice(ce.RoutedVoiceTyping, "Open the on-screen keyboard")
        voice.command_mode = True
        rec.on_recognize = voice.stop_session.set
        self.assertTrue(voice._phrase(self.audio))
        self.assertTrue(voice.commands.empty())
        self.assertEqual(typist.ops, [])

    def test_focus_change_aborts_session_instead_of_retargeting_backlog(self):
        voice, state, rec, typist, tracker = self.voice()
        rec.on_recognize = lambda: setattr(tracker, "window", 202)
        self.assertTrue(voice._phrase(self.audio))
        self.assertTrue(voice.stop_session.is_set())
        self.assertEqual(typist.ops, [])
        self.assertIn("target", state.read().voice_hint.lower())

    def test_click_change_aborts_session(self):
        voice, state, rec, typist, _ = self.voice()
        rec.on_recognize = lambda: state.update(click_count=1)
        self.assertTrue(voice._phrase(self.audio))
        self.assertTrue(voice.stop_session.is_set())
        self.assertEqual(typist.ops, [])

    def test_target_disappearing_while_restoring_focus_aborts_session(self):
        voice, _, _, typist, tracker = self.voice()
        tracker.target = lambda: None
        self.assertTrue(voice._phrase(self.audio))
        self.assertTrue(voice.stop_session.is_set())
        self.assertEqual(typist.ops, [])

    def test_spoken_stop_survives_focus_change(self):
        voice, _, rec, typist, tracker = self.voice(text="Stop listening.")
        rec.on_recognize = lambda: setattr(tracker, "window", 202)
        self.assertTrue(voice._phrase(self.audio))
        self.assertEqual(typist.ops, [])

    def test_command_panel_accepts_every_original_stop_alias(self):
        for phrase in vt.STOP_PHRASES:
            with self.subTest(phrase=phrase):
                voice, _, _, typist, _ = self.voice(ce.RoutedVoiceTyping, phrase)
                voice.command_mode = True
                self.assertTrue(voice._phrase(self.audio))
                self.assertTrue(voice.commands.empty())
                self.assertEqual(typist.ops, [])

    def test_model_is_loaded_before_microphone_opens(self):
        events = []
        rec, mic, vad = Recognizer(events=events), Mic(events), Vad()
        voice, _, _, _, _ = self.voice(recognizer=rec, mic_factory=lambda: mic, vad_factory=lambda: vad)
        voice._listen = Mock(return_value="stopped")
        voice._run_session("test")
        self.assertEqual(events, ["model loaded", "microphone opened"])
        self.assertEqual((mic.closed, vad.closed, rec.closed), (1, 1, 1))

    def test_model_failure_closes_resources_without_opening_microphone(self):
        def fail():
            raise RuntimeError("Fake model error")
        rec = Recognizer()
        rec.on_load = fail
        mic_factory, vad_factory = Mock(), Mock()
        voice, state, _, _, _ = self.voice(recognizer=rec, mic_factory=mic_factory, vad_factory=vad_factory)
        with self.assertLogs("eyesync.voice", "WARNING"):
            voice._run_session("test")
        self.assertEqual(rec.closed, 1)
        self.assertEqual(state.read().voice, "error")
        mic_factory.assert_not_called()
        vad_factory.assert_not_called()

    def test_vad_failure_closes_model_without_opening_microphone(self):
        rec = Recognizer()
        mic_factory = Mock()
        voice, state, _, _, _ = self.voice(recognizer=rec, mic_factory=mic_factory,
                                          vad_factory=Mock(side_effect=RuntimeError("Fake VAD error")))
        with self.assertLogs("eyesync.voice", "WARNING"):
            voice._run_session("test")
        self.assertEqual(rec.closed, 1)
        self.assertEqual(state.read().voice, "error")
        mic_factory.assert_not_called()

    def test_microphone_start_failure_still_closes_all_resources(self):
        rec, mic, vad = Recognizer(), Mic(open_error=True), Vad()
        voice, state, _, _, _ = self.voice(recognizer=rec, mic_factory=lambda: mic, vad_factory=lambda: vad)
        with self.assertLogs("eyesync.voice", "WARNING"):
            voice._run_session("test")
        self.assertEqual((mic.closed, vad.closed, rec.closed), (1, 1, 1))
        self.assertEqual(state.read().voice, "error")
        self.assertIsNone(voice._live_mic)

    def test_stop_during_model_load_never_opens_microphone(self):
        rec = Recognizer()
        mic_factory = Mock()
        voice, _, _, _, _ = self.voice(recognizer=rec, mic_factory=mic_factory)
        rec.on_load = voice.stop_session.set
        voice._run_session("test")
        mic_factory.assert_not_called()
        self.assertEqual(rec.closed, 1)

    def test_explicit_stop_discards_unfinished_speech(self):
        voice, _, rec, typist, _ = self.voice()
        mic, vad = Mic(), Vad()
        audio = np.ones(vt.RATE, np.float32) * .1
        blocks = iter([audio])
        def read(timeout):
            try:
                return next(blocks)
            except StopIteration:
                voice.stop_session.set()
                return None
        mic.read = read
        self.assertEqual(voice._listen(mic, vad), "cancelled")
        self.assertEqual(rec.calls, 0)
        self.assertEqual(typist.ops, [])

    def test_capture_fault_during_inference_prevents_typing(self):
        voice, _, _, typist, _ = self.voice()
        mic = Mic()
        mic.check_health = Mock(side_effect=OSError("Fake capture overflow"))
        voice._live_mic = mic
        with self.assertRaisesRegex(OSError, "capture overflow"):
            voice._phrase(self.audio)
        self.assertEqual(typist.ops, [])

    def test_partial_keyboard_output_clears_composition_and_stops(self):
        voice, state, _, typist, _ = self.voice()
        typist.run = lambda ops: 1
        self.assertTrue(voice._phrase(self.audio))
        self.assertTrue(voice.stop_session.is_set())
        self.assertEqual(voice.composer.typed, [])
        self.assertIn("interrupted", state.read().voice_hint.lower())

    def test_transcript_does_not_enter_logs(self):
        phrase = "Secret mango paperclip password sentence."
        voice, _, _, typist, _ = self.voice(text=phrase)
        with self.assertLogs("eyesync.voice", "INFO") as logs:
            self.assertFalse(voice._phrase(self.audio))
        self.assertEqual(typist.ops, [("text", phrase)])
        self.assertNotIn("mango", " ".join(logs.output).lower())
        self.assertNotIn("paperclip", " ".join(logs.output).lower())

    def test_close_ends_control_thread_without_requiring_application_stop(self):
        voice, state, _, _, _ = self.voice()
        voice.start()
        try:
            voice.close(timeout=1.)
            self.assertFalse(voice.thread.is_alive())
            self.assertFalse(state.stop.is_set())
        finally:
            voice.close(timeout=1.)


if __name__ == '__main__':
    unittest.main()
