"""Behavioral tests with synthetic landmarks, fake input and fake audio only."""
import hashlib
import json
from pathlib import Path
import queue
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

import numpy as np

import control_engine as ce
import face_control as fc
import voice_commands as vc
import voice_typing as vt


class GestureTests(unittest.TestCase):
    def smile(self):
        d = fc.LandmarkSmileClickDetector()
        for _ in range(30):
            d.update(1.0, 0.02, 0.109, now=1)
        return d

    def eyes(self):
        d = fc.BilateralEyeClosureDetector()
        for _ in range(35):
            d.update(0.3, 0.3, now=1)
        return d

    def test_smile_requires_full_hold_and_rearms(self):
        d = self.smile()
        self.assertTrue(d.ready)
        self.assertFalse(d.update(1.2, 0.02, 0.109, now=2))
        self.assertFalse(d.update(1.2, 0.02, 0.109, now=3.1))
        self.assertTrue(d.update(1.2, 0.02, 0.109, now=3.21))
        self.assertFalse(d.update(1.2, 0.02, 0.109, now=5))
        d.update(1, 0.02, 0.109, now=6)
        d.update(1.2, 0.02, 0.109, now=7)
        self.assertTrue(d.update(1.2, 0.02, 0.109, now=8.21))

    def test_open_mouth_cancels_smile(self):
        d = self.smile()
        d.update(1.2, 0.02, 0.109, now=2)
        self.assertFalse(d.update(1.2, 0.2, 0.109, now=4))
        self.assertFalse(d.active)

    def test_mouth_one_click_per_hold(self):
        d = fc.MouthOpenClickDetector()
        self.assertFalse(d.update(0.2, now=2))
        self.assertFalse(d.update(0.2, now=2.99))
        self.assertTrue(d.update(0.2, now=3.01))
        self.assertFalse(d.update(0.2, now=9))
        d.update(0.02, now=10)
        d.update(0.2, now=11)
        self.assertTrue(d.update(0.2, now=12.01))

    def test_natural_blink_does_not_toggle_voice(self):
        d = self.eyes()
        d.update(0.1, 0.1, now=2)
        self.assertFalse(d.update(0.3, 0.3, now=2.2))

    def test_one_eye_never_toggles_voice(self):
        d = self.eyes()
        for t in np.arange(2, 6, 0.05):
            self.assertFalse(d.update(0.1, 0.3, now=t))

    def test_eye_toggle_requires_reopening(self):
        d = self.eyes()
        d.update(0.1, 0.1, now=2)
        self.assertTrue(d.update(0.1, 0.1, now=3.01))
        self.assertFalse(d.update(0.1, 0.1, now=8))
        d.update(0.3, 0.3, now=9)
        d.update(0.3, 0.3, now=9.3)
        d.update(0.1, 0.1, now=10)
        self.assertTrue(d.update(0.1, 0.1, now=11.01))

    def test_loss_of_face_cancels_holds(self):
        c = ce.Controller((1920, 1080), gesture_profile="face_os")
        c.mouse = Mock()
        c.smile = self.smile()
        c.smile.update(1.2, 0.02, 0.109, now=2)
        c.mouth.update(0.2, now=2)
        c._lost_face()
        self.assertFalse(c.smile.active)
        self.assertFalse(c.mouth.active)
        self.assertTrue(c.must_relax)


class CalibrationTests(unittest.TestCase):
    def calibration(self):
        targets = np.array(ce.POINTS)
        center = np.c_[0.25 + targets[:, 0] * 0.3, 0.2 + targets[:, 1] * 0.3]
        features = np.c_[center, center + 0.2]
        return features, targets

    def test_mapping_spans_screen_and_predicts_between_dots(self):
        x, y = self.calibration()
        m = ce.PointerModel.fit(x, y, (320, 240), (1920, 1080))
        for feature, target in zip(x, y):
            np.testing.assert_allclose(m.normalized(feature), target, atol=0.03)
        halfway = (x[0] + x[1]) / 2
        np.testing.assert_allclose(m.normalized(halfway), (y[0] + y[1]) / 2, atol=0.04)

    def test_calibration_round_trip_and_camera_mismatch(self):
        x, y = self.calibration()
        m = ce.PointerModel.fit(x, y, (320, 240), (1920, 1080))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "pointer.npz"
            m.save(path)
            loaded = ce.PointerModel.load(path, (320, 240), (1920, 1080))
            np.testing.assert_allclose(loaded.predict(x[4]), m.predict(x[4]))
            with self.assertRaises(ValueError):
                ce.PointerModel.load(path, (640, 480), (1920, 1080))
            with self.assertRaises(ValueError):
                ce.PointerModel.load(path, (320, 240), (1280, 720))

    def test_still_head_and_nonfinite_calibrations_rejected(self):
        x, y = self.calibration()
        with self.assertRaises(ValueError):
            ce.PointerModel.fit(np.repeat(x[:1], 9, axis=0), y, (320, 240), (1920, 1080))
        x[0, 0] = np.nan
        with self.assertRaises(ValueError):
            ce.PointerModel.fit(x, y, (320, 240), (1920, 1080))

    def test_bad_landmarks_rejected(self):
        for x in (np.zeros((12, 3)), np.full((478, 3), np.nan)):
            with self.assertRaises(ValueError):
                ce.measurements(x, 320, 240)


class IntegrationTests(unittest.TestCase):
    def test_camera_worker_opens_backend_once_and_releases_it(self):
        c = ce.Controller((1920, 1080), mouse_factory=Mock)
        camera, mesh = Mock(), Mock()
        camera.get.side_effect = [320, 240]
        camera.read.return_value = (True, np.zeros((240, 320, 3), np.uint8))
        mesh.last_xyz = None
        mesh.process.side_effect = lambda frame: c.worker_stop.set()
        with patch.object(ce.fe, "open_camera", autospec=True, return_value=camera) as opened, \
             patch.object(ce.fe, "FaceMeshBackend", return_value=mesh), \
             patch.object(ce.native_input, "hotkey_kill_pressed", return_value=False):
            c._run()
        opened.assert_called_once_with(0, width=320, height=240)
        camera.release.assert_called_once()
        mesh.close.assert_called_once()
        self.assertFalse(c.state.read().running)
        self.assertEqual([c.events.get_nowait()[0] for _ in range(2)], ["started", "stopped"])

    def controller(self):
        c = ce.Controller((1920, 1080), gesture_profile="face_os")
        c.mouse = Mock()
        c.model = Mock()
        c.model.predict.return_value = np.array([600, 400])
        c.state.update(enabled=True, face=True, calibrated=True)
        for i in range(45):
            c._track((np.array([.2, .2, .6, .6]), 1., .02, .3, .3), i / 30)
        c.mouse.reset_mock()
        return c

    def feed(self, c, width=1., gap=.02, eyes=.3, start=3., duration=2.):
        for t in np.arange(start, start + duration, 1 / 30):
            c._track((np.array([.2, .2, .6, .6]), width, gap, eyes, eyes), t)

    def test_source_clicks_reach_mouse_once(self):
        c = self.controller()
        self.feed(c, width=1.25)
        c.mouse.click.assert_called_once_with("left")
        self.assertEqual(c.state.read().click_count, 1)
        c = self.controller()
        self.feed(c, gap=.2)
        c.mouse.click.assert_called_once_with("right")
        self.assertEqual(c.state.read().right_click_count, 1)

    def test_speaking_never_right_clicks(self):
        c = self.controller()
        c.state.update(voice="listening")
        self.feed(c, gap=.3, duration=6)
        c.mouse.click.assert_not_called()

    def test_dwell_clicks_once_per_stationary_hold(self):
        c = self.controller()
        c.state.update(click_mode="dwell")
        self.feed(c, start=3, duration=3)
        self.feed(c, start=6, duration=3)
        c.mouse.click.assert_called_once_with("left")

    def test_dashboard_enable_waits_for_calibration_and_pause_cancels_it(self):
        c = self.controller()
        c.model = None
        c.state.update(enabled=False)
        c.request_enable()
        self.feed(c, duration=1)
        self.assertFalse(c.state.read().enabled)
        self.assertTrue(c.enable_requested.is_set())
        c.pause()
        self.assertFalse(c.enable_requested.is_set())

    def test_no_control_without_calibration(self):
        c = self.controller()
        c.model = None
        c.mouse.reset_mock()
        self.feed(c, gap=.2)
        c.mouse.move.assert_not_called()
        c.mouse.click.assert_not_called()

    def test_pausing_blocks_pointer_and_releases_drag(self):
        c = self.controller()
        c.dragging = True
        c.pause()
        c._release_drag()
        self.feed(c, gap=.2)
        c.mouse.click.assert_not_called()
        c.mouse.button_up.assert_called_once_with("left")
        self.assertFalse(c.state.read().enabled)

    def test_face_loss_releases_drag(self):
        c = self.controller()
        c.dragging = True
        c._lost_face()
        c.mouse.button_up.assert_called_once_with("left")

    def test_eye_gesture_requests_native_voice_not_tab(self):
        c = self.controller()
        with patch.object(vc, "set_tab_key_down") as tab:
            self.feed(c, eyes=.1)
        self.assertEqual(c.state.voice_requests.get_nowait(), (None, "eyes closed"))
        self.assertTrue(c.state.voice_requests.empty())
        tab.assert_not_called()

    def test_first_run_eye_movements_do_not_start_microphone_before_calibration(self):
        c = self.controller()
        c.state.update(calibrated=False)
        self.feed(c, eyes=.1)
        self.assertTrue(c.state.voice_requests.empty())

    def test_only_exact_panel_commands_run(self):
        c = self.controller()
        with patch.object(vc, "open_onscreen_keyboard") as opened:
            c._execute_command("please open the on screen keyboard and delete files")
            opened.assert_not_called()
            c._execute_command("Open the on-screen keyboard.")
            opened.assert_called_once()

    def test_drag_requires_pointer_and_titlebar(self):
        c = self.controller()
        with patch.object(vc, "is_keyboard_titlebar_cursor_position", return_value=False):
            c._execute_command("Let me adjust the position of the on-screen keyboard")
        c.mouse.button_down.assert_not_called()
        with patch.object(vc, "is_keyboard_titlebar_cursor_position", return_value=True):
            c._execute_command("Let me adjust the position of the on-screen keyboard")
        c.mouse.button_down.assert_called_once_with("left")
        c._execute_command("Done adjusting the position of the on-screen keyboard")
        c.mouse.button_up.assert_called_once_with("left")

    def test_speech_module_is_byte_for_byte_eyesync(self):
        manifest = json.loads((ce.ROOT / "docs" / "SOURCE_MANIFEST.json").read_text())
        digest = hashlib.sha256((ce.ROOT / "voice_typing.py").read_bytes()).hexdigest()
        self.assertEqual(digest, manifest["voice_typing.py"]["sha256"])


class VoiceRouteTests(unittest.TestCase):
    def voice(self, phrase):
        state = ce.State()
        typist = Mock()
        typist.run.side_effect = lambda ops: len(vt.key_events(ops))
        recognizer = Mock(return_value=phrase)
        v = ce.RoutedVoiceTyping(state, queue.Queue(), recognizer=recognizer, typist=typist,
                                tracker=None, sounds=False, check=lambda: "")
        return v, state, typist

    def test_command_mode_routes_without_typing(self):
        v, state, typist = self.voice("Open the on-screen keyboard.")
        v.command_mode = True
        self.assertTrue(v._phrase(np.zeros(16000)))
        self.assertEqual(v.commands.get_nowait(), ("command", "Open the on-screen keyboard."))
        typist.run.assert_not_called()

    def test_unrecognized_command_never_becomes_text(self):
        v, state, typist = self.voice("Hello this should not be typed.")
        v.command_mode = True
        self.assertFalse(v._phrase(np.zeros(16000)))
        self.assertTrue(v.commands.empty())
        typist.run.assert_not_called()

    def test_normal_typing_uses_original_composer(self):
        v, state, typist = self.voice("Hello there.")
        self.assertFalse(v._phrase(np.zeros(16000)))
        typist.run.assert_called_once_with([("text", "Hello there.")])

    def test_original_stop_listening_works(self):
        v, state, typist = self.voice("Stop listening.")
        self.assertTrue(v._phrase(np.zeros(16000)))
        typist.run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
