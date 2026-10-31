"""Synthetic training and input regressions; never access user hardware."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
from types import SimpleNamespace

import numpy as np

from calibration import StableCapture
import control_engine as ce
from eyesync_app import App


class CaptureTests(unittest.TestCase):
    feature = np.array([.25, .25, .55, .65])

    def test_stable_noisy_hold_captures_median_after_settling(self):
        rng = np.random.default_rng(42)
        capture = StableCapture(0)
        result = None
        for t in np.arange(0, 3, 1 / 30):
            result = capture.update(self.feature + rng.normal(0, .001, 4), t)
            if result is not None:
                self.assertGreaterEqual(t, 2.1)
                break
        self.assertIsNotNone(result)
        np.testing.assert_allclose(result, self.feature, atol=.002)

    def test_moving_head_is_not_recorded(self):
        capture = StableCapture(0)
        for t in np.arange(0, 8, 1 / 30):
            self.assertIsNone(capture.update(self.feature + t * .04, t))

    def test_face_loss_discards_partial_capture_and_restarts_settling(self):
        capture = StableCapture(0)
        for t in np.arange(0, 1.9, 1 / 30):
            self.assertIsNone(capture.update(self.feature, t))
        capture.missing(1.9)
        self.assertEqual(capture.progress, 0)
        for t in np.arange(2, 3.8, 1 / 30):
            self.assertIsNone(capture.update(self.feature, t))

    def test_camera_gap_does_not_count_as_a_hold(self):
        capture = StableCapture(0)
        for t in np.arange(1.3, 1.8, 1 / 30):
            capture.update(self.feature, t)
        self.assertIsNone(capture.update(self.feature, 2.2))
        self.assertEqual(capture.progress, 0)

    def test_lost_face_cannot_extend_timeout_forever(self):
        capture = StableCapture(0)
        for t in range(26):
            capture.missing(t)
        self.assertTrue(capture.expired(25))

    def test_nonfinite_samples_are_not_training_data(self):
        capture = StableCapture(0)
        self.assertIsNone(capture.update([.1, .2, float('nan'), .4], 2))
        self.assertEqual(len(capture.samples), 0)


class TrainingTests(unittest.TestCase):
    def samples(self):
        targets = np.asarray(ce.POINTS)
        centers = .25 + targets * .25
        return np.c_[centers, centers + .2], targets

    def test_noisy_training_generalizes_to_unseen_positions(self):
        x, y = self.samples()
        rng = np.random.default_rng(51)
        model = ce.PointerModel.fit(x + rng.normal(0, .001, x.shape), y, (320, 240), (1920, 1080))
        target = rng.uniform(.15, .85, (100, 2))
        center = .25 + target * .25
        prediction = np.array([model.normalized(v) for v in np.c_[center, center + .2]])
        self.assertLess(float(np.quantile(np.linalg.norm(prediction - target, axis=1), .95)), .03)
        self.assertLess(model.validation_error, .04)

    def test_scrambled_targets_are_rejected(self):
        x, y = self.samples()
        y = y[np.random.default_rng(87).permutation(9)]
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            ce.PointerModel.fit(x, y, (320, 240), (1920, 1080))

    def test_bad_single_position_is_rejected(self):
        x, y = self.samples()
        x[1] += .18
        with self.assertRaises(ValueError):
            ce.PointerModel.fit(x, y, (320, 240), (1920, 1080))

    def test_training_quality_survives_save_load(self):
        x, y = self.samples()
        model = ce.PointerModel.fit(x, y, (320, 240), (1920, 1080))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "pointer.npz"
            model.save(path)
            restored = ce.PointerModel.load(path, (320, 240), (1920, 1080))
            self.assertAlmostEqual(model.validation_error, restored.validation_error)

    def test_train_button_requests_training_and_enable_together(self):
        app = App.__new__(App)
        app.smoke = app.dashboard_test = False
        app.controller = ce.Controller((1920, 1080))
        app.calibrate = Mock()
        app.enable_pointer()
        app.calibrate.assert_called_once()
        self.assertTrue(app.controller.enable_requested.is_set())

    def test_queued_enable_cannot_undo_emergency_pause(self):
        c = ce.Controller((1920, 1080), mouse_factory=Mock)
        c.request("enable")
        c.pause()
        camera, mesh = Mock(), Mock()
        camera.get.side_effect = [320, 240]
        camera.read.return_value = (True, np.zeros((240, 320, 3), np.uint8))
        mesh.last_xyz = None
        mesh.process.side_effect = lambda frame: c.worker_stop.set()
        with patch.object(ce.fe, "open_camera", return_value=camera), \
             patch.object(ce.fe, "FaceMeshBackend", return_value=mesh), \
             patch.object(ce.native_input, "hotkey_kill_pressed", return_value=False):
            c._run()
        self.assertFalse(c.state.read().enabled)
        self.assertFalse(c.enable_requested.is_set())

    def test_fine_cursor_motion_is_not_lost_to_ten_pixel_deadzone(self):
        c = ce.Controller((1920, 1080))
        c.mouse, c.model = Mock(), Mock()
        c.model.predict.return_value = np.array([600., 400.])
        c.state.update(enabled=True, face=True, calibrated=True)
        values = (np.array([.2, .2, .6, .6]), 1., .02, .3, .3)
        for t in np.arange(0, 2, 1 / 30):
            c._track(values, t)
        c.mouse.reset_mock()
        c.model.predict.return_value = np.array([606., 400.])
        for t in np.arange(2, 4, 1 / 30):
            c._track(values, t)
        self.assertTrue(c.mouse.move.called)
        self.assertGreaterEqual(c.mouse.move.call_args.args[0], 603)

    def run_training_worker(self, folder, valid, late_cancel=False):
        c = ce.Controller((1920, 1080), mouse_factory=Mock)
        c.model_path = Path(folder) / "pointer.npz"
        x, _ = self.samples()
        position = [-1]
        def capture(now):
            position[0] += 1
            return StableCapture(now)
        clock = [0.]
        def now():
            clock[0] += .05
            if clock[0] > 60:
                raise AssertionError("Synthetic training failed to terminate")
            return clock[0]
        camera, mesh = Mock(), Mock()
        camera.get.side_effect = [320, 240]
        camera.read.return_value = (True, np.zeros((240, 320, 3), np.uint8))
        mesh.last_xyz = np.zeros((478, 3))
        mouse = Mock()
        c.mouse_factory = lambda: mouse
        put = c.events.put
        completed = [False]
        def event(item):
            put(item)
            if item[0] == "calibration_done":
                completed[0] = True
                c.training_result = c.state.read()
                if late_cancel:
                    c.request("cancel_calibration")
                else:
                    c.worker_stop.set()
        if late_cancel:
            def read(**kwargs):
                if completed[0]:
                    c.training_result = c.state.read()
                    c.worker_stop.set()
                    return False, None
                return True, np.zeros((240, 320, 3), np.uint8)
            camera.read.side_effect = read
        c.events.put = event
        c.request("calibrate")
        with patch.object(ce.fe, "open_camera", return_value=camera), \
             patch.object(ce.fe, "FaceMeshBackend", return_value=mesh), \
             patch.object(ce, "time", SimpleNamespace(monotonic=now)), \
             patch.object(ce, "StableCapture", side_effect=capture), \
             patch.object(ce, "measurements", side_effect=lambda *args: (x[position[0] if valid else 0], 1., .02, .3, .3)), \
             patch.object(ce.native_input, "hotkey_kill_pressed", return_value=False):
            c._run()
        mouse.move.assert_not_called()
        mouse.click.assert_not_called()
        camera.release.assert_called_once()
        return c

    def test_full_training_worker_saves_valid_profile_without_mouse_input(self):
        with tempfile.TemporaryDirectory() as folder:
            c = self.run_training_worker(folder, valid=True)
            self.assertTrue(c.state.read().calibrated)
            self.assertTrue(c.model_path.exists())
            self.assertEqual(c.state.read().calibration_progress, 1)
            ce.PointerModel.load(c.model_path, (320, 240), (1920, 1080))

    def test_failed_training_keeps_existing_profile_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            x, y = self.samples()
            path = Path(folder) / "pointer.npz"
            ce.PointerModel.fit(x, y, (320, 240), (1920, 1080)).save(path)
            original = path.read_bytes()
            c = self.run_training_worker(folder, valid=False)
            self.assertEqual(path.read_bytes(), original)
            self.assertIn("Move your head", c.state.read().hint)
            self.assertEqual(c.training_result.calib_step, "done")
            self.assertEqual(c.state.read().calibration_text, c.state.read().hint)
            self.assertLess(c.state.read().calibration_progress, 1)

    def test_delayed_cancel_cannot_overwrite_saved_calibration(self):
        with tempfile.TemporaryDirectory() as folder:
            c = self.run_training_worker(folder, valid=True, late_cancel=True)
            self.assertEqual(c.training_result.calib_step, "done")
            self.assertEqual(c.state.read().calibration_text, "Head calibration saved.")
            self.assertEqual(c.state.read().calibration_progress, 1)


if __name__ == "__main__":
    unittest.main()
