"""Calibration and shutdown regressions using synthetic data only."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from calibration import StableCapture
import control_engine as ce


class LowFrameRateTests(unittest.TestCase):
    def test_steady_ten_fps_camera_can_complete_calibration(self):
        capture = StableCapture(0)
        result = None
        for now in np.arange(0, 4, .1):
            result = capture.update([.25, .25, .55, .65], now)
            if result is not None:
                break
        self.assertIsNotNone(result)
        self.assertGreaterEqual(now, capture.SETTLE_SECONDS + capture.HOLD_SECONDS)
        np.testing.assert_allclose(result, [.25, .25, .55, .65])

    def test_moving_ten_fps_camera_still_cannot_train(self):
        capture = StableCapture(0)
        for now in np.arange(0, 5, .1):
            self.assertIsNone(capture.update(np.full(4, .2 + now * .04), now))


class ShutdownTests(unittest.TestCase):
    def test_cleanup_failures_do_not_skip_other_resources_or_stopped_state(self):
        for resource in ("camera", "mesh", "both"):
            with self.subTest(resource=resource), tempfile.TemporaryDirectory() as folder:
                controller = ce.Controller((1920, 1080), mouse_factory=Mock)
                controller.model_path = Path(folder) / "absent.npz"
                controller.worker_stop.set()
                controller.state.update(face=True, enabled=True)
                camera, mesh = Mock(), Mock()
                camera.get.side_effect = [320, 240]
                if resource in ("camera", "both"):
                    camera.release.side_effect = RuntimeError("camera cleanup failed")
                if resource in ("mesh", "both"):
                    mesh.close.side_effect = RuntimeError("mesh cleanup failed")
                with patch.object(ce.fe, "open_camera", return_value=camera), \
                     patch.object(ce.fe, "FaceMeshBackend", return_value=mesh), \
                     patch.object(ce, "MODEL_PATH", Path(folder) / "no-public-model"), \
                     self.assertLogs(ce.log, level="ERROR"):
                    controller._run()
                camera.release.assert_called_once()
                mesh.close.assert_called_once()
                snap = controller.state.read()
                self.assertFalse(snap.running)
                self.assertFalse(snap.enabled)
                self.assertFalse(snap.face)
                self.assertIsNone(controller.mouse)
                self.assertIsNone(controller.state.preview)
                self.assertEqual(controller.events.get_nowait(), ("started", None))
                self.assertEqual(controller.events.get_nowait(), ("stopped", None))


class InputRecoveryTests(unittest.TestCase):
    def test_blocked_mouse_does_not_stop_camera_worker(self):
        c = ce.Controller((1920, 1080), mouse_factory=Mock)
        c.model_path = Path('missing-test-profile.npz')
        camera, mesh = Mock(), Mock()
        camera.get.side_effect = [320, 240]
        camera.read.return_value = (True, np.zeros((240, 320, 3), dtype=np.uint8))
        mesh.last_xyz = np.zeros((478, 3))
        mesh.last_blendshapes = {}
        mesh.last_pose = (0., 0.)
        def blocked(*args):
            c.worker_stop.set()
            raise OSError('Windows blocked input')
        with patch.object(ce.fe, 'open_camera', return_value=camera), \
             patch.object(ce.fe, 'FaceMeshBackend', return_value=mesh), \
             patch.object(ce, 'MODEL_PATH', Path('missing-public-model.npz')), \
             patch.object(ce, 'measurements', return_value=(np.zeros(4), 1., .02, .3, .3)), \
             patch.object(ce.fe, 'nose_pointer', return_value=(np.zeros(2), 100.)), \
             patch.object(c, '_track', side_effect=blocked), \
             self.assertLogs(ce.log, level='WARNING') as logs:
            c._run()
        self.assertIn('Mouse input unavailable; pointer paused', '\n'.join(logs.output))
        self.assertNotIn('Tracking stopped', '\n'.join(logs.output))
        self.assertTrue(c.state.read().paused)
        self.assertFalse(c.state.read().enabled)
