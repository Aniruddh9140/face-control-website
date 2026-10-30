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
