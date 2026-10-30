"""Camera worker failure regressions, using fake capture objects only."""
import threading
import unittest
from unittest.mock import Mock

import numpy as np

import face_engine as fe


class CameraLifecycleTests(unittest.TestCase):
    def camera(self):
        camera = fe.LatestFrameCamera.__new__(fe.LatestFrameCamera)
        camera._cap = Mock()
        camera._cond = threading.Condition()
        camera._frame = np.zeros((2, 2, 3), dtype=np.uint8)
        camera._seq, camera._read_seq = 1, 0
        camera._failed, camera._running = False, True
        return camera

    def test_capture_exception_marks_closed_and_discards_stale_frame(self):
        camera = self.camera()
        camera._cap.read.side_effect = OSError("device disconnected")
        with self.assertLogs(fe.log, level="ERROR"):
            camera._grab()
        self.assertTrue(camera._failed)
        self.assertFalse(camera.isOpened())
        self.assertEqual(camera.read(timeout=0), (False, None))
        camera._cap.release.assert_called_once()

    def test_capture_failure_wakes_reader_even_if_release_fails(self):
        camera = self.camera()
        camera._read_seq = camera._seq
        entered, fail = threading.Event(), threading.Event()
        def read():
            entered.set()
            fail.wait(2)
            raise RuntimeError("backend lost")
        camera._cap.read.side_effect = read
        camera._cap.release.side_effect = RuntimeError("release failed")
        result = []
        with self.assertLogs(fe.log, level="ERROR"):
            worker = threading.Thread(target=camera._grab)
            reader = threading.Thread(target=lambda: result.append(camera.read(timeout=2)))
            worker.start()
            self.assertTrue(entered.wait(1))
            reader.start()
            fail.set()
            reader.join(.5)
            worker.join(.5)
        self.assertFalse(reader.is_alive(), "Dead capture must wake waiting readers")
        self.assertFalse(worker.is_alive())
        self.assertEqual(result, [(False, None)])
        self.assertFalse(camera.isOpened())
