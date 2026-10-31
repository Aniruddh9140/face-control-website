import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import control_engine as ce
from eyesync_app import App


class CameraSelectionTests(unittest.TestCase):
    def test_selection_is_saved_and_profiles_are_separate(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(ce, 'CAMERA_CHOICE_PATH', Path(folder) / 'camera.txt'):
            controller = ce.Controller((1920, 1080))
            original = controller.model_path
            controller.request_enable()
            controller.select_camera(1)
            self.assertFalse(controller.enable_requested.is_set())
            self.assertFalse(controller.state.read().mouse_requested)
            self.assertEqual(controller.camera, 1)
            self.assertEqual(controller.model_path.name, 'pointer-camera-1.npz')
            self.assertEqual(ce.Controller((1920, 1080), camera=None).camera, 1)
            self.assertEqual(ce.Controller((1920, 1080), camera=0).camera, 0)
            controller.select_camera(0)
            self.assertEqual(controller.model_path, original)

    def test_active_worker_must_release_camera_before_selection(self):
        controller = ce.Controller((1920, 1080))
        controller.thread = Mock()
        controller.thread.is_alive.return_value = True
        with patch.object(ce, 'CAMERA_CHOICE_PATH') as choice:
            with self.assertRaisesRegex(ValueError, 'Stop'):
                controller.select_camera(1)
            choice.parent.mkdir.assert_not_called()
        self.assertEqual(controller.camera, 0)

    def test_invalid_choice_and_unwritable_settings_keep_current_camera(self):
        controller = ce.Controller((1920, 1080))
        for index in (-1, 10, True, '1'):
            with self.assertRaises(ValueError):
                controller.select_camera(index)
        with patch.object(ce, 'CAMERA_CHOICE_PATH') as choice:
            choice.parent.mkdir.side_effect = OSError('Cannot save camera choice')
            with self.assertRaises(OSError):
                controller.select_camera(1)
        self.assertEqual(controller.camera, 0)

    def test_invalid_saved_camera_falls_back_to_default(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(ce, 'CAMERA_CHOICE_PATH', Path(folder) / 'camera.txt') as choice:
            self.assertEqual(ce.saved_camera(), 0)
            for value in ('oops', '-1', '10'):
                choice.write_text(value)
                self.assertEqual(ce.saved_camera(), 0)

    def test_ui_restores_selection_after_error(self):
        app = App.__new__(App)
        app.controller = Mock(camera=0)
        app.camera_choice = Mock()
        app.camera_choice.get.return_value = 'Camera 1'
        app.controller.select_camera.side_effect = ValueError('Stop camera first')
        app.choose_camera()
        app.camera_choice.set.assert_called_once_with('Camera 0')
        app.controller.state.update.assert_called_once_with(hint='Stop camera first')
