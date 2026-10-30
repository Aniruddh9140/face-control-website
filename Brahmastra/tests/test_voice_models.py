"""Accurate voice mode: choosing the model and fetching it. No network or real model."""
import hashlib
import io
import queue
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import control_engine as ce
from eyesync_app import App
import voice_models as vm
import voice_typing as vt


class AccurateModeTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        root = Path(folder.name)
        self.body = b"model" * 1000
        for name, value in (("FOLDER", root / "models"), ("FLAG", root / "data" / "on"),
                            ("BYTES", len(self.body)), ("SHA1", hashlib.sha1(self.body).hexdigest())):
            patcher = patch.object(vm, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def fetch(self, body):
        seen = []
        with patch.object(vm.urllib.request, "urlopen", return_value=io.BytesIO(body)):
            vm.fetch(seen.append)
        return seen

    def install(self):
        self.fetch(self.body)

    def test_download_is_verified_then_used(self):
        self.assertEqual(self.fetch(self.body)[-1], 100)
        vm.set_enabled(True)
        self.assertEqual(vm.recognizer().model_path, vm.model_path())

    def test_a_file_failing_its_checksum_is_never_kept(self):
        with self.assertRaisesRegex(ValueError, "safety check"):
            self.fetch(b"x" * len(self.body))
        self.assertFalse(vm.ready())
        self.assertEqual(list(vm.FOLDER.iterdir()), [])

    def test_a_broken_download_leaves_nothing_and_can_be_retried(self):
        with patch.object(vm.urllib.request, "urlopen", side_effect=OSError("offline")):
            with self.assertRaises(OSError):
                vm.fetch(print)
        self.install()
        self.assertTrue(vm.ready())

    def test_ticking_again_repairs_a_damaged_model_of_the_right_size(self):
        self.install()
        vm.model_path().write_bytes(b"x" * len(self.body))
        self.assertTrue(vm.ready())
        self.install()
        self.assertEqual(vm.model_path().read_bytes(), self.body)

    def test_a_good_model_is_not_downloaded_again(self):
        self.install()
        with patch.object(vm.urllib.request, "urlopen") as urlopen:
            vm.fetch(print)
        urlopen.assert_not_called()

    def test_a_second_toggle_during_a_download_never_says_ready(self):
        state = ce.State()
        app = SimpleNamespace(controller=SimpleNamespace(state=state))
        vm.set_enabled(True)
        self.install()
        vm.model_path().write_bytes(b"x" * len(self.body))      # same size: looks ready
        with vm._fetching, patch.object(vm.urllib.request, "urlopen") as urlopen:
            App._fetch_accurate(app)
        urlopen.assert_not_called()
        self.assertEqual(state.read().voice_hint, "")

    def test_fast_model_unless_accurate_is_on_and_ready(self):
        fast = vt.Recognizer().model_path
        self.assertEqual(vm.recognizer().model_path, fast)
        vm.set_enabled(True)
        self.assertEqual(vm.recognizer().model_path, fast)      # chosen, not downloaded yet
        self.install()
        vm.set_enabled(False)
        self.assertEqual(vm.recognizer().model_path, fast)

    def test_setting_survives_restart_and_turns_off(self):
        vm.set_enabled(True)
        self.assertTrue(vm.enabled())
        vm.set_enabled(False)
        vm.set_enabled(False)
        self.assertFalse(vm.enabled())

    def test_each_session_uses_the_chosen_model_but_keeps_an_injected_one(self):
        self.install()
        vm.set_enabled(True)
        voice = ce.RoutedVoiceTyping(ce.State(), queue.Queue(), check=lambda: "", sounds=False,
                                     vad_factory=lambda: None)
        with patch.object(vt.Recognizer, "load", side_effect=OSError("stop here")):
            voice._run_session("test")
        self.assertEqual(voice.recognizer.model_path, vm.model_path())
        injected = vt.Recognizer()
        voice = ce.RoutedVoiceTyping(ce.State(), queue.Queue(), recognizer=injected, check=lambda: "",
                                     sounds=False, vad_factory=lambda: None)
        with patch.object(vt.Recognizer, "load", side_effect=OSError("stop here")):
            voice._run_session("test")
        self.assertIs(voice.recognizer, injected)


if __name__ == "__main__":
    unittest.main()
