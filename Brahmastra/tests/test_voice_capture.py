"""Capture lifecycle checks use fake PortAudio streams; never open a microphone."""
import sys
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np

from voice_capture import Microphone


class Stream:
    def __init__(self, start_error=False, stop_error=False):
        self.start_error, self.stop_error = start_error, stop_error
        self.started = self.stopped = self.closed = 0

    def start(self):
        self.started += 1
        if self.start_error:
            raise OSError("Unsupported sample rate")

    def stop(self):
        self.stopped += 1
        if self.stop_error:
            raise OSError("Disconnected microphone")

    def close(self):
        self.closed += 1


class CaptureTests(unittest.TestCase):
    def sounddevice(self, streams, native=16000):
        return types.SimpleNamespace(
            query_devices=Mock(return_value=dict(name="Test microphone", default_samplerate=native)),
            InputStream=Mock(side_effect=streams))

    def open_mic(self, sd):
        self.modules = patch.dict(sys.modules, sounddevice=sd)
        self.modules.start()
        self.addCleanup(self.modules.stop)
        mic = Microphone()
        self.addCleanup(mic.close)
        mic.open()
        return mic

    def send(self, mic, data=None, status=False):
        if data is None:
            data = np.full((800, 1), .1, np.float32)
        mic._callback(data, len(data), None, status)
        return data

    def test_opening_retains_original_mono_rate_and_block_size(self):
        sd = self.sounddevice([Stream()])
        mic = self.open_mic(sd)
        kwargs = sd.InputStream.call_args.kwargs
        self.assertEqual((kwargs['samplerate'], kwargs['channels'], kwargs['blocksize']), (16000, 1, 800))
        self.assertEqual(mic.name, "Test microphone")

    def test_callback_copies_audio_before_portaudio_reuses_memory(self):
        mic = self.open_mic(self.sounddevice([Stream()]))
        data = self.send(mic)
        data[:] = 0
        np.testing.assert_allclose(mic.read(.01), .1)

    def test_fallback_closes_failed_start_before_retrying(self):
        failed, working = Stream(start_error=True), Stream()
        sd = self.sounddevice([failed, working], native=48000)
        mic = self.open_mic(sd)
        self.assertEqual(failed.closed, 1)
        self.assertEqual([call.kwargs['samplerate'] for call in sd.InputStream.call_args_list], [16000, 48000])
        self.send(mic, np.full((2400, 1), .1, np.float32))
        self.assertEqual(mic.read(.01).shape, (800,))

    def test_identical_native_rate_is_not_retried(self):
        failed = Stream(start_error=True)
        sd = self.sounddevice([failed], native=16000)
        with patch.dict(sys.modules, sounddevice=sd):
            with self.assertRaises(OSError):
                Microphone().open()
        self.assertEqual(sd.InputStream.call_count, 1)
        self.assertEqual(failed.closed, 1)

    def test_all_failed_streams_are_closed(self):
        first, second = Stream(start_error=True), Stream(start_error=True)
        sd = self.sounddevice([first, second], native=44100)
        mic = Microphone()
        with patch.dict(sys.modules, sounddevice=sd):
            with self.assertRaises(OSError):
                mic.open()
        self.assertIsNone(mic.stream)
        self.assertEqual((first.closed, second.closed), (1, 1))

    def test_close_still_releases_stream_if_stop_raises(self):
        stream = Stream(stop_error=True)
        mic = self.open_mic(self.sounddevice([stream]))
        self.send(mic)
        mic.close()
        self.assertEqual(stream.closed, 1)
        self.assertIsNone(mic.stream)
        self.assertTrue(mic.blocks.empty())
        mic.close()
        self.assertEqual(stream.closed, 1)

    def test_bounded_queue_overflow_fails_instead_of_transcribing_old_audio(self):
        mic = self.open_mic(self.sounddevice([Stream()]))
        for _ in range(mic.MAX_BLOCKS + 1):
            self.send(mic)
        self.assertEqual(mic.blocks.qsize(), mic.MAX_BLOCKS)
        self.assertGreaterEqual(mic.MAX_BLOCKS * .05, 30)     # the slowest phrase must fit
        with self.assertRaisesRegex(OSError, "fell behind"):
            mic.read(.01)
        with self.assertRaises(OSError):
            mic.read(.01)

    def test_portaudio_status_invalidates_already_queued_audio(self):
        mic = self.open_mic(self.sounddevice([Stream()]))
        self.send(mic)
        self.send(mic, status=True)
        with self.assertRaisesRegex(OSError, "interrupted or overflowed"):
            mic.read(.01)

    def test_inference_can_check_for_capture_failure_before_typing(self):
        mic = self.open_mic(self.sounddevice([Stream()]))
        self.send(mic)
        mic.read(.01)
        self.send(mic, status=True)
        with self.assertRaisesRegex(OSError, "interrupted or overflowed"):
            mic.check_health()

    def test_nonfinite_audio_is_a_capture_failure(self):
        for value in (float('nan'), float('inf'), -float('inf')):
            with self.subTest(value=value):
                mic = self.open_mic(self.sounddevice([Stream()]))
                self.send(mic, np.full((800, 1), value, np.float32))
                with self.assertRaisesRegex(OSError, "invalid audio"):
                    mic.read(.01)
                mic.close()

    def test_malformed_audio_shape_is_reported_on_consumer_thread(self):
        mic = self.open_mic(self.sounddevice([Stream()]))
        self.send(mic, np.zeros((800, 2), np.float32))
        with self.assertRaisesRegex(OSError, "invalid audio"):
            mic.read(.01)

    def test_reopening_discards_old_audio_and_old_faults(self):
        sd = self.sounddevice([Stream(), Stream()])
        mic = self.open_mic(sd)
        self.send(mic, status=True)
        mic.close()
        mic.open()
        self.send(mic, np.full((800, 1), .25, np.float32))
        np.testing.assert_allclose(mic.read(.01), .25)

    def test_callbacks_after_close_are_ignored(self):
        mic = self.open_mic(self.sounddevice([Stream()]))
        mic.close()
        self.send(mic)
        self.assertTrue(mic.blocks.empty())
        self.assertIsNone(mic.read(.001))

    def test_absent_device_is_reported_without_creating_stream(self):
        sd = self.sounddevice([])
        sd.query_devices.side_effect = RuntimeError("No default input")
        with patch.dict(sys.modules, sounddevice=sd):
            with self.assertRaisesRegex(OSError, "No microphone found"):
                Microphone().open()
        sd.InputStream.assert_not_called()


if __name__ == '__main__':
    unittest.main()
