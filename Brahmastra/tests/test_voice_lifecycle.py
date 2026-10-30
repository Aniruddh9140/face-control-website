"""Full control/session cancellation while inference is blocked; all I/O is fake."""
import threading
import unittest
from unittest.mock import Mock
import numpy as np

import control_engine as ce
import voice_typing as vt


class LifecycleTests(unittest.TestCase):
    def test_stop_releases_mic_before_inference_finishes_and_discards_text(self):
        recognizing, release, closed = threading.Event(), threading.Event(), threading.Event()
        class Recognizer:
            def load(self): pass
            def close(self): pass
            def __call__(self, audio):
                recognizing.set()
                if not release.wait(4): raise TimeoutError('Test inference was not released')
                return 'Never deliver this cancelled speech.'
        class Mic:
            name='Fake input'
            delivered=False
            def open(self): pass
            def close(self): closed.set()
            def read(self, timeout):
                if not self.delivered:
                    self.delivered=True
                    return np.r_[np.ones(16000,np.float32)*.1,np.zeros(16000,np.float32)]
                closed.wait(.01)
                return None
        state=ce.State()
        typist=Mock()
        typist.run.side_effect=lambda ops:len(vt.key_events(ops))
        tracker=Mock()
        tracker.peek.return_value=tracker.target.return_value=101
        tracker.elevated.return_value=False
        voice=ce.RoutedVoiceTyping(state,ce.queue.Queue(),recognizer=Recognizer(),mic_factory=Mic,
                vad_factory=lambda:lambda block:float(np.any(block)),typist=typist,
                tracker=tracker,check=lambda:'',sounds=False).start()
        try:
            vt.request(state,True,'test')
            self.assertTrue(recognizing.wait(3),'Session never reached fake inference')
            vt.request(state,False,'stop button')
            self.assertTrue(closed.wait(2),'Microphone was retained until inference finished')
            self.assertTrue(voice.active(),'Inference should still be blocked at this assertion')
            release.set()
            voice.session.join(2)
            self.assertFalse(voice.active())
            typist.run.assert_not_called()
            self.assertEqual(state.read().voice,'')
        finally:
            release.set()
            voice.close()
        self.assertFalse(voice.thread.is_alive())
        self.assertFalse(state.stop.is_set())

if __name__=='__main__': unittest.main()
