import unittest
from unittest.mock import Mock, patch
import numpy as np
import control_engine as ce

class GestureRoutingTests(unittest.TestCase):
    def setUp(self):
        self.sounds=patch.multiple(ce.desktop,play_tick_sound=Mock(),play_warning_sound=Mock(),play_click_sound=Mock())
        self.sounds.start()
        self.addCleanup(self.sounds.stop)
        self.c=ce.Controller((1920,1080))
        self.c.model=Mock()
        self.c.model.predict.return_value=np.array([600.,400.])
        self.c.mouse=Mock()
        self.c.state.update(enabled=True,face=True,calibrated=True,mouse_requested=True)
        self.t=1.
        self.feed(1.)
        self.c.mouse.reset_mock()

    def feed(self, seconds, eyes=(.3,.3), **fields):
        frame=dict(ears=eyes,scores={'mouthSmileLeft':0.,'mouthSmileRight':0.,'jawOpen':0.},
                   pose=(0.,0.),mar=.02,mouth_shift=0.,roll=0.,head=np.zeros(2),winks_ok=True)
        frame.update(fields)
        for _ in range(int(seconds*30)):
            self.t+=1/30
            self.c._track((np.array([.2,.2,.6,.6]),1.,.02,*eyes),self.t,frame)

    def test_blink_left_and_long_blink_right(self):
        self.feed(.6,eyes=(.07,.07));self.feed(.4)
        self.c.mouse.click.assert_called_once_with('left')
        self.c.mouse.reset_mock()
        self.feed(1.);self.feed(1.6,eyes=(.07,.07));self.feed(.4)
        self.c.mouse.click.assert_called_once_with('right')

    def test_long_closure_toggles_voice_once_and_never_clicks(self):
        self.feed(3.,eyes=(.07,.07));self.feed(.5)
        self.assertEqual(self.c.state.voice_requests.get_nowait(),(None,'eyes closed'))
        self.assertTrue(self.c.state.voice_requests.empty())
        self.c.mouse.click.assert_not_called()

    def test_smile_pauses_and_resumes_without_clicking(self):
        smile={'mouthSmileLeft':.95,'mouthSmileRight':.95,'jawOpen':0.}
        self.feed(1.8,scores=smile)
        self.assertTrue(self.c.state.read().paused)
        self.c.mouse.click.assert_not_called()
        self.feed(1.);self.feed(1.8,scores=smile);self.feed(.3)
        self.assertFalse(self.c.state.read().paused)
        self.assertTrue(self.c.state.read().enabled)

    def test_tilt_scrolls_and_freezes_pointer(self):
        self.feed(.3,roll=35.)
        self.c.mouse.reset_mock()
        self.feed(1.,roll=35.)
        self.assertTrue(self.c.mouse.scroll.called)
        self.assertGreater(self.c.mouse.scroll.call_args.args[0],0)
        self.c.mouse.move.assert_not_called()
        self.c.mouse.click.assert_not_called()

    def test_wide_mouth_requests_quit_without_right_click(self):
        self.feed(3.2,mar=.8,scores={'jawOpen':.8})
        events=[]
        while not self.c.events.empty(): events.append(self.c.events.get_nowait())
        self.assertIn(('dashboard_quit',None),events)
        self.c.mouse.click.assert_not_called()

    def test_speech_does_not_trigger_mouth_quit(self):
        self.c.state.update(voice='listening')
        self.feed(4.,mar=.8,scores={'jawOpen':.8,'mouthSmileLeft':.9,'mouthSmileRight':.9})
        self.assertTrue(self.c.events.empty())
        self.c.mouse.click.assert_not_called()

    def test_mouth_left_requests_recenter(self):
        self.feed(1.8,mouth_shift=.035,scores={'mouthLeft':.8,'mouthSmileLeft':.2})
        self.assertEqual(self.c.commands.get_nowait(),('recenter',None))
        self.c.mouse.click.assert_not_called()

    def test_mouth_right_requests_personal_calibration(self):
        self.feed(3.3,mouth_shift=-.035,scores={'mouthRight':.8,'mouthSmileRight':.2})
        self.assertEqual(self.c.events.get_nowait(),('dashboard_calibrate','head'))
        self.c.mouse.click.assert_not_called()

if __name__=='__main__': unittest.main()
