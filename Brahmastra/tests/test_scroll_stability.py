import unittest
import numpy as np
import test_ai_gesture_routing as routing


class ScrollStabilityTests(unittest.TestCase):
    setUp = routing.GestureRoutingTests.setUp
    feed = routing.GestureRoutingTests.feed

    def test_right_tilt_scrolls_down_and_straightening_stops(self):
        self.feed(1.,roll=-35.)
        self.assertLess(self.c.mouse.scroll.call_args.args[0],0)
        self.assertEqual(self.c.state.read().scrolling,-1)
        self.feed(.3)
        self.c.mouse.reset_mock()
        self.feed(.4)
        self.c.mouse.scroll.assert_not_called()
        self.assertEqual(self.c.state.read().scrolling,0)

    def test_pause_stops_scrolling_immediately(self):
        self.feed(1.,roll=35.)
        self.c.pause()
        self.assertEqual(self.c.state.read().scrolling,0)
        self.c.mouse.reset_mock()
        self.feed(1.,roll=35.)
        self.c.mouse.scroll.assert_not_called()
        self.c.mouse.move.assert_not_called()

    def test_disabled_gestures_cannot_scroll(self):
        self.c.state.update(gestures=False)
        self.feed(1.,roll=35.)
        self.c.mouse.scroll.assert_not_called()

    def test_tilt_rewinds_to_position_before_lean_and_settles_on_release(self):
        origin=self.c.last_xy.copy()
        self.c.model.predict.return_value=np.array([1000.,600.])
        self.feed(.2,roll=14.)
        self.feed(.7,roll=35.)
        np.testing.assert_allclose(self.c.last_xy,origin)
        self.feed(.1)  # median window clears the tilt first
        self.c.mouse.reset_mock()
        self.feed(.1)
        self.c.mouse.move.assert_not_called()
        self.c.mouse.click.assert_not_called()

    def test_camera_gap_cancels_a_partly_held_quit(self):
        self.feed(1.8,mar=.8,scores={'jawOpen':.8})
        self.t+=2.
        self.feed(1.5,mar=.8,scores={'jawOpen':.8})
        self.assertTrue(self.c.events.empty())

    def test_camera_gap_cancels_a_partly_held_blink(self):
        self.feed(.6,eyes=(.07,.07))
        self.t+=1.
        self.feed(.1,eyes=(.07,.07))
        self.feed(.5)
        self.c.mouse.click.assert_not_called()
        self.assertTrue(self.c.state.voice_requests.empty())

    def test_invalid_roll_cannot_scroll_or_move(self):
        self.feed(.5,roll=float('nan'))
        self.c.mouse.scroll.assert_not_called()
        self.c.mouse.move.assert_not_called()
        self.c.mouse.click.assert_not_called()

    def test_face_loss_stops_scroll_and_requires_fresh_tilt_hold(self):
        self.feed(1.,roll=35.)
        self.c._lost_face()
        self.assertEqual(self.c.state.read().scrolling,0)
        self.c.mouse.reset_mock()
        self.c.state.update(face=True)
        self.feed(.2,roll=35.)
        self.c.mouse.scroll.assert_not_called()

    def test_scroll_does_not_interfere_with_voice_typing(self):
        self.c.state.update(voice='listening')
        self.feed(1.,roll=35.)
        self.assertTrue(self.c.mouse.scroll.called)
        self.c.mouse.click.assert_not_called()

    def test_quit_event_is_emitted_once(self):
        self.feed(4.,mar=.8,scores={'jawOpen':.8})
        events=[]
        while not self.c.events.empty(): events.append(self.c.events.get_nowait())
        self.assertEqual(events.count(('dashboard_quit',None)),1)

    def test_pause_cancels_dwell_timer(self):
        self.c.state.update(click_mode='dwell')
        self.feed(.8)
        self.c.pause()
        self.t+=5.
        self.c.request_enable()
        self.feed(.3)
        self.c.mouse.click.assert_not_called()


if __name__=='__main__': unittest.main()
