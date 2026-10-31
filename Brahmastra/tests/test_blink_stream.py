import unittest
from eyesync_gestures import BlinkDetector


class BlinkStreamTests(unittest.TestCase):
    def detector(self):
        b = BlinkDetector(None, .55, .7, .3, 1.5, .6)
        b.baseline = .28
        return b

    def test_continuous_deliberate_blink_clicks_once(self):
        b = self.detector()
        clicks = []
        for i, ear in enumerate([.28] * 30 + [.07] * 15 + [.28] * 30):
            click = b.update(ear, ear, 1 + i / 30)[1]
            if click:
                clicks.append(click)
        self.assertEqual(clicks, ['left'])

    def test_gap_cannot_complete_hold(self):
        b = self.detector()
        for i in range(5):
            b.update(.07, .07, 1 + i / 30)
        self.assertEqual(b.update(.07, .07, 2)[1], '')
        self.assertEqual(b.cued, 0)
        self.assertEqual(b.closed_since, 2)

    def test_invalid_eyes_cancel_armed_click(self):
        for bad in (float('nan'), float('inf'), 0, -1):
            b = self.detector()
            for i in range(15):
                b.update(.07, .07, 1 + i / 30)
            b.update(bad, .07, 1.5)
            self.assertIsNone(b.closed_since)
            self.assertEqual(b.cued, 0)
            for i in range(5):
                self.assertEqual(b.update(.28, .28, 1.6 + i / 30)[1], '')
