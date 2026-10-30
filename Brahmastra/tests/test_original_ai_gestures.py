"""Unchanged detector regression cases from original EyeSync."""
from __future__ import annotations
import unittest, math, time, types, threading
from collections import deque
import numpy as np
import eyesync_gestures as li
import face_engine as fe
FPS = 30

class BlinkTests(unittest.TestCase):
    O, C = (0.28, 0.07)

    def buttons(self, seq, winks_ok=True) -> list[str]:
        """seq of (left eye EAR, right eye EAR) at 30 FPS -> the clicks, e.g. ["left", "right"]."""
        b = li.BlinkDetector(None, 0.55, 0.7, 0.25, 1.5, 0.6)
        b.baseline = self.O
        t, out = (0.0, [])
        for left, right in seq:
            t += 1 / FPS
            click = b.update(left, right, t, None, winks_ok)[1]
            if click:
                out.append(click)
        return out

    def clicks(self, seq) -> int:
        return len(self.buttons(seq))

    def test_cases(self):
        O, C = (self.O, self.C)
        cases = {'involuntary blink 100 ms': ([(O, O)] * 20 + [(C, C)] * 3 + [(O, O)] * 20, 0), 'deliberate blink 400 ms': ([(O, O)] * 20 + [(C, C)] * 12 + [(O, O)] * 20, 1), 'deliberate blink with a 1-frame glitch': ([(O, O)] * 20 + [(C, C)] * 6 + [(O, O)] + [(C, C)] * 6 + [(O, O)] * 20, 1), 'flicker at the threshold': ([(O, O)] * 20 + [(C, C)] * 5 + [(0.17, 0.17), (0.14, 0.14)] * 4 + [(O, O)] * 20, 1), 'quick look down': ([(O, O)] * 20 + [(0.21, 0.21)] * 15 + [(O, O)] * 20, 0), 'resting eyes 3 s': ([(O, O)] * 20 + [(C, C)] * 90 + [(O, O)] * 20, 0), 'single-frame glitch': ([(O, O)] * 20 + [(C, C)] + [(O, O)] * 20, 0)}
        for name, (seq, expected) in cases.items():
            with self.subTest(name):
                self.assertEqual(self.clicks(seq), expected)

    def test_blink_seen_only_by_blink_score(self):
        """With glasses the eyelid points can miss a blink; the learned score still rises."""
        b = li.BlinkDetector(None, 0.55, 0.7, 0.2, 1.5, 0.6)
        b.baseline = self.O
        t, clicks = (0.0, 0)
        seq = [(0.02, 0.02)] * 30 + [(0.35, 0.33)] * 10 + [(0.02, 0.02)] * 20
        for sc in seq:
            t += 1 / FPS
            clicks += bool(b.update(self.O, self.O, t, sc)[1])
        self.assertEqual(clicks, 1)

    def test_looking_down_is_not_a_blink(self):
        b = li.BlinkDetector(None, 0.55, 0.7, 0.2, 1.5, 0.6)
        b.baseline = self.O
        t, clicks = (0.0, 0)
        seq = [(self.O, 0.03)] * 30 + [(0.22, 0.15)] * 20 + [(self.O, 0.03)] * 20
        for ear, sc in seq:
            t += 1 / FPS
            clicks += bool(b.update(ear, ear, t, (sc, sc))[1])
        self.assertEqual(clicks, 0)

    def test_winks(self):
        O, C = (self.O, self.C)
        pad = [(O, O)] * 20
        cases = {'both eyes 400 ms = left click': ([(C, C)] * 12, ['left']), 'left eye 400 ms = left click': ([(C, O)] * 12, ['left']), 'right eye 400 ms = right click': ([(O, C)] * 12, ['right']), 'right wink, other eye squinting a little': ([(0.21, C)] * 12, ['right']), 'involuntary one-eye twitch 100 ms': ([(O, C)] * 3, []), 'wink that turns into a blink = left click': ([(O, C)] * 4 + [(C, C)] * 10, ['left']), 'one eye resting 3 s': ([(O, C)] * 90, [])}
        for name, (closure, expected) in cases.items():
            with self.subTest(name):
                self.assertEqual(self.buttons(pad + closure + pad), expected)
        self.assertEqual(self.buttons(pad + [(O, C)] * 12 + pad, winks_ok=False), [])

    def scored(self, seq) -> list[str]:
        """seq of ((left EAR, right EAR), (left score, right score)) at 30 FPS -> the clicks."""
        b = li.BlinkDetector(None, 0.55, 0.7, 0.2, 1.5, 0.6)
        b.baseline = self.O
        t, out = (0.0, [])
        for ears, scores in seq:
            t += 1 / FPS
            click = b.update(ears[0], ears[1], t, scores, True)[1]
            if click:
                out.append(click)
        return out

    def test_winks_when_both_blink_scores_rise(self):
        """
        MediaPipe's blink score rises for the OPEN eye too during a wink (it
        squints a little). Judged by the scores, a right wink was a blink: a
        LEFT click. The eyelids tell which eye winked.
        """
        O, C = (self.O, self.C)
        pad = [((O, O), (0.03, 0.03))] * 30
        cases = {'right wink': ([((O, C), (0.45, 0.9))] * 12, ['right']), 'left wink': ([((C, O), (0.9, 0.45))] * 12, ['left']), 'right wink, the left eye squinting': ([((0.21, C), (0.5, 0.9))] * 12, ['right']), 'blink, one eyelid a frame behind': ([((C, 0.2), (0.9, 0.6))] + [((C, C), (0.9, 0.9))] * 11, ['left']), 'blink': ([((C, C), (0.9, 0.9))] * 12, ['left']), 'short right twitch': ([((O, C), (0.45, 0.9))] * 3, [])}
        for name, (closure, expected) in cases.items():
            with self.subTest(name):
                self.assertEqual(self.scored(pad + closure + pad), expected)

    def ticked(self, seq, **kw):
        """seq of (left EAR, right EAR) at 30 FPS -> (clicks, [(time, tick)]) - see BlinkDetector.cue."""
        b = li.BlinkDetector(None, 0.55, 0.7, li.BLINK_MIN_S, li.BLINK_MAX_S, 0.6, **kw)
        b.baseline = self.O
        t, clicks, ticks = (0.0, [], [])
        for left, right in seq:
            t += 1 / FPS
            click = b.update(left, right, t, None, True)[1]
            if b.cue:
                ticks.append((t, b.cue))
            if click:
                clicks.append(click)
        return (clicks, ticks)

    def test_long_blink_is_a_right_click(self):
        """A real user's right wink closed both eyes: a longer blink is the right click."""
        O, C = (self.O, self.C)
        pad = [(O, O)] * 30
        cases = {'0.4 s: click': (12, ['left'], ['tick']), '1.0 s: a slow reaction to the tick, still a click': (30, ['left'], ['tick']), '1.4 s: right click': (42, ['right'], ['tick', 'double']), '2.2 s: a slow reaction to the double tick, still a right click': (66, ['right'], ['tick', 'double']), '2.6 s: resting, no click': (78, [], ['tick', 'double']), '0.1 s natural blink: nothing': (3, [], [])}
        for name, (frames, clicks, ticks) in cases.items():
            with self.subTest(name):
                got_clicks, got_ticks = self.ticked(pad + [(C, C)] * frames + pad)
                self.assertEqual(got_clicks, clicks)
                self.assertEqual([k for _, k in got_ticks], ticks)
        self.assertEqual(self.ticked(pad + [(C, C)] * 45 + pad, right_s=None), (['left'], [(self.ticked(pad + [(C, C)] * 45 + pad)[1][0][0], 'tick')]))
        self.assertEqual([k for _, k in self.ticked(pad + [(O, C)] * 30 + pad)[1]], ['tick'])
        self.assertEqual(self.ticked(pad + [(O, C)] * 30 + pad)[0], ['right'])

    def test_the_double_tick_keeps_its_second_after_a_calibrated_tick(self):
        """1.5.9's calibrated tick (up to 0.65 s) kept the double tick at 1.3 s:
        0.65 s to react, and the user opens 0.55-0.9 s after the tick - a left
        click came out as a right click."""
        O, C = (self.O, self.C)
        for min_s in (li.BLINK_MIN_S, 0.45):
            with self.subTest(min_s):

                def run(frames):
                    b = li.BlinkDetector(None, 0.55, 0.7, min_s, li.BLINK_MAX_S, 0.6)
                    b.baseline = O
                    t, clicks, ticks = (0.0, [], [])
                    for left, right in [(O, O)] * 30 + [(C, C)] * frames + [(O, O)] * 30:
                        t += 1 / FPS
                        click = b.update(left, right, t, None, True)[1]
                        if b.cue:
                            ticks.append(t)
                        if click:
                            clicks.append(click)
                    return (clicks, ticks)
                _, ticks = run(60)
                self.assertAlmostEqual(ticks[1] - ticks[0], 1.0, delta=1.5 / FPS)
                react = int(round((min_s + 0.85) * FPS))
                self.assertEqual(run(react)[0], ['left'])

    def ear(self, closedness: float) -> float:
        """The EAR at this closedness (1.0 = the close threshold; see BlinkDetector.closedness)."""
        return self.O * (1 - closedness * (1 - 0.55))

    def test_a_blink_that_starts_on_one_side_is_a_blink(self):
        """1.5.2's log: the user's right eye closes later and less - blinks began as
        "left winks" and stayed so (one tick, no double, no voice typing)."""
        O, C = (self.O, self.C)
        pad = [(O, O)] * 30
        lag = [(C, self.ear(0.5))] * 3
        for name, right, frames, want, ticks in (('... then closes only to 0.7', 0.7, 45, ['right'], ['tick', 'double']), ('... then closes fully', 1.5, 45, ['right'], ['tick', 'double']), ('... a short one', 0.7, 15, ['left'], ['tick'])):
            with self.subTest(name):
                clicks, got = self.ticked(pad + lag + [(C, self.ear(right))] * frames + pad)
                self.assertEqual(clicks, want)
                self.assertEqual([k for _, k in got], ticks)
        clicks, got = self.ticked(pad + [(C, self.ear(0.7))] * 15 + pad)
        self.assertEqual((clicks, [k for _, k in got]), (['left'], ['tick']))
        self.assertEqual(self.ticked(pad + [(C, self.ear(0.3))] * 15 + pad)[0], ['left'])
        self.assertEqual(self.ticked(pad + [(self.ear(0.3), C)] * 15 + pad)[0], ['right'])

    def test_a_blink_that_reopens_slowly_is_not_a_click(self):
        """33 people's videos (the GRID corpus): natural blinks that reopen slowly,
        the eyes looking down - half open for 0.2-1 s - passed the tick and
        clicked (4 times a minute; one person's half-shut eye made false right
        clicks). The tick comes only while the eyes are shut (TICK_WHEN_SHUT)."""
        O, C = (self.O, self.C)
        pad = [(O, O)] * 30
        half = (self.ear(0.8), self.ear(0.8))
        slow = pad + [(C, C)] * 5 + [half] * 15 + pad
        self.assertEqual(self.ticked(slow), ([], []))
        self.assertEqual(self.ticked(pad + [(O, C)] * 4 + [(O, self.ear(0.8))] * 15 + pad), ([], []))
        meant = pad + [(C, C)] * 12 + [half] * 6 + pad
        clicks, ticks = self.ticked(meant)
        self.assertEqual((clicks, [k for _, k in ticks]), (['left'], ['tick']))
        li.BlinkDetector.TICK_WHEN_SHUT = False
        try:
            self.assertEqual(self.ticked(slow)[0], ['left'])
        finally:
            li.BlinkDetector.TICK_WHEN_SHUT = True

    def test_every_missed_click_is_logged(self):
        """One hint per 20 s hid how many meant clicks went missing (2026-10-28)."""
        O, C = (self.O, self.C)
        half = (self.ear(0.8), self.ear(0.8))
        missed = [(C, C)] * 6 + [half] * 8 + [(O, O)] * 30
        with self.assertLogs('eyesync', 'INFO') as logs:
            self.assertEqual(self.ticked([(O, O)] * 30 + missed * 2), ([], []))
        self.assertEqual(sum(('opening before the tick' in m for m in logs.output)), 2)
        refused = [m for m in logs.output if 'opening before the tick' in m]
        self.assertTrue(all('deepest' in m and 'weaker eye' in m for m in refused))

    def test_tick_requires_both_eyes_to_stay_closed(self):
        """One eye reopening must not let the other eye's high score arm a click.

        Asymmetric deliberate blinks still work: both eyes participate, even when
        one closes only partly. The failing pattern is one eye open while the
        other's score is so high that their average alone looks shut.
        """
        O, C = (self.O, self.C)
        pad = [(O, O)] * 30
        one_open = [(self.ear(0.2), self.ear(2.0))] * 15
        self.assertEqual(self.ticked(pad + [(C, C)] * 4 + one_open + pad), ([], []))
        partial = [(C, self.ear(0.7))] * 15
        clicks, ticks = self.ticked(pad + partial + pad)
        self.assertEqual(clicks, ['left'])
        self.assertEqual([kind for _, kind in ticks], ['tick'])

    def drifting(self, drift_from: float, frames: int=36) -> list[str]:
        """A closure the eyelids show only partly (0.9 - MediaPipe's blink score
        says shut) while the head drifts 0.18 face widths sideways from
        ``drift_from`` s on, then opens: the clicks."""
        b = li.BlinkDetector(None, 0.55, 0.7, li.BLINK_MIN_S, li.BLINK_MAX_S, 0.6)
        b.baseline = self.O
        half = self.ear(0.9)
        seq = [(self.O, (0.03, 0.03))] * 30 + [(half, (0.9, 0.9))] * frames + [(self.O, (0.03, 0.03))] * 30
        t, out = (0.0, [])
        for k, (ear, sc) in enumerate(seq):
            t += 1 / FPS
            held = (k - 30) / FPS
            x = 0.18 * float(np.clip((held - drift_from) / 0.4, 0, 1)) if k >= 30 else 0.0
            click = b.update(ear, ear, t, sc, True, None, np.array([x, 0.0]))[1]
            if click:
                out.append(click)
        return out

    def test_a_ticked_click_survives_the_head_drifting(self):
        """1.5.10's log (2026-10-27/28): 1131 and 1173 ms closures, eyelids 0.86 and
        1.10 of 1.25, cancelled as "half closed" when the head had drifted 0.17-0.18
        face widths - long after their tick. The tick promised a click."""
        self.assertEqual(self.drifting(0.6), ['left'])
        li.BlinkDetector.KEEP_TICKED = False
        try:
            self.assertEqual(self.drifting(0.6), [])
        finally:
            li.BlinkDetector.KEEP_TICKED = True
        self.assertEqual(self.drifting(-0.2), [])

    def test_ticks_come_when_opening_would_click(self):
        O, C = (self.O, self.C)
        pad = [(O, O)] * 30
        _, ticks = self.ticked(pad + [(C, C)] * 40 + pad)
        (t1, _), (t2, _) = ticks
        self.assertAlmostEqual(t1 - 30 / FPS, li.BLINK_MIN_S, delta=0.07)
        self.assertAlmostEqual(t2 - t1, li.BLINK_RIGHT_S - li.BLINK_MIN_S, delta=0.04)
        for frames in range(4, 50):
            clicks, ticks = self.ticked(pad + [(C, C)] * frames + pad)
            if clicks:
                self.assertEqual(clicks, ['right' if ticks[-1][1] == 'double' else 'left'], frames)
        blink = [(C, C)] * 12 + [(O, O)] * 3
        clicks, ticks = self.ticked(pad + blink * 2 + pad)
        self.assertEqual((clicks, [k for _, k in ticks]), (['left'], ['tick']))
        rest = [(C, C)] * 75 + [(O, O)] * 9
        clicks, ticks = self.ticked(pad + rest + [(C, C)] * 10 + pad)
        self.assertEqual(clicks, [])
        self.assertEqual([k for t, k in ticks if t > 4.0], [])

    def test_cooldown(self):
        O, C = (self.O, self.C)
        blink = [(C, C)] * 12 + [(O, O)] * 3
        self.assertEqual(self.clicks([(O, O)] * 20 + blink * 2 + [(O, O)] * 20), 1)

    def run_posed(self, frames):
        """frames of (EAR, blink score, pitch) at 30 FPS -> (clicks, frames reported closed)."""
        b = li.BlinkDetector(None, 0.55, 0.7, 0.2, 1.5, 0.6)
        t, clicks, closed = (0.0, [], 0)
        for ear, sc, pitch in frames:
            t += 1 / FPS
            shut, click = b.update(ear, ear, t, (sc, sc), True, (0.0, pitch))[:2]
            closed += shut
            if click:
                clicks.append(click)
        return (clicks, closed)

    @staticmethod
    def move(frm, to, frames):
        """Head turning from pitch frm to pitch to: (EAR, score, pitch). Looking further
        down lowers the eyelids (EAR 0.28 -> 0.15) and raises the blink score (to 0.55)."""
        out = []
        for i in range(1, frames + 1):
            p = frm + (to - frm) * i / frames
            k = min(max(-p / 20.0, 0.0), 1.0)
            out.append((0.28 - 0.13 * k, 0.03 + 0.52 * k, p))
        return out

    def test_moving_the_head_down_and_back_is_not_a_blink(self):
        centre = [(0.28, 0.03, 0.0)] * 60
        at_bottom = self.move(-20, -20, 20)
        seq = centre + self.move(0, -20, 9) + at_bottom + self.move(-20, 0, 9) + centre
        self.assertEqual(self.run_posed(seq)[0], [])

    def test_blinks_work_while_looking_at_the_screen_bottom(self):
        centre = [(0.28, 0.03, 0.0)] * 60
        bottom = self.move(-20, -20, 1)[0]
        blink = [(0.05, 0.95, -20.0)] * 12
        seq = centre + self.move(0, -20, 9) + [bottom] * 60 + blink + [bottom] * 30
        clicks, closed = self.run_posed(seq)
        self.assertEqual(clicks, ['left'])
        self.assertLess(closed, 12 + 10)

    def test_blink_with_a_small_nod_still_clicks(self):
        centre = [(0.28, 0.03, 0.0)] * 60
        nod = [(0.05, 0.95, -3.0 * i / 6) for i in range(1, 7)] + [(0.05, 0.95, -3.0)] * 6
        clicks, _ = self.run_posed(centre + nod + centre)
        self.assertEqual(clicks, ['left'])

    def run_nose(self, frames, min_s=0.2, max_s=1.5):
        """frames of (EAR, score, nose x in face widths, head-angle reading) -> (clicks,
        frames reported closed, frames that rewound the cursor)."""
        b = li.BlinkDetector(None, 0.55, 0.7, min_s, max_s, 0.6)
        t, clicks, closed, rewinds = (0.0, [], 0, 0)
        for ear, sc, nose_x, pitch in frames:
            t += 1 / FPS
            shut, click, just = b.update(ear, ear, t, (sc, sc), True, (0.0, pitch), np.array([nose_x, 0.0]))
            closed += shut
            rewinds += just
            if click:
                clicks.append(click)
        return (clicks, closed, rewinds)

    def test_still_blink_clicks_although_the_head_angle_reading_jumps(self):
        """MediaPipe's head angles jump ~8 degrees when both eyes close; the nose stays."""
        still = [(0.28, 0.03, 0.0, 0.0)] * 60
        blink = [(0.05, 0.95, 0.002 * (i % 2), -8.5) for i in range(14)]
        self.assertEqual(self.run_nose(still + blink + still)[0], ['left'])

    def test_natural_blink_during_a_head_turn_does_not_stop_the_cursor(self):
        turn = [(0.28, 0.03, i / 30, 0.0) for i in range(30)]
        blink = [(0.05, 0.95, (30 + i) / 30, 0.0) for i in range(5)]
        after = [(0.28, 0.03, (35 + i) / 30, 0.0) for i in range(30)]
        clicks, closed, rewinds = self.run_nose(turn + blink + after)
        self.assertEqual((clicks, closed, rewinds), ([], 0, 0))

    def test_deliberate_blink_right_after_slow_aiming_clicks(self):
        aim = [(0.28, 0.03, 0.25 * i / 30, 0.0) for i in range(45)]
        x = 0.25 * 45 / 30
        blink = [(0.05, 0.95, x, -8.0)] * 12
        still = [(0.28, 0.03, x, 0.0)] * 30
        self.assertEqual(self.run_nose(aim + blink + still)[0], ['left'])

    def test_turning_the_head_with_eyes_only_half_closed_is_not_a_click(self):
        """Eyelids at ~55% of open + a high blink score (looking down): a head position."""
        still = [(0.28, 0.03, 0.0, 0.0)] * 60
        half = [(0.15, 0.6, 0.4 * i / 12, 0.0) for i in range(1, 13)]
        self.assertEqual(self.run_nose(still + half + [(0.28, 0.03, 0.4, 0.0)] * 30)[0], [])
        moving_half = [(0.15, 0.6, 0.03 * i, 0.0) for i in range(1, 25)]
        self.assertEqual(self.run_nose(still + moving_half + [(0.28, 0.03, 0.72, 0.0)] * 30)[0], [])

    def test_deliberate_blink_while_moving_the_head_clicks(self):
        move = [(0.28, 0.03, i / 30, 0.0) for i in range(30)]
        blink = [(0.07, 0.9, (30 + i) / 30, 0.0) for i in range(16)]
        after = [(0.28, 0.03, (46 + i) / 30, 0.0) for i in range(30)]
        clicks, closed, rewinds = self.run_nose(move + blink + after)
        self.assertEqual(clicks, ['left'])
        self.assertEqual(rewinds, 1)
        self.assertLess(closed, 16)

    def test_natural_blinks_while_reading_do_not_click(self):
        """Both recordings of a real user: natural blinks while reading lasted up to
        274-286 ms and clicked 3-6 times a minute with the tick at 0.2 s (1.5.3)."""

        def blink(frames, speed):
            still = [(0.28, 0.03, 0.0, 0.0)] * 60
            moving = [(0.28, 0.03, speed * i / 30, 0.0) for i in range(30)]
            x = speed
            shut = [(0.07, 0.9, x + speed * i / 30, 0.0) for i in range(frames)]
            x += speed * frames / 30
            return self.run_nose(still + moving + shut + [(0.28, 0.03, x + speed * i / 30, 0.0) for i in range(40)], li.BLINK_MIN_S, li.BLINK_MAX_S)[0]
        for speed in (0.0, 0.7):
            for frames in (4, 6, 8, 9):
                with self.subTest(speed=speed, ms=round(frames * 1000 / 30)):
                    self.assertEqual(blink(frames, speed), [])
        self.assertEqual(blink(11, 0.0), ['left'])
        self.assertEqual(blink(11, 0.7), [])
        self.assertEqual(blink(15, 0.7), ['left'])

    def test_ticks_wait_for_a_moving_head(self):
        """A closure that begins with the head moving ticks from BLINK_MOVING_MIN_S, and
        its double tick keeps the second after it."""
        b = li.BlinkDetector(None, 0.55, 0.7, li.BLINK_MIN_S, li.BLINK_MAX_S, 0.6)
        t, ticks = (0.0, [])
        for i in range(160):
            t += 1 / FPS
            ear = 0.07 if 30 <= i < 90 else 0.28
            b.update(ear, ear, t, (0.03, 0.03), True, None, np.array([0.7 * min(i, 60) / 30, 0.0]))
            if b.cue:
                ticks.append((t - 31 / FPS, b.cue))
        (t1, k1), (t2, k2) = ticks
        self.assertEqual((k1, k2), ('tick', 'double'))
        self.assertAlmostEqual(t1, li.BLINK_MOVING_MIN_S, delta=0.05)
        self.assertAlmostEqual(t2 - t1, li.BLINK_RIGHT_S - li.BLINK_MIN_S, delta=0.05)

    def test_wink_while_tilting_the_head_clicks(self):
        """Tilting moves the nose sideways too; a right-eye wink is still a right click."""
        b = li.BlinkDetector(None, 0.55, 0.7, 0.2, 1.5, 0.6)
        t, clicks = (0.0, [])
        for i in range(90):
            t += 1 / FPS
            right = 0.06 if 40 <= i < 56 else 0.28
            sc = (0.03, 0.9 if 40 <= i < 56 else 0.03)
            click = b.update(0.28, right, t, sc, True, (0.0, 0.0), np.array([0.025 * i, 0.01 * i]))[1]
            if click:
                clicks.append(click)
        self.assertEqual(clicks, ['right'])

    def test_click_lands_where_the_blink_began(self):
        """The detector says when the eyes began to close, for putting the click there."""
        b = li.BlinkDetector(None, 0.55, 0.7, 0.2, 1.5, 0.6)
        t, held_at = (0.0, None)
        for i in range(70):
            t += 1 / FPS
            ear = 0.07 if 30 <= i < 46 else 0.28
            shut, click, just = b.update(ear, ear, t, (0.03, 0.03), True, None, np.array([i / 30, 0]))
            if just:
                held_at = t
        self.assertTrue(31 / FPS <= b.started_at <= 33 / FPS, b.started_at)
        self.assertLess(b.started_at - 31 / FPS, li.CLICK_REWIND_S)
        self.assertGreaterEqual(held_at - b.started_at, li.BLINK_MOVING_MIN_S - 1e-06)

    def test_blink_after_resting_eyes_is_not_a_click(self):
        O, C = (self.O, self.C)
        rest = [(C, C)] * 75 + [(O, O)] * 9
        blink = [(C, C)] * 10 + [(O, O)] * 30
        self.assertEqual(self.clicks([(O, O)] * 20 + rest + blink), 0)
        later = [(O, O)] * 45 + blink
        self.assertEqual(self.clicks([(O, O)] * 20 + rest + later), 1)
        long_blink = [(C, C)] * 20 + [(O, O)] * 30
        self.assertEqual(self.clicks([(O, O)] * 20 + rest + long_blink), 1)

class FaceGestureTests(unittest.TestCase):
    REST = {'mouthSmileLeft': 0.001, 'mouthSmileRight': 0.001, 'mouthPucker': 0.17, 'mouthLeft': 0.003, 'mouthRight': 0.003, 'mouthShift': 0.0}
    TO_LEFT = {'mouthRight': 0.6, 'mouthShift': 0.06}
    TO_RIGHT = {'mouthLeft': 0.6, 'mouthShift': -0.06}

    def run_seq(self, segments):
        """segments: (seconds, {score overrides}) -> list of fired gestures."""
        g, t, fired = (li.FaceGestures(), 0.0, [])
        for secs, over in segments:
            for _ in range(int(round(secs * FPS))):
                t += 1 / FPS
                f = g.update({**self.REST, **over}, t)
                if f:
                    fired.append(f)
        return fired

    def test_each_gesture_fires_once(self):
        smile = {'mouthSmileLeft': 0.8, 'mouthSmileRight': 0.75}
        self.assertEqual(self.run_seq([(3, {}), (1.7, smile), (1, {})]), ['smile'])
        self.assertEqual(self.run_seq([(3, {}), (1.7, {'mouthPucker': 0.8}), (1, {})]), [])
        self.assertEqual(self.run_seq([(3, {}), (1.7, self.TO_LEFT), (1, {})]), ['mouth_left'])
        self.assertEqual(self.run_seq([(3, {}), (3.3, self.TO_RIGHT), (1, {})]), ['mouth_right'])

    def test_mouth_side_gestures(self):
        self.assertEqual(self.run_seq([(3, {}), (1.7, self.TO_RIGHT), (1, {})]), [])
        self.assertEqual(self.run_seq([(3, {}), (1.7, {'mouthLeft': 0.6, 'mouthShift': 0.06})]), ['mouth_left'])
        self.assertEqual(self.run_seq([(3, {}), (3.3, {'mouthRight': 0.6, 'mouthShift': 0.005})]), [])
        self.assertEqual(self.run_seq([(3, {'mouthShift': None}), (3.3, {'mouthRight': 0.6})]), [])
        crooked = {'mouthShift': 0.04}
        self.assertEqual(self.run_seq([(3, crooked), (1.7, {**self.TO_LEFT, 'mouthShift': 0.045})]), [])
        self.assertEqual(self.run_seq([(3, crooked), (1.7, {**self.TO_LEFT, 'mouthShift': 0.1})]), ['mouth_left'])
        self.assertEqual(self.run_seq([(3, {}), (1.0, self.TO_LEFT), (0.6, self.TO_RIGHT), (1, {})]), [])

    def test_this_users_pulls(self):
        """Measured 2026-10-20 (MOUTH_PULL_FW): MediaPipe scores this user's pull to
        the left as a lopsided smile, and wavers on their steady pull to the right."""
        lopsided = {'mouthSmileLeft': 0.78, 'mouthSmileRight': 0.45, 'mouthShift': 0.03}
        self.assertEqual(self.run_seq([(3, {}), (1.7, lopsided), (1, {})]), ['mouth_left'])
        wavering = [(0.2, {'mouthRight': v, 'mouthSmileRight': 0.1, 'mouthShift': -0.04}) for v in (0.6, 0.12, 0.5, 0.18, 0.7, 0.1, 0.45, 0.3) * 2]
        self.assertEqual(self.run_seq([(3, {})] + wavering + [(1, {})]), ['mouth_right'])
        even = {'mouthSmileLeft': 0.9, 'mouthSmileRight': 0.88, 'mouthShift': 0.015}
        self.assertEqual(self.run_seq([(3, {}), (1.7, even), (1, {})]), ['smile'])
        self.assertEqual(self.run_seq([(3, {}), (3.3, {'mouthShift': -0.05})]), [])
        self.assertEqual(self.run_seq([(3, {}), (3.3, {'mouthRight': 0.6, 'mouthShift': -0.05, 'jawOpen': 0.3})]), [])

    def test_a_lopsided_smile_never_starts_calibrate(self):
        """1.5.11's log (2026-10-28): "Mouth to your right: calibrate" beeped 3 times,
        2-4 s before a smile pause or a tilt, never meant. A smile rising more on
        one side, the mouth drifting with it, is no pull without a pull score."""
        lopsided = {'mouthSmileLeft': 0.15, 'mouthSmileRight': 0.45, 'mouthShift': -0.03}
        g, t, seen = (li.FaceGestures(), 0.0, set())
        for secs, over in ((3, {}), (3.3, lopsided), (1, {})):
            for _ in range(int(round(secs * FPS))):
                t += 1 / FPS
                self.assertIsNone(g.update({**self.REST, **over}, t))
                seen.add(g.active)
        self.assertNotIn('mouth_right', seen)
        even = {'mouthSmileLeft': 0.8, 'mouthSmileRight': 0.75}
        self.assertEqual(self.run_seq([(3, {}), (1.0, lopsided), (1.7, even), (1, {})]), ['smile'])
        g = li.FaceGestures()
        for k in range(int(4 * FPS)):
            g.update({**self.REST, **(self.TO_RIGHT if k >= 3 * FPS else {})}, (k + 1) / FPS)
        self.assertEqual(g.active, 'mouth_right')
        self.assertIn('to your right, mouthLeft 0.60', g.detail)

    def test_the_face_eases_out_of_a_gesture_before_blinks_count(self):
        """1.5.2's log: a "left wink" click the moment a smile resumed EyeSync."""
        g, t = (li.FaceGestures(), 0.0)
        smile = {**self.REST, 'mouthSmileLeft': 0.8, 'mouthSmileRight': 0.75}
        easing = {**self.REST, 'mouthSmileLeft': 0.45, 'mouthSmileRight': 0.4}
        relaxed = []
        for secs, bs in ((3, self.REST), (1.7, smile), (1.0, easing), (0.5, self.REST)):
            for _ in range(int(secs * FPS)):
                t += 1 / FPS
                g.update(bs, t)
                relaxed.append(g.relaxed)
        n = int(3 * FPS)
        self.assertTrue(all(relaxed[:n + 40]))
        self.assertFalse(any(relaxed[n + 46:n + 81]))
        self.assertTrue(relaxed[-1])
        g.update(smile, t + 10)
        self.assertTrue(g.relaxed)

    def test_a_head_turn_is_not_a_pull(self):
        g, t, fired = (li.FaceGestures(), 0.0, [])
        for secs, yaw in ((3, 0.0), (3.3, 20.0), (3.3, -20.0), (1, 0.0)):
            for _ in range(int(secs * FPS)):
                t += 1 / FPS
                turned = {**self.REST, 'mouthShift': -0.196 * np.sin(np.radians(yaw)), 'mouthRight': 0.2 if yaw < 0 else 0.0, 'mouthLeft': 0.2 if yaw > 0 else 0.0}
                if (f := g.update(turned, t, (yaw, 0.0))):
                    fired.append(f)
        self.assertEqual(fired, [])

    def test_mouth_shift_measurement(self):
        lm = np.zeros((478, 2), np.float32)
        lm[33], lm[263] = ((0.4, 0.4), (0.6, 0.4))
        lm[fe.SUBNASALE] = (0.5, 0.55)
        for off, want in ((0.0, 0.0), (0.02, 0.1), (-0.03, -0.15)):
            lm[61], lm[291] = ((0.45 + off, 0.62), (0.55 + off, 0.62))
            self.assertAlmostEqual(fe.mouth_shift(lm, 640, 480), want, places=4)

    def test_holding_on_fires_only_once(self):
        smile = {'mouthSmileLeft': 0.8, 'mouthSmileRight': 0.8}
        self.assertEqual(self.run_seq([(3, {}), (5, smile)]), ['smile'])
        self.assertEqual(self.run_seq([(3, {}), (1.7, smile), (0.5, {}), (1.7, smile)]), ['smile', 'smile'])

    def test_short_expressions_do_nothing(self):
        smile = {'mouthSmileLeft': 0.8, 'mouthSmileRight': 0.8}
        self.assertEqual(self.run_seq([(3, {}), (1.0, smile), (1, {}), (0.8, {'mouthPucker': 0.8})]), [])

    def test_talking_does_nothing(self):
        rng = np.random.default_rng(2)
        segs = [(3, {})] + [(0.2, {'mouthPucker': float(rng.uniform(0.1, 0.66)), 'mouthSmileLeft': float(rng.uniform(0, 0.3)), 'mouthLeft': float(rng.uniform(0, 0.2))}) for _ in range(60)]
        self.assertEqual(self.run_seq(segs), [])

    def test_brief_sag_does_not_cancel(self):
        smile = {'mouthSmileLeft': 0.8, 'mouthSmileRight': 0.8}
        self.assertEqual(self.run_seq([(3, {}), (0.8, smile), (0.2, {}), (0.9, smile)]), ['smile'])

    def test_head_tilts_are_not_hold_gestures_any_more(self):
        """Tilts scroll now (TiltScroll); they no longer recenter or calibrate."""
        self.assertNotIn('tilt_left', li.GESTURES)
        self.assertEqual(self.run_seq([(3, {}), (1.7, {'headRoll': 20.0}), (1, {})]), [])

    def test_head_roll_measurement(self):
        lm = np.zeros((478, 2), np.float32)
        for angle in (0.0, 15.0, -20.0):
            a = np.radians(angle)
            lm[fe.EYE_OUTER[0]] = (0.4, 0.5)
            lm[fe.EYE_OUTER[1]] = (0.4 + 0.2 * np.cos(a), 0.5 + 0.2 * np.sin(a))
            self.assertAlmostEqual(fe.head_roll(lm, 640, 640), angle, places=3)

    def test_light_smile_does_nothing(self):
        light = {'mouthSmileLeft': 0.5, 'mouthSmileRight': 0.45, 'mouthRight': 0.3}
        self.assertEqual(self.run_seq([(3, {}), (4, light), (1, {})]), [])

    def test_big_smile_works_with_the_head_turned(self):
        """At a screen edge the far half of the mouth is scored low."""
        g, t, fired = (li.FaceGestures(), 0.0, [])
        smile = {**self.REST, 'mouthSmileLeft': 0.85, 'mouthSmileRight': 0.3}
        for secs, bs in ((3, self.REST), (1.7, smile), (1, self.REST)):
            for _ in range(int(secs * FPS)):
                t += 1 / FPS
                if (f := g.update(bs, t, (24.0, -5.0))):
                    fired.append(f)
        self.assertEqual(fired, ['smile'])
        self.assertEqual(self.run_seq([(3, {}), (1.7, {'mouthSmileLeft': 0.85, 'mouthSmileRight': 0.3})]), [])

    def test_resting_levels_are_learned_per_head_direction(self):
        """Turned to a side, this face's resting side-mouth score is 0.25."""
        g, t, fired = (li.FaceGestures(), 0.0, [])
        side_rest = {**self.REST, 'mouthRight': 0.25}
        pull = {**self.REST, 'mouthRight': 0.5, 'mouthShift': 0.06}
        for secs, bs, pose in ((3, self.REST, (0.0, 0.0)), (3, side_rest, (24.0, 0.0)), (1.7, pull, (24.0, 0.0)), (1.0, self.REST, (0.0, 0.0)), (1.7, pull, (0.0, 0.0))):
            for _ in range(int(secs * FPS)):
                t += 1 / FPS
                if (f := g.update(bs, t, pose)):
                    fired.append((f, pose[0]))
        self.assertEqual(fired, [('mouth_left', 0.0)])

class TiltScrollTests(unittest.TestCase):

    def run_tilts(self, segments, s=None):
        """segments: (seconds, head tilt in degrees, + = left) -> (wheel turns per frame, scroller)."""
        s, t, out = (s or li.TiltScroll(), 0.0, [])
        for secs, roll in segments:
            for _ in range(int(round(secs * FPS))):
                t += 1 / FPS
                out.append(s.update(roll, t))
        return (out, s)

    def test_tilt_left_scrolls_up_right_scrolls_down(self):
        up, _ = self.run_tilts([(2, 0.0), (1.0, 30.0), (1, 0.0)])
        down, _ = self.run_tilts([(2, 0.0), (1.0, -30.0), (1, 0.0)])
        self.assertTrue(all((u >= 0 for u in up)) and sum(up) > 0)
        self.assertTrue(all((d <= 0 for d in down)) and sum(down) < 0)
        self.assertEqual(sum(up), -sum(down))
        self.assertTrue(all((u % li.TiltScroll.WHEEL == 0 for u in up)))
        first = next((i for i, u in enumerate(up) if u))
        delay = li.TiltScroll.START_S + li.TiltScroll.MEDIAN_FRAMES // 2 / FPS
        self.assertAlmostEqual((first + 1) / FPS - 2.0, delay, delta=0.07)

    def test_further_is_faster_and_straightening_stops(self):
        gentle, _ = self.run_tilts([(2, 0.0), (3, 24.0)])
        far, _ = self.run_tilts([(2, 0.0), (3, 40.0)])
        self.assertGreater(sum(far), 3 * sum(gentle))
        notches = sum(far) / li.TiltScroll.WHEEL
        delay = li.TiltScroll.START_S + li.TiltScroll.MEDIAN_FRAMES // 2 / FPS
        self.assertAlmostEqual(notches, 1 + li.TiltScroll.FAST * (3 - delay), delta=2)
        out, s = self.run_tilts([(2, 0.0), (1, 30.0), (0.5, 8.0), (1, 8.0)])
        self.assertFalse(s.active)
        self.assertEqual(sum(out[int(3.1 * FPS):]), 0)

    def test_brief_or_small_tilts_do_nothing(self):
        out, s = self.run_tilts([(2, 0.0), (0.25, 30.0), (1, 0.0), (3, 12.0), (3, -12.0)])
        self.assertEqual(sum((abs(u) for u in out)), 0)
        self.assertFalse(s.tilted)

    def test_a_head_turned_to_a_screen_side_is_not_a_tilt(self):
        """A real user's head leans as it turns to a screen side: up to 15.5 degrees
        in their recordings, 19 while talking. Not a tilt: no scrolling, and
        the cursor and blinks carry on (``tilted`` stays false)."""
        for lean in (15.5, -15.5, 19.0, -19.0, 21.0):
            with self.subTest(lean=lean):
                out, s = self.run_tilts([(2, 0.0), (0.3, lean / 2), (3, lean), (1, 0.0)])
                self.assertEqual(sum((abs(u) for u in out)), 0)
                self.assertFalse(s.tilted)

    def test_a_usual_head_tilt_is_learned(self):
        """Someone who sits with the head tilted 8 degrees tilts 22 more to scroll."""
        out, s = self.run_tilts([(6, 8.0), (2, 27.0)])
        self.assertEqual(sum(out), 0)
        self.assertAlmostEqual(s.usual, 8.0, delta=1.0)
        out, _ = self.run_tilts([(6, 8.0), (2, 32.0)])
        self.assertGreater(sum(out), 0)

    def test_losing_the_face_stops(self):
        s = li.TiltScroll()
        self.run_tilts([(2, 0.0), (1, 30.0)], s)
        self.assertTrue(s.active)
        self.assertEqual(s.update(None, 10.0), 0)
        self.assertFalse(s.active)

import threading
class CloseGestureTests(unittest.TestCase):

    def worker(self):
        w = li.CloseGesture.__new__(li.CloseGesture)
        w.args = types.SimpleNamespace(no_close_gesture=False, close_gesture_mar=0.6, close_gesture_hold=3.0)
        w.gesture_since = w.gesture_low_since = None
        w.gesture_announced = False
        w.state = types.SimpleNamespace(stop=threading.Event())
        return w

    def quits(self, seq, jaw=None) -> bool:
        """seq of (mouth aspect ratio, eyes open) at 30 FPS -> did EyeSync shut down?"""
        w, t = (self.worker(), 0.0)
        for mar, eyes_open in seq:
            t += 1 / FPS
            w._close_gesture(mar, eyes_open, t, jaw)
        return w.state.stop.is_set()

    def test_brief_wide_opening_does_not_even_beep(self):
        w, t, beeps = (self.worker(), 0.0, [])
        old, li.desktop.play_warning_sound = (li.desktop.play_warning_sound, lambda: beeps.append(1))
        try:
            for mar in [0.7] * 12 + [0.05] * 30:
                t += 1 / FPS
                w._close_gesture(mar, True, t)
        finally:
            li.desktop.play_warning_sound = old
        self.assertEqual(beeps, [])
        self.assertFalse(w.state.stop.is_set())

    def test_jaw_not_dropped_does_not_start(self):
        """The ratio reads wide open but MediaPipe sees the jaw closed (e.g. head turned)."""
        self.assertFalse(self.quits([(0.7, True)] * 120, jaw=0.15))
        self.assertTrue(self.quits([(0.7, True)] * 96, jaw=0.6))

    def test_head_pose_and_turned_mouth(self):

        def rot(yaw, pitch):
            a, b = (np.radians(yaw), np.radians(pitch))
            ry = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
            rx = np.array([[1, 0, 0], [0, np.cos(b), -np.sin(b)], [0, np.sin(b), np.cos(b)]])
            m = np.eye(4)
            m[:3, :3] = ry @ rx
            return m
        for yaw, pitch in ((0, 0), (20, 0), (-15, 10), (25, -12)):
            got = fe.head_pose(rot(yaw, pitch))
            self.assertAlmostEqual(abs(got[0]), abs(yaw), delta=1.5)
            self.assertAlmostEqual(abs(got[1]), abs(pitch), delta=1.5)
        lm = np.zeros((478, 2), np.float32)
        lm[fe.MOUTH_LEFT], lm[fe.MOUTH_RIGHT] = ((0.45, 0.6), (0.45 + 0.1 * np.cos(np.radians(30)), 0.6))
        lm[fe.UPPER_LIP], lm[fe.LOWER_LIP] = ((0.5, 0.6), (0.5, 0.6 + 0.045))
        self.assertGreater(fe.mouth_aspect_ratio(lm, 640, 640), 0.51)
        self.assertAlmostEqual(fe.mouth_aspect_ratio(lm, 640, 640, (30.0, 0.0)), 0.45, places=2)

    def test_talking_does_not_quit(self):
        self.assertFalse(self.quits([(0.45 * abs(np.sin(i / 3)), True) for i in range(600)]))

    def test_yawn_does_not_start(self):
        self.assertFalse(self.quits([(0.8, False)] * 150))

    def test_short_open_cancels(self):
        self.assertFalse(self.quits([(0.7, True)] * 60 + [(0.05, True)] * 90))

    def test_held_open_quits(self):
        self.assertTrue(self.quits([(0.7, True)] * 96))

    def test_wavering_and_squinting_still_quits(self):
        seq = [(0.7, True)] * 10
        for i in range(90):
            mar = 0.5 if i % 15 < 9 else 0.35 if i % 15 < 12 else 0.65
            seq.append((mar, i % 4 != 0))
        self.assertTrue(self.quits(seq))

    def test_half_second_sag_still_quits(self):
        seq = [(0.7, True)] * 40 + [(0.3, True)] * 15 + [(0.55, False)] * 50
        self.assertTrue(self.quits(seq))

    def test_closing_mouth_cancels(self):
        self.assertFalse(self.quits([(0.7, True)] * 45 + [(0.1, True)] * 15 + [(0.7, False)] * 60))
