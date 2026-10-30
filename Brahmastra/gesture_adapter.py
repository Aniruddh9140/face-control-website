"""Combine original EyeSync AI detectors without opening hardware or importing its app."""
from types import SimpleNamespace
import threading
import numpy as np
import desktop
import eyesync_gestures as eg


class EyeSyncGestures:
    def __init__(self):
        self.blink = eg.BlinkDetector(None, .55, .70, eg.BLINK_MIN_S, eg.BLINK_MAX_S, .6)
        self.expressions = eg.FaceGestures()
        self.scroll = eg.TiltScroll()
        self.close = eg.CloseGesture()
        self.close.args = SimpleNamespace(no_close_gesture=False, close_gesture_mar=.6, close_gesture_hold=3.)
        self.close.state = SimpleNamespace(stop=threading.Event())
        self.close.gesture_since = self.close.gesture_low_since = None
        self.close.gesture_announced = False
        self.voice_closure = None
        self.open_frames = 0
        self.last_update = None
        self.hold_until = 0.
        self.quit_emitted = False

    @property
    def ready(self):
        return self.blink.baseline is not None and self.open_frames >= 24

    def lost_face(self):
        self.blink.reset()
        self.expressions.reset()
        self.scroll.update(None, 0.)
        self.close.gesture_since = None
        self.voice_closure = None
        self.open_frames = 0
        self.last_update = None
        self.hold_until = 0.

    def update(self, left, right, now, frame=None, *, gestures=True, active=False, speaking=False):
        f = frame or {}
        left, right = f.get('ears', (left, right))
        pose, bs = f.get('pose'), f.get('scores')
        # A missing interval is not time spent deliberately holding a gesture.
        if self.last_update is not None and (now-self.last_update > .25 or now <= self.last_update):
            self.lost_face()
        self.last_update = now
        try:
            numbers = [left, right, f.get('mar', 0.)]
            if f.get('roll') is not None:
                numbers.append(f['roll'])
            if pose is not None:
                numbers.extend(pose)
            if bs is not None:
                numbers.extend(bs.values())
            if f.get('mouth_shift') is not None:
                numbers.append(f['mouth_shift'])
            if f.get('head') is not None:
                numbers.extend(f['head'])
            valid = np.isfinite(np.asarray(numbers, float)).all()
        except (TypeError, ValueError):
            valid = False
        if not valid:
            self.lost_face()
            return dict(click='', frozen=True, closed=False, voice=False, quit=False,
                        action=None, wheel=0, progress=0., hint='Tracking interrupted — hold your face steady.',
                        tilted=False, scroll_direction=0)
        eyes_open = self.blink.eyes_open(left, right, pose)
        if eyes_open:
            self.open_frames += 1
        result = dict(click='', frozen=False, closed=False, voice=False, quit=False,
                      action=None, wheel=0, progress=0., hint='', tilted=False, scroll_direction=0)
        if gestures and not speaking and self.close._close_gesture(f.get('mar', 0.), eyes_open, now,
                                        None if bs is None else bs.get('jawOpen')):
            self.blink.reset()
            self.expressions.reset()
            quitting = self.close.state.stop.is_set() and not self.quit_emitted
            self.quit_emitted = self.quit_emitted or quitting
            self.scroll.update(None, now)
            result.update(frozen=True, quit=quitting,
                          progress=min((now-self.close.gesture_since)/3.,1.),
                          hint=f"Closing EyeSync in {max(0.,3.-(now-self.close.gesture_since)):.1f}s — close your mouth to cancel.")
            return result
        else:
            self.close.gesture_since = None
        wheel = self.scroll.update(f.get('roll') if gestures and active else None, now)
        if self.scroll.tilted:
            self.blink.reset()
            self.expressions.reset()
            self.hold_until = now + .2
            direction = self.scroll.direction
            hint = ('Scrolling up' if direction > 0 else 'Scrolling down') if direction else 'Tilt detected — hold to scroll'
            result.update(frozen=True, wheel=wheel, tilted=True, scroll_direction=direction,
                          hint=hint + ' — straighten your head to stop.')
            return result
        if now < self.hold_until:
            self.blink.reset()
            self.expressions.reset()
            result.update(frozen=True, hint='Scrolling stopped — settling pointer.')
            return result
        expressions = None if not gestures or bs is None else {**bs, 'mouthShift': f.get('mouth_shift')}
        action = self.expressions.update(expressions, now, pose,
                            eg.VOICE_IGNORED_GESTURES if speaking else frozenset())
        result['action'] = action
        if self.expressions.active or action:
            self.blink.reset()
            result.update(frozen=True, progress=self.expressions.progress(now),
                          hint=eg.GESTURE_LABELS.get(self.expressions.active or action, ''))
            return result
        if self.expressions.relaxed:
            scores = None if bs is None else (bs.get('eyeBlinkLeft',0.), bs.get('eyeBlinkRight',0.))
            closed, click, _ = self.blink.update(left,right,now,scores,f.get('winks_ok',True),pose,f.get('head'))
        else:
            self.blink.reset()
            closed, click = False, ''
        if self.blink.cue and active:
            desktop.play_tick_sound(double=self.blink.cue=='double')
        since = self.blink.closed_since
        if not closed or since is None:
            self.voice_closure = None
        elif gestures and self.voice_closure != since and now-since >= eg.VOICE_EYES_CLOSED_S and self.blink.deep >= eg.BLINK_DEEP:
            self.voice_closure = since
            result['voice'] = True
        result.update(closed=closed, frozen=closed, click=click,
                      progress=min(self.blink.closed_for(now)/eg.VOICE_EYES_CLOSED_S,1.) if closed else 0.)
        return result
