"""Original EyeSync AI detectors extracted unchanged from E:/EyeSync/live_inference.py."""
from __future__ import annotations
import logging, math, time
from collections import deque
import numpy as np
import desktop
log=logging.getLogger("eyesync.gestures")
Pose = tuple[float, float]

def pose_change(a: Pose | None, b: Pose | None) -> float:
    """How far the head turned between two poses (degrees, largest axis)."""
    if a is None or b is None:
        return 0.0
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))

class PoseBins:
    """
    A resting level learned separately for each head direction. Looking at
    the bottom of the screen lowers the eyelids and raises MediaPipe's blink
    score; turning to a side changes the expression scores. Judged against
    one overall level, a head position alone looked like a blink or a
    gesture (or hid a real one), so each direction - a cell of YAW_DEG x
    PITCH_DEG - keeps its own level, once it has seen MIN_FRAMES.
    """
    YAW_DEG, PITCH_DEG = (8.0, 6.0)
    MIN_FRAMES = 15
    RATE = 0.03

    def __init__(self):
        self.cells: dict[tuple[int, int], list[float]] = {}

    @classmethod
    def key(cls, pose: Pose) -> tuple[int, int]:
        return (round(pose[0] / cls.YAW_DEG), round(pose[1] / cls.PITCH_DEG))

    def get(self, pose: Pose | None) -> float | None:
        """The learned level for this direction, or None (not learned yet)."""
        if pose is None:
            return None
        cell = self.cells.get(self.key(pose))
        return cell[0] if cell is not None and cell[1] >= self.MIN_FRAMES else None

    def learn(self, pose: Pose | None, value: float) -> None:
        if pose is None:
            return
        cell = self.cells.setdefault(self.key(pose), [value, 0])
        cell[1] += 1
        cell[0] += max(1.0 / cell[1], self.RATE) * (value - cell[0])
GESTURES = {'smile': ('Big smile: pause / resume', 0.6, 0.45), 'side': ('Mouth pulled to one side', 0.45, 0.35)}
SIDE_GESTURES = {1: 'mouth_left', -1: 'mouth_right'}
GESTURE_LABELS = {'smile': GESTURES['smile'][0], 'mouth_left': 'Mouth to your left: recenter', 'mouth_right': 'Mouth to your right: calibrate'}
GESTURE_HOLD_S = {'mouth_right': 3.0}
MOUTH_SHIFT_MIN_FW = 0.012
MOUTH_PULL_FW = 0.018
MOUTH_SHIFT_PER_YAW = 0.2
MOUTH_PULL_LOPSIDED = 0.12
MOUTH_PULL_SCORE = 0.15
MOUTH_PULL_MAX_YAW = 25.0
MOUTH_OPEN_JAW = 0.15
VOICE_EYES_CLOSED_S = 2.5
VOICE_IGNORED_GESTURES = frozenset({'smile', 'mouth_left', 'mouth_right'})

class FaceGestures:
    """
    Detects a held facial expression and reports it once. Each expression's
    threshold adapts to the user's resting level (a slow average while it is
    well below the threshold, per head direction - see PoseBins), a short
    sag (GRACE_S) doesn't cancel the hold, and after a gesture fires the face
    must relax before another can start - so holding a smile toggles pause
    once, not every 1.5 s.

    With the head turned, the camera sees the far half of the mouth badly
    and scores its smile low, so the smile counts the visible side more
    (fully from SMILE_TURN_DEG on): a big smile works at the screen edges too.

    Thresholds and resting levels are per channel (GESTURES); ``active`` and
    what update() returns are gesture names - the "side" channel becomes
    mouth_left or mouth_right by where the mouth went (``mouthShift`` in the
    scores, fe.mouth_shift, against its own resting place). With no shift
    given, or the mouth not clearly to one side, the side gestures don't fire.
    """
    HOLD_S = 1.5
    GRACE_S = 0.3
    SMILE_TURN_DEG = 25.0
    RELAX_MAX_S = 3.0

    def __init__(self):
        self.base: dict[str, float] = {}
        self.bins: dict[str, PoseBins] = {n: PoseBins() for n in GESTURES}
        self.minimum = {n: g[1] for n, g in GESTURES.items()}
        self.active: str | None = None
        self.since = 0.0
        self.low_since: float | None = None
        self.released = True
        self.relaxed = True
        self.fired_at = -1000000000.0
        self.rest_shift: float | None = None
        self.side_seen = (0.0, 0.0, 0.0, 0.0, 0.0)
        self.detail = ''

    def hold_s(self, name: str | None=None) -> float:
        return GESTURE_HOLD_S.get(name or self.active, self.HOLD_S)

    def _side(self, bs: dict[str, float], side_score: float, side_thr: float, yaw: float) -> int:
        """
        Which way the mouth is pulled: +1 the user's left, -1 right, 0 neither
        or unknown. Either the landmarks moved MOUTH_PULL_FW with a score on
        that side agreeing (see MOUTH_PULL_FW), or the side score passed its
        threshold with the landmarks showing the way. The mouth's resting
        place (fe.mouth_shift, the head turn taken out) is learned only while
        no score shows a pull - never in the middle of a long hold.
        """
        shift = bs.get('mouthShift')
        if shift is None:
            return 0
        here = shift + MOUTH_SHIFT_PER_YAW * float(np.sin(np.radians(yaw)))
        if self.rest_shift is None:
            self.rest_shift = here
        sl, sr = (bs.get('mouthSmileLeft', 0.0), bs.get('mouthSmileRight', 0.0))
        ml, mr = (bs.get('mouthLeft', 0.0), bs.get('mouthRight', 0.0))
        steady = abs(yaw) <= MOUTH_PULL_MAX_YAW and bs.get('jawOpen', 0.0) < MOUTH_OPEN_JAW
        if steady and abs(sl - sr) < MOUTH_PULL_LOPSIDED / 2 and (max(ml, mr) < MOUTH_PULL_SCORE / 2) and ((sl + sr) / 2 < 0.3):
            self.rest_shift += 0.02 * (here - self.rest_shift)
        off = here - self.rest_shift
        self.side_seen = (off, sl, sr, ml, mr)
        if not steady or abs(off) < MOUTH_SHIFT_MIN_FW:
            return 0
        way = 1 if off > 0 else -1
        agrees = (sl - sr if way > 0 else sr - sl) >= MOUTH_PULL_LOPSIDED or (ml if way > 0 else mr) >= MOUTH_PULL_SCORE
        return way if abs(off) >= MOUTH_PULL_FW and agrees or side_score >= side_thr else 0

    @classmethod
    def scores(cls, bs: dict[str, float], yaw: float=0.0) -> dict[str, float]:
        left, right = (bs.get('mouthSmileLeft', 0.0), bs.get('mouthSmileRight', 0.0))
        side_weight = min(abs(yaw) / cls.SMILE_TURN_DEG, 1.0)
        return {'smile': (left + right) / 2 + side_weight * (max(left, right) - (left + right) / 2), 'side': max(bs.get('mouthLeft', 0.0), bs.get('mouthRight', 0.0))}

    def threshold(self, name: str, pose: Pose | None=None) -> float:
        margin = GESTURES[name][2]
        base = self.bins[name].get(pose)
        return max(self.minimum[name], (self.base.get(name, 0.0) if base is None else base) + margin)

    def reset(self) -> None:
        self.active, self.low_since = (None, None)

    def progress(self, now: float) -> float:
        return 0.0 if self.active is None else min(1.0, (now - self.since) / self.hold_s())

    def update(self, bs: dict[str, float] | None, now: float, pose: Pose | None=None, ignore=frozenset()) -> str | None:
        """
        Feed one frame's expression scores (with ``mouthShift`` for the side
        gestures). Returns a gesture name when one fires. Gestures in
        ``ignore`` don't fire (their resting levels are still learned).
        """
        if not self.relaxed and now - self.fired_at > self.RELAX_MAX_S:
            self.relaxed = True
        if bs is None:
            self.reset()
            return None
        yaw = 0.0 if pose is None else float(pose[0])
        sc = self.scores(bs, yaw)
        thr = {n: self.threshold(n, pose) for n in sc}
        side = self._side(bs, sc['side'], thr['side'], yaw)
        names = {'smile': 'smile', 'side': SIDE_GESTURES.get(side)}
        on = {'smile': sc['smile'] >= thr['smile'] and (not side), 'side': bool(side)}
        above = {names[n] for n in sc if on[n] and names[n] is not None and (names[n] not in ignore)}
        margin = {names[n]: sc[n] - thr[n] for n in sc if names[n] is not None}
        for n, v in sc.items():
            if v < 0.6 * thr[n]:
                self.base[n] = v if n not in self.base else self.base[n] + 0.01 * (v - self.base[n])
                self.bins[n].learn(pose, v)
        if not side and all((v < 0.6 * thr[n] for n, v in sc.items())):
            self.relaxed = True
        if self.active is not None:
            if self.active in above:
                self.low_since = None
            else:
                self.low_since = self.low_since or now
                if now - self.low_since > self.GRACE_S:
                    self.reset()
                    return None
            if now - self.since >= self.hold_s():
                fired, self.released, self.relaxed = (self.active, False, False)
                self.fired_at = now
                self.reset()
                return fired
            return None
        if not above:
            self.released = True
            return None
        if self.released:
            name = max(above, key=lambda n: margin[n])
            if name == 'mouth_right' and max(bs.get('mouthLeft', 0.0), bs.get('mouthRight', 0.0)) < MOUTH_PULL_SCORE:
                return None
            self.active = name
            self.since, self.low_since = (now, None)
            off, sl, sr, ml, mr = self.side_seen
            self.detail = f"mouth {abs(off):.3f} face widths to your {('left' if off > 0 else 'right')}, mouthLeft {ml:.2f} / mouthRight {mr:.2f}, smile {sl:.2f} / {sr:.2f}"
        return None

class TiltScroll:
    """
    Scroll by tilting the head - an ear towards a shoulder - and holding it:
    left = up, right = down. The further the tilt, the faster (SLOW to FAST
    wheel notches a second over FULL_DEG past the start); straightening the
    head stops it. Tilt isn't used for pointing, isn't an expression (glasses
    don't matter) and doesn't change when talking, so scrolling never fights
    the cursor, blinks or voice typing.

    The tilt is measured from the user's usual head tilt (learned while the
    head is near it). Scrolling starts once it passes START_DEG for START_S -
    a glance at a screen corner or a quick lean doesn't scroll - and stops
    when it falls below STOP_FRAC of that. ``tilted`` is true from the moment
    it passes START_DEG: the cursor is held from then on - and put back where
    it was when the head was last straight (``calm_at``), as the nose moves
    sideways with the tilt and had moved the cursor on the way.
    """
    START_DEG = 22.0
    CALM_DEG = 5.0
    START_S = 0.35
    STOP_FRAC = 0.7
    FULL_DEG = 15.0
    SLOW, FAST = (2.0, 14.0)
    LEARN_DEG = 9.0
    LEARN_RATE = 0.01
    USUAL_MAX_DEG = 12.0
    WHEEL = 120
    MEDIAN_FRAMES = 5

    def __init__(self):
        self.usual: float | None = None
        self.calm_at = -1000000000.0
        self.recent: deque[float] = deque(maxlen=self.MEDIAN_FRAMES)
        self.reset()

    def reset(self) -> None:
        self.since: float | None = None
        self.direction = 0
        self.carry = 0.0
        self.t: float | None = None
        self.notches = 0

    @property
    def active(self) -> bool:
        return self.direction != 0

    @property
    def tilted(self) -> bool:
        return self.since is not None

    def update(self, roll: float | None, now: float) -> int:
        """
        Feed one frame's head tilt (fe.head_roll; None: no face, or gestures
        off). Returns the wheel turn to send now, in Windows' units (+ = up).
        """
        if roll is None:
            self.recent.clear()
            self.reset()
            return 0
        self.recent.append(roll)
        settled = len(self.recent) == self.MEDIAN_FRAMES
        roll = float(np.median(self.recent))
        if self.usual is None:
            self.usual = float(np.clip(roll, -self.USUAL_MAX_DEG, self.USUAL_MAX_DEG))
        rel = roll - self.usual
        dt = 0.0 if self.t is None else min(max(now - self.t, 0.0), 0.1)
        self.t = now
        if abs(rel) < self.CALM_DEG:
            self.calm_at = now
        if self.active:
            if abs(rel) < self.STOP_FRAC * self.START_DEG or np.sign(rel) != self.direction:
                log.info(f"[scroll] Stopped ({self.notches} notches {('up' if self.direction > 0 else 'down')})")
                self.reset()
                self.t = now
                return 0
        elif abs(rel) >= self.START_DEG and settled:
            self.since = self.since if self.since is not None else now
            if now - self.since < self.START_S:
                return 0
            self.direction, self.carry, self.notches = (int(np.sign(rel)), 1.0, 0)
            log.info(f"[scroll] Head tilted {('left' if rel > 0 else 'right')} {abs(rel):.0f} deg: scrolling {('up' if rel > 0 else 'down')}")
        else:
            self.since = None
            if abs(rel) < self.LEARN_DEG:
                self.usual = float(np.clip(self.usual + self.LEARN_RATE * rel, -self.USUAL_MAX_DEG, self.USUAL_MAX_DEG))
            return 0
        past = (abs(rel) - self.START_DEG) / self.FULL_DEG
        self.carry += (self.SLOW + (self.FAST - self.SLOW) * float(np.clip(past, 0.0, 1.0))) * dt
        n = int(self.carry)
        self.carry -= n
        self.notches += n
        return n * self.WHEEL * self.direction
BLINK_SCORE_DELTA = 0.25
BLINK_MAX_MOVE_FW = 0.15
BLINK_MOVING_FW_S = 0.5
BLINK_PASSING_MAX_S = 0.5
BLINK_DEEP = 1.25
BLINK_MIN_S = 0.3
BLINK_MOVING_MIN_S = 0.4
BLINK_CALIBRATED_MAX_S = BLINK_MIN_S
BLINK_RIGHT_S = 1.3
BLINK_MAX_S = 2.4
CLICK_REWIND_S = 0.13
DEG_PER_FACE_WIDTH = 29.0
BLINK_SCORE_BASE_MAX = 0.5

class BlinkDetector:
    """
    Turns per-eye EAR (eyelid geometry) and MediaPipe's learned blink scores
    into deliberate-blink clicks.

    Two signals: an eye counts as closed when EITHER says so, each judged
    against the user's own open-eye level. With glasses, the eyelid points
    can be thrown off by the frames while the learned score still rises (on
    the developer's face it only reached ~0.33 in a blink, but from an
    open-eye level of 0.02 - so it is judged relative to that level).

    Signal clean-up
      * Each eye's EAR is median-filtered over 3 frames, removing one-frame
        landmark glitches that used to look like (or break up) a blink.
      * A blink needs BOTH eyes closed; a squint doesn't count.
      * Winks: ONE eye closed while the other stays clearly open. The left
        eye gives a left click, the right eye a right click (both eyes: a
        left click). Not while the head is turned far to a side
        (``winks_ok``), where one eye is partly hidden and looks closed. If
        the other eye closes too during a wink, it is a normal blink.

    Thresholds with hysteresis
      Eyes become CLOSED when both drop below close_ratio x baseline, and are
      OPEN again only once their average rises above open_ratio x baseline.
      The gap stops a blink near the threshold being split into two clicks
      or ending early. The baseline is a slow running average of the user's
      open-eye EAR (~2 s), so looking down - which lowers the eyelids to about
      70-80% of normal - doesn't trigger clicks, while a real blink drives
      EAR to about 20-40%. --ear-threshold sets a fixed close threshold.

    Head position and movement
      How open the eyes look depends on where the head points: looking at
      the bottom of the screen lowers the eyelids and raises the blink score,
      so moving the head there looked like a blink and clicked (a real
      user: "due to the position of my face it registers a blink"). So:
      * the open-eye levels are learned per head direction (``pose``,
        PoseBins); during a closure, those of where it began are used;
      * head MOVEMENT is measured with the nose (``head``, in face widths),
        not with MediaPipe's head angles: those jump ~8 degrees when both
        eyes close, which cancelled every deliberate blink of a real user;
      * real blinks and fake ones (eyes that only LOOK closed where the head
        went) differ in depth: a blink shuts the eyelids to ~20-45% of open,
        a head position lowers them to ~60-80%. With the head moving, only a
        DEEP closure (BLINK_DEEP, eyelids alone) counts - and then it clicks,
        even while moving or tilting the head (a real user blinks that way);
      * eyes that close while the head is moving (> BLINK_MOVING_FW_S) don't
        hold the cursor at first (``deferred``): a natural blink mid-turn
        never does - holding it made turns stutter. Once the closure is deep
        and as long as a click, the cursor is held, back where it began;
      * a SHALLOW closure during which the head moves more than
        BLINK_MAX_MOVE_FW is the head moving, not a blink: no click, and the
        eyes count as open until they look open again - learning that
        direction's open-eye level meanwhile, so blinks work there right after.

    Click timing
      A click fires when the eyes REOPEN after being closed for min_s..max_s:
      shorter closures are involuntary blinks and are ignored; longer ones
      mean the user is resting their eyes. Both eyes closed for right_s or
      more (BLINK_RIGHT_S, ~1.3 s) is a RIGHT click: many people can't wink
      one eye cleanly (a real user's right wink closed both eyes, the right
      only a little more - indistinguishable from a blink), but everyone can
      blink longer. A cooldown blocks double-clicks, and the short natural
      blink that often follows resting the eyes (within AFTER_REST_S of
      opening them) is not a click either.

    Ticks while the eyes are closed (``cue``, played by the worker): "tick"
    the moment opening the eyes would click, "double" the moment it would
    right-click. They wait out the cooldown and the after-rest rule,
    so a tick always means "open now and it clicks" - and teaches how long
    to close the eyes, the most common reason a blink was missed.
    """
    AFTER_REST_S = 1.0
    AFTER_REST_MAX_BLINK_S = 0.4
    MOVING_MIN_S = BLINK_MOVING_MIN_S
    TICK_WHEN_SHUT = True
    SHUT = 1.0
    KEEP_TICKED = True
    HEAD_FRAMES = 12

    def __init__(self, fixed_threshold: float | None, close_ratio: float, open_ratio: float, min_s: float, max_s: float, cooldown_s: float, right_s: float | None=BLINK_RIGHT_S):
        self.fixed = fixed_threshold
        self.close_ratio, self.open_ratio = (close_ratio, open_ratio)
        self.min_s, self.max_s, self.cooldown_s = (min_s, max_s, cooldown_s)
        self.right_s = right_s
        self.cue = ''
        self.cued = 0
        self.shut = False
        self.baseline: float | None = None
        self.history: deque[tuple[float, float]] = deque(maxlen=3)
        self.score_history: deque[tuple[float, float]] = deque(maxlen=3)
        self.score_base: float | None = None
        self.score_delta = BLINK_SCORE_DELTA
        self.ear_bins = PoseBins()
        self.score_bins = PoseBins()
        self.pose: Pose | None = None
        self.heads: deque[tuple[float, np.ndarray]] = deque(maxlen=self.HEAD_FRAMES)
        self.closed_pose: Pose | None = None
        self.closed_head: np.ndarray | None = None
        self.peak, self.peak_weaker = (0.0, 0.0)
        self.moved = 0.0
        self.moved_by = np.zeros(2)
        self.head_moved = False
        self.deep = 0.0
        self.moving_start = False
        self.deferred = False
        self.started_at = 0.0
        self.closed_since: float | None = None
        self.last_click = 0.0
        self.rested_at = -1000000000.0
        self.wink_eye: str | None = None
        self.wink_since = 0.0
        self.wink_by_lids = False
        self.last_hint = -1000000000.0
        self.value: float | None = None

    def _ear_base(self) -> float:
        base = self.ear_bins.get(self.pose)
        return base if base is not None else self.baseline or 0.28

    def thresholds(self) -> tuple[float, float]:
        """(close_threshold, open_threshold) for the current head direction."""
        if self.fixed is not None:
            return (self.fixed, self.fixed * self.open_ratio / self.close_ratio)
        base = self._ear_base()
        return (float(np.clip(self.close_ratio * base, 0.08, 0.28)), float(np.clip(self.open_ratio * base, 0.1, 0.32)))

    def closed_for(self, now: float) -> float:
        return now - self.closed_since if self.closed_since is not None else 0.0

    def reset(self) -> None:
        """Forget any in-progress closure (e.g. the face was lost)."""
        self.closed_since = None
        self.wink_eye = None
        self.head_moved = self.deferred = False
        self.history.clear()
        self.score_history.clear()
        self.heads.clear()
        self.value = None

    def closedness(self, l: float, r: float, scores) -> tuple[float, float]:
        """Per eye: 1.0 = at the close threshold, ~0.67 = the reopen threshold."""
        close_thr, _ = self.thresholds()
        base = close_thr / self.close_ratio
        c = [(base - e) / max(base - close_thr, 1e-06) for e in (l, r)]
        if scores is not None:
            sb = self.score_bins.get(self.pose)
            if sb is None:
                sb = self.score_base if self.score_base is not None else min(scores)
            c = [max(ce, (sc - sb) / self.score_delta) for ce, sc in zip(c, scores)]
        return (c[0], c[1])

    def eyes_open(self, left: float, right: float, pose: Pose | None=None) -> bool:
        """Both eyes open by their eyelids, for this head direction (close gesture)."""
        self.pose = pose
        return max(self.closedness(left, right, None)) < 1.0

    def learn_open(self, left: float, right: float, scores, pose: Pose | None) -> None:
        """How open eyes look in this head direction (from the calibration)."""
        self.pose = pose
        self._learn((left + right) / 2.0, scores)

    def _learn(self, mean: float, sc, overall: bool=True) -> None:
        """Eyes open: adapt the open-eye levels (overall ~2 s, and this direction's)."""
        m = None if sc is None else min(float(np.mean(sc)), BLINK_SCORE_BASE_MAX)
        if overall:
            self.baseline = mean if self.baseline is None else self.baseline + 0.015 * (mean - self.baseline)
            if m is not None:
                self.score_base = m if self.score_base is None else self.score_base + 0.015 * (m - self.score_base)
        self.ear_bins.learn(self.pose, mean)
        if m is not None:
            self.score_bins.learn(self.pose, m)
    WINK_OTHER_OPEN = 0.6

    def update(self, left: float, right: float, now: float, scores: tuple[float, float] | None=None, winks_ok: bool=True, pose: Pose | None=None, head=None) -> tuple[bool, str, bool]:
        """
        Feed one frame: per-eye EAR, and optionally per-eye blink scores
        (MediaPipe eyeBlinkLeft/Right), the head pose and the head position
        (the nose in face widths, see _process). Returns (eyes_closed,
        click_fired, just_closed). click_fired is "" (none), "left" or "right".
        """
        self.cue = ''
        self.history.append((left, right))
        l, r = np.median(np.array(self.history), axis=0)
        sc = None
        if scores is not None:
            self.score_history.append(scores)
            sc = tuple(np.median(np.array(self.score_history), axis=0))
        mean = float(l + r) / 2.0
        self.value = mean
        closing = self.wink_eye is not None or self.closed_since is not None
        self.pose = self.closed_pose if closing else pose
        head = self._head(head, pose)
        if head is not None:
            self.heads.append((now, head))
        cl, cr = self.closedness(l, r, sc)
        el, er = self.closedness(l, r, None)
        reopen = (1 - self.open_ratio) / max(1 - self.close_ratio, 1e-06)
        if self.head_moved:
            if (cl + cr) / 2 < reopen:
                self.head_moved = False
            else:
                self._learn(mean, sc, overall=False)
                return (False, '', False)
        if closing:
            self.deep = max(self.deep, self._depth(el, er))
            if head is not None and self.closed_head is not None:
                by = head - self.closed_head
                if float(np.linalg.norm(by)) > self.moved:
                    self.moved, self.moved_by = (float(np.linalg.norm(by)), by)
            if self.moved > BLINK_MAX_MOVE_FW and self.deep < BLINK_DEEP and (not (self.KEEP_TICKED and self.cued)):
                started = self.wink_since if self.wink_eye is not None else self.closed_since
                note = log.info if now - started >= self.min_s else log.debug
                note(f"[blink] Head moved while the eyes only looked half closed ({(now - started) * 1000:.0f} ms{(', began while turning' if self.deferred else '')}; eyelids {self.deep:.2f} of {BLINK_DEEP:g}, {self._moved_text()}) - not a blink, no click")
                self.closed_since, self.wink_eye, self.deferred = (None, None, False)
                self.head_moved = True
                return (False, '', False)
        if self.wink_eye is not None:
            return self._wink(cl, cr, el, er, reopen, now)
        if self.closed_since is None:
            lid_wink = winks_ok and max(el, er) >= 1.0 and (min(el, er) < self.WINK_OTHER_OPEN)
            both = ((cl + cr) / 2 >= 1.0 and min(cl, cr) >= self.WINK_OTHER_OPEN) and (not lid_wink)
            wink = lid_wink or (winks_ok and max(cl, cr) >= 1.0 and (min(cl, cr) < self.WINK_OTHER_OPEN))
            if not (both or wink):
                self._learn(mean, sc)
                return (False, '', False)
            moving = self._head_speed(now) > BLINK_MOVING_FW_S
            if both:
                self.closed_since = now
            else:
                self.wink_by_lids = lid_wink
                shut_left = el >= er if lid_wink else cl >= 1.0
                self.wink_eye, self.wink_since = ('left' if shut_left else 'right', now)
            self._closing(pose, head, now, moving, self._depth(el, er))
            return (not moving, '', not moving)
        if (cl + cr) / 2 >= reopen:
            self.shut = (cl + cr) / 2 >= self.SHUT and min(cl, cr) >= self.WINK_OTHER_OPEN
            self.peak, self.peak_weaker = (max(self.peak, (cl + cr) / 2), max(self.peak_weaker, min(cl, cr)))
            return self._still_closed(now)
        duration = now - self.closed_since
        self.closed_since = None
        if self.deferred:
            self.deferred = False
            return (False, '', False)
        click = ''
        after_rest = now - duration - self.rested_at < self.AFTER_REST_S and duration < self.AFTER_REST_MAX_BLINK_S
        if self.cued >= 1 and self.min_s <= duration <= self.max_s and (now - self.last_click >= self.cooldown_s) and (not after_rest):
            self.last_click = now
            click = 'right' if self.cued == 2 else 'left'
            log.info(f"[blink] {('right click (long blink, ' if click == 'right' else 'click (')}{duration * 1000:.0f} ms closure{(', head moving' if self.moving_start or self.moved > BLINK_MAX_MOVE_FW else '')})")
        elif self.min_s <= duration <= self.max_s and after_rest:
            log.info(f'[blink] blink right after resting the eyes ({duration * 1000:.0f} ms) - not a click')
        elif duration > self.max_s:
            self.rested_at = now
            log.info(f'[blink] eyes closed {duration:.1f}s - treated as resting, no click')
        elif 0.15 <= duration and self.cued == 0 and (duration >= BLINK_MIN_S or now - self.last_hint > 20):
            self.last_hint = now
            first = self._first_tick()
            why = 'the eyes were opening before the tick' if duration >= first else 'too short for a click'
            # Numbers for tuning: which of the tick's conditions was missed.
            seen = f'deepest {self.peak:.2f}, weaker eye {self.peak_weaker:.2f}'
            if now - duration + first - self.last_click < self.cooldown_s:
                seen += ', just after a click'
            log.info(f'[blink] {duration * 1000:.0f} ms closure - {why} (keep your eyes shut until the tick, about {first:.1f} s; {seen})')
        return (False, click, False)

    def _depth(self, el: float, er: float) -> float:
        """
        How far the closing eye(s) shut by the eyelids alone (closedness
        units). A blink: both eyes on average - many people close one eye
        less (a real user's right eye stopped at 1.2-1.3 while the left
        reached 1.4-1.5, so judged by the less closed eye, firm blinks
        counted as "half closed" and were dropped). A head position that only
        makes the eyes LOOK closed lowers both alike, to about 1.0.
        """
        if self.wink_eye == 'left':
            return el
        if self.wink_eye == 'right':
            return er
        return (el + er) / 2

    def _closing(self, pose: Pose | None, head: np.ndarray | None, now: float, moving: bool, depth: float) -> None:
        """A closure starts: remember where the head was, and whether it was moving."""
        self.closed_pose = pose
        earlier = [h for t, h in self.heads if t <= now - 0.1]
        self.closed_head = earlier[-1] if earlier else head
        self.moved, self.moved_by, self.deep = (0.0, np.zeros(2), depth)
        self.peak, self.peak_weaker = (0.0, 0.0)
        self.moving_start = self.deferred = moving
        self.started_at = now
        self.cued = 0

    def _cue(self, now: float) -> None:
        """
        Tick once opening the eyes now would click (see the class
        docstring): "tick" when it becomes a click, "double" when it becomes
        a right click (both eyes only - a wink's button is its eye).
        """
        wink = self.wink_eye is not None
        started = self.wink_since if wink else self.closed_since
        if started is None or (self.TICK_WHEN_SHUT and (not self.shut)):
            return
        held = now - started
        after_rest = started - self.rested_at < self.AFTER_REST_S
        first = self._first_tick(after_rest and (not wink))
        if self.cued == 0 and first <= held <= self.max_s and (now - self.last_click >= self.cooldown_s):
            self.cued, self.cue = (1, 'tick')
        elif self.cued == 1 and (not wink) and (self.right_s is not None) and (self._double_tick(first) <= held <= self.max_s):
            self.cued, self.cue = (2, 'double')

    def _first_tick(self, after_rest: bool=False) -> float:
        """When this closure's tick comes: later right after resting the eyes,
        or when it began with the head moving (natural blinks while reading)."""
        return max(self.min_s, self.AFTER_REST_MAX_BLINK_S if after_rest else 0.0, self.MOVING_MIN_S if self.moving_start else 0.0)

    def _double_tick(self, first: float) -> float:
        """When the double tick comes: as long after the first tick as the
        defaults leave (1.0 s), also when a calibrated tick came later - 1.5.9
        kept it at 1.3 s, so a 0.6 s tick left 0.7 s, and the user opens
        0.55-0.9 s after the tick they wait for: left clicks became right ones."""
        return self.right_s + first - min(self.min_s, BLINK_MIN_S)

    def _still_closed(self, now: float) -> tuple[bool, str, bool]:
        """
        A closure goes on. Started while the head moved (``deferred``), the
        cursor keeps moving until it is clearly a click: deep (BLINK_DEEP) and
        as long as a click (min_s) - then it is held, back where the blink
        began (just_closed). A natural blink mid-turn never holds it.
        """
        if not self.deferred:
            self._cue(now)
            return (True, '', False)
        if self.deep >= BLINK_DEEP and now - self.started_at >= self._first_tick():
            self.deferred = False
            self._cue(now)
            return (True, '', True)
        if now - self.started_at > BLINK_PASSING_MAX_S:
            log.info(f'[blink] Eyes closed while the head moved, but the eyelids never shut fully ({(now - self.started_at) * 1000:.0f} ms; eyelids {self.deep:.2f} of {BLINK_DEEP:g}, {self._moved_text()}) - no click')
            self.closed_since, self.wink_eye, self.deferred = (None, None, False)
            self.head_moved = True
        return (False, '', False)

    def _moved_text(self) -> str:
        """How the head moved during this closure, for the log: sideways
        (turning) or up-down (e.g. looking at the screen bottom)."""
        x, y = (abs(float(v)) for v in self.moved_by)
        return f'head moved {self.moved:.2f} face widths ({x:.2f} sideways, {y:.2f} up-down)'

    @staticmethod
    def _head(head, pose: Pose | None) -> np.ndarray | None:
        """Head position in face widths: the nose (preferred), else from the angles."""
        if head is not None:
            return np.asarray(head, np.float64)
        if pose is not None:
            return np.array(pose, np.float64) / DEG_PER_FACE_WIDTH
        return None

    def _head_speed(self, now: float) -> float:
        """How fast the head moved over the last ~0.2 s (face widths a second)."""
        recent = [(t, h) for t, h in self.heads if now - t <= 0.25]
        if len(recent) < 3:
            return 0.0
        (t0, h0), (t1, h1) = (recent[0], recent[-1])
        return float(np.linalg.norm(h1 - h0)) / max(t1 - t0, 0.001)

    def _wink(self, cl: float, cr: float, el: float, er: float, reopen: float, now: float) -> tuple[bool, str, bool]:
        """
        One eye closed (see the class docstring): left eye = left click, right
        = right. A wink the eyelids showed is followed by the eyelids (the
        blink scores of both eyes rise during it); otherwise by both signals.
        """
        if self.wink_by_lids:
            cl, cr = (el, er)
        shut, other = (cl, cr) if self.wink_eye == 'left' else (cr, cl)
        if other >= self.WINK_OTHER_OPEN:
            self.wink_eye, self.closed_since = (None, self.wink_since)
            self.shut = (cl + cr) / 2 >= self.SHUT and min(cl, cr) >= self.WINK_OTHER_OPEN
            return self._still_closed(now)
        if shut >= reopen:
            self.shut = shut >= self.SHUT
            return self._still_closed(now)
        eye, self.wink_eye = (self.wink_eye, None)
        if self.deferred:
            self.deferred = False
            return (False, '', False)
        duration = now - self.wink_since
        if self.cued >= 1 and self.min_s <= duration <= self.max_s and (now - self.last_click >= self.cooldown_s):
            self.last_click = now
            log.info(f"[blink] {eye} click ({eye} eye closed {duration * 1000:.0f} ms{(', head moving' if self.moving_start or self.moved > BLINK_MAX_MOVE_FW else '')})")
            return (False, eye, False)
        return (False, '', False)

class CloseGesture:

    def _close_gesture(self, mar: float, eyes_open: bool, now: float, jaw: float | None=None) -> bool:
        """
        Mouth wide open + eyes open, held for --close-gesture-hold seconds ->
        shut EyeSync down. Returns True while the gesture is in progress.
        ``jaw`` is MediaPipe's jawOpen score, when available.
        """
        if self.args.no_close_gesture:
            return False
        if self.state.stop.is_set():
            return True
        if self.gesture_since is None:
            if mar >= self.args.close_gesture_mar and eyes_open and (jaw is None or jaw >= CLOSE_JAW_MIN):
                self.gesture_since = now
                self.gesture_low_since = None
                self.gesture_announced = False
                return True
            return False
        if mar >= self.args.close_gesture_mar * GESTURE_KEEP:
            self.gesture_low_since = None
        else:
            self.gesture_low_since = self.gesture_low_since or now
            grace = GESTURE_CANCEL_S if mar < GESTURE_CLOSED_MAR else GESTURE_GRACE_S
            if now - self.gesture_low_since > grace:
                if self.gesture_announced:
                    log.info(f'[gesture] Close gesture cancelled (mouth closed to {mar:.2f} after {now - self.gesture_since:.1f}s)')
                self.gesture_since = None
                return False
        if not self.gesture_announced and self.gesture_low_since is None and (now - self.gesture_since >= GESTURE_CONFIRM_S):
            self.gesture_announced = True
            log.info(f'[gesture] Mouth wide open - hold {self.args.close_gesture_hold:.0f}s to close EyeSync (close your mouth to cancel)')
            desktop.play_warning_sound()
        if now - self.gesture_since >= self.args.close_gesture_hold:
            log.info('[gesture] Close gesture completed - shutting EyeSync down')
            desktop.play_click_sound()
            self.state.stop.set()
        return True

GESTURE_DROPOUT_S = 0.7
GESTURE_KEEP = 0.65
GESTURE_GRACE_S = 0.6
GESTURE_CLOSED_MAR = 0.2
GESTURE_CANCEL_S = 0.25
GESTURE_CONFIRM_S = 0.5
CLOSE_JAW_MIN = 0.3
