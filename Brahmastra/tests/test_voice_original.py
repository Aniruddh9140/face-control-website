"""Unmodified standalone speech tests transplanted from EyeSync.
EyeSync UI/server tests are replaced by Brahmastra integration tests.
"""
import unittest
import numpy as np
from unittest.mock import patch
import voice_typing as vt

class VoiceTypingTests(unittest.TestCase):
    def test_phrases_join_like_writing(self):
            c = vt.Composer()
            self.assertEqual(c.plan(" Hello"), [("text", "Hello")])
            # A sentence carried on after a pause: no capital, a space between.
            self.assertEqual(c.plan(" How are you doing today?"), [("text", " how are you doing today?")])
            self.assertEqual(c.plan("I'm fine."), [("text", " I'm fine.")])
            # Whisper ended it with a full stop, but it went on: the full stop is taken back.
            self.assertEqual(c.plan("and you?"), [("back", 1), ("text", " and you?")])
            self.assertEqual(c.plan("Great."), [("text", " Great.")])
            c = vt.Composer()
            c.plan("I met")
            self.assertEqual(c.plan("John today."), [("text", " John today.")])     # names stay
            c = vt.Composer()
            c.plan("I would like to...")
            self.assertEqual(c.plan("schedule a meeting."), [("back", 3), ("text", " schedule a meeting.")])
            self.assertEqual(c.plan("Okay, thanks"), [("text", " Okay, thanks")])
            self.assertEqual(c.plan(", bye"), [("text", ", bye")])
            # Lower case after a full stop because Whisper swallowed the first word
            # ("The"): a new sentence, not a reason to take the full stop back.
            c = vt.Composer()
            c.plan("Please send the report.")
            self.assertEqual(c.plan("quick brown fox."), [("text", " Quick brown fox.")])

    def test_voice_commands(self):
            c = vt.Composer()
            self.assertEqual(c.plan("Delete that."), [])                  # nothing typed yet
            c.plan("Dear John,")
            self.assertEqual(c.plan("New line."), [("key", "newline")])
            self.assertEqual(c.plan("Thanks for the update."), [("text", "Thanks for the update.")])
            self.assertEqual(c.plan("Question mark."), [("back", 1), ("text", "?")])  # replaces the "."
            self.assertEqual(c.plan("Delete that."), [("back", 1)])
            self.assertEqual(c.plan("Scratch that"), [("back", len("Thanks for the update"))])
            self.assertEqual(c.plan("Thanks."), [("text", "Thanks.")])
            self.assertEqual(c.plan('"Kama"'), [("back", 1), ("text", ",")])    # a misheard "comma"
            self.assertEqual(c.plan("New paragraph"), [("key", "newline"), ("key", "newline")])
            self.assertEqual(c.plan("Press enter."), [("key", "enter")])
            self.assertEqual(c.plan("Tab."), [("key", "tab")])
            self.assertEqual(c.plan("Backspace"), [("key", "backspace")])
            self.assertEqual(c.plan("space"), [("text", " ")])
            self.assertEqual(c.plan("Stop listening."), [("stop", "")])
            # At the start or end of a longer phrase they still count...
            c = vt.Composer()
            self.assertEqual(c.plan("Best wishes, new line."), [("text", "Best wishes,"), ("key", "newline")])
            self.assertEqual(c.plan("New line. See you soon."), [("key", "newline"), ("text", "See you soon.")])
            self.assertEqual(c.plan("That's all, stop listening."), [("text", " That's all"), ("stop", "")])
            # ... but not in the middle of a sentence.
            self.assertEqual(vt.Composer().plan("We launched a new line of products."),
                             [("text", "We launched a new line of products.")])

    def test_sound_labels_and_video_credits_are_not_typed(self):
            for raw in ("[BLANK_AUDIO]", "(buzzing)", "[MUSIC PLAYING]", "*laughs*", "♪ ♪", " ...",
                        "Thanks for watching!", "Subtitles by the Amara.org community"):
                with self.subTest(raw):
                    self.assertEqual(vt.Composer().plan(raw), [])
            self.assertEqual(vt.clean(" Hello (laughs) there."), "Hello there.")
            self.assertEqual(vt.clean("[BLANK_AUDIO] Hello [BLANK_AUDIO]"), "Hello")

    def test_short_phrases_use_a_15_s_window(self):
            """Whisper's encoder was 1.5 of a short phrase's 1.8 s (the user's log: 2.8-3.9 s
            with face tracking running). A 15 s window is ~30% faster and as accurate
            (96 EdAcc clips); a longer phrase still gets the window it needs."""
            self.assertEqual(vt.available(), "", "voice typing is not completely installed")
            rec = vt.Recognizer()
            try:
                for secs, ctx in ((1.5, 750), (16.0, 864)):
                    rec(np.zeros(int(secs * vt.RATE), np.float32))
                    self.assertEqual(rec._params.audio_ctx, ctx)
            finally:
                rec.close()

    def test_key_presses(self):
            U, UP = vt.KEYEVENTF_UNICODE, vt.KEYEVENTF_KEYUP
            ev = vt.key_events([("text", "Hi"), ("key", "newline"), ("back", 2), ("text", "\U0001F600")])
            self.assertEqual(ev[:4], [(0, ord("H"), U), (0, ord("H"), U | UP),
                                      (0, ord("i"), U), (0, ord("i"), U | UP)])
            # A new line is Shift+Enter: in a chat box, Enter alone would send the message.
            self.assertEqual(ev[4:8], [(vt.VK_SHIFT, 0, 0), (vt.VK_RETURN, 0, 0),
                                       (vt.VK_RETURN, 0, UP), (vt.VK_SHIFT, 0, UP)])
            self.assertEqual(ev[8:12], [(vt.VK_BACK, 0, 0), (vt.VK_BACK, 0, UP)] * 2)
            self.assertEqual([e[1] for e in ev[12:]], [0xD83D, 0xD83D, 0xDE00, 0xDE00])  # 2 UTF-16 units
            self.assertEqual(vt.key_events([("key", "enter")]),
                             [(vt.VK_RETURN, 0, 0), (vt.VK_RETURN, 0, UP)])

    def test_resample(self):
            for rate in (48000, 44100):
                x = np.sin(2 * np.pi * 440 * np.arange(rate) / rate).astype(np.float32)
                y = vt.resample(x, rate, vt.RATE)
                self.assertEqual(y.size, vt.RATE)
                peak = np.argmax(np.abs(np.fft.rfft(y))) * vt.RATE / y.size
                self.assertAlmostEqual(peak, 440, delta=2)
