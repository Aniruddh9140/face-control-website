"""Keyboard batches use fake ctypes records and a fake Win32 SendInput only."""
import ctypes
import unittest
from unittest.mock import Mock, patch

from improved_voice import ReliableTypist
import voice_typing as vt

class KeyRecord(ctypes.Structure):
    _fields_ = [('vk',ctypes.c_int),('scan',ctypes.c_int),('flags',ctypes.c_int),
                ('time',ctypes.c_int),('extra',ctypes.c_int)]

class InputRecord(ctypes.Structure):
    _fields_ = [('type',ctypes.c_int),('ki',KeyRecord)]

class DeliveryTests(unittest.TestCase):
    def typist(self, returns):
        t=ReliableTypist.__new__(ReliableTypist)
        t._INPUT,t._KEYBDINPUT=InputRecord,KeyRecord
        t._user32=Mock()
        calls=[]
        answers=iter(returns)
        def send(count,pointer,size):
            data=ctypes.cast(pointer,ctypes.POINTER(InputRecord*count)).contents
            calls.append([(item.ki.vk,item.ki.scan,item.ki.flags) for item in data])
            result=next(answers)
            if isinstance(result,Exception): raise result
            return result
        t._user32.SendInput.side_effect=send
        return t,calls

    def test_partial_write_stops_and_releases_only_accepted_keydowns(self):
        t,calls=self.typist([1,1])
        self.assertEqual(t.run([('key','newline'),('text','Do not type')]),1)
        self.assertEqual(calls[1],[(vt.VK_SHIFT,0,vt.KEYEVENTF_KEYUP)])
        self.assertEqual(len(calls),2)

    def test_unicode_half_write_sends_keyup_cleanup(self):
        t,calls=self.typist([1,1])
        self.assertEqual(t.run([('text','AB')]),1)
        self.assertEqual(calls[1],[(0,ord('A'),vt.KEYEVENTF_UNICODE|vt.KEYEVENTF_KEYUP)])

    def test_cancel_before_first_batch_types_nothing(self):
        t,calls=self.typist([])
        t.cancelled=lambda:True
        self.assertEqual(t.run([('text','AB')]),0)
        self.assertEqual(calls,[])

    def test_cancellation_between_batches_stops_remaining_text(self):
        t,calls=self.typist([4])
        t.BATCH=4
        checks=iter([False,True])
        t.cancelled=lambda:next(checks)
        with patch('improved_voice.time.sleep'):
            self.assertEqual(t.run([('text','ABCDE')]),4)
        self.assertEqual(len(calls),1)

    def test_target_guard_exception_still_releases_held_shift(self):
        t,calls=self.typist([1,1])
        t.BATCH=1
        checks=iter([False,RuntimeError('target check failed')])
        def cancelled():
            answer=next(checks)
            if isinstance(answer,Exception): raise answer
            return answer
        t.cancelled=cancelled
        with patch('improved_voice.time.sleep'),self.assertRaises(RuntimeError):
            t.run([('key','newline')])
        self.assertEqual(calls[-1],[(vt.VK_SHIFT,0,vt.KEYEVENTF_KEYUP)])

if __name__=='__main__': unittest.main()
