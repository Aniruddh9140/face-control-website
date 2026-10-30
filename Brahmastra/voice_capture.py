"""Bounded microphone capture for the retained EyeSync offline voice engine.

No device is opened at import time. Capture discontinuities are explicit errors:
silently dropping old audio could join unrelated words into a misleading phrase.
"""
from __future__ import annotations

import queue
import threading

import numpy as np

import voice_typing as vt


class Microphone(vt.Microphone):
    """Original mono capture/resampling with bounded buffering and safe cleanup."""

    # Recognition runs on the thread reading this queue, so it must hold speech
    # for the slowest phrase: 5 s here ended real sessions after long phrases.
    MAX_BLOCKS = 1200            # callbacks contain 50 ms: at most one minute

    def __init__(self, rate: int = vt.RATE):
        if not isinstance(rate, int) or rate <= 0:
            raise ValueError("The microphone sample rate must be a positive integer")
        super().__init__(rate)
        self.blocks = queue.Queue(maxsize=self.MAX_BLOCKS)
        self._faults = queue.Queue(maxsize=1)
        self._fault_detail = ""
        self._failed = threading.Event()
        self._accepting = threading.Event()

    def _clear_buffers(self):
        for buffer in (self.blocks, self._faults):
            while True:
                try:
                    buffer.get_nowait()
                except queue.Empty:
                    break
        self._fault_detail = ""
        self._failed.clear()

    def _fail(self, message):
        # PortAudio's callback must never wait for a consumer or a queue slot.
        try:
            self._faults.put_nowait(message)
        except queue.Full:
            pass
        self._failed.set()

    def _callback(self, data, frames, when, status):
        if not self._accepting.is_set() or self._failed.is_set():
            return
        try:
            if status:
                self._fail("Microphone audio was interrupted or overflowed. Restart voice typing.")
                return
            samples = np.asarray(data)
            if (samples.ndim != 2 or samples.shape != (frames, 1)
                    or frames <= 0 or not np.isfinite(samples).all()):
                self._fail("Microphone returned invalid audio. Restart voice typing.")
                return
            # The callback owns data only until it returns.
            self.blocks.put_nowait(np.array(samples[:, 0], dtype=np.float32, copy=True))
        except queue.Full:
            self._fail("Speech capture fell behind by a minute. Restart and use shorter phrases.")
        except Exception:  # callback errors must be reported on the consumer thread
            self._fail("Microphone capture failed. Restart voice typing.")

    @staticmethod
    def _dispose(stream):
        try:
            stream.stop()
        except Exception:
            pass
        finally:
            try:
                stream.close()
            except Exception:
                pass

    def open(self):
        import sounddevice as sd

        self.close()
        self._clear_buffers()
        try:
            info = sd.query_devices(kind="input")
        except Exception as exc:
            raise OSError("No microphone found") from exc
        self.name = str(info.get("name", "microphone"))
        try:
            native = int(info.get("default_samplerate") or 0)
        except (TypeError, ValueError, OverflowError):
            native = 0
        rates = list(dict.fromkeys(rate for rate in (self.rate, native) if rate > 0))
        error = None
        for rate in rates:
            stream = None
            self._clear_buffers()
            try:
                stream = sd.InputStream(samplerate=rate, channels=1, dtype="float32",
                                        blocksize=max(1, int(rate * .05)), callback=self._callback)
                self.native = rate
                self._accepting.set()
                stream.start()
                self.stream = stream
                return
            except Exception as exc:
                error = exc
                self._accepting.clear()
                if stream is not None:
                    self._dispose(stream)
        self._clear_buffers()
        raise OSError(f"Cannot start the microphone: {error}") from error

    def _raise_fault(self):
        if self._failed.is_set():
            if not self._fault_detail:
                try:
                    self._fault_detail = self._faults.get_nowait()
                except queue.Empty:
                    self._fault_detail = "Microphone audio was interrupted. Restart voice typing."
            raise OSError(self._fault_detail)

    def check_health(self):
        """Reject delivery if capture failed while a phrase was being decoded."""
        self._raise_fault()

    def read(self, timeout):
        self._raise_fault()
        try:
            block = self.blocks.get(timeout=timeout)
        except queue.Empty:
            self._raise_fault()
            return None
        self._raise_fault()  # queued audio is unusable after a capture discontinuity
        return block if self.native == self.rate else vt.resample(block, self.native, self.rate)

    def close(self):
        self._accepting.clear()
        stream, self.stream = self.stream, None
        if stream is not None:
            self._dispose(stream)
        self._clear_buffers()
