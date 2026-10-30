"""Optional Accurate voice mode: a larger Whisper model, fetched once on request.

It isn't bundled because EyeSync.exe unpacks itself at every start, so 190 MB
more would slow every launch for everyone, including people who never use it.
Only the model is downloaded; speech is still recognised on this PC.
"""
from __future__ import annotations

import hashlib
import threading
import urllib.request

import app_paths
import voice_typing as vt

FILE = "ggml-small.en-q5_1.bin"
URL = f"https://huggingface.co/ggerganov/whisper.cpp/resolve/main/{FILE}"
SHA1 = "20f54878d608f94e4a8ee3ae56016571d47cba34"
BYTES = 190098681
# Outside DATA_DIR: from source that folder is on OneDrive, which would sync 190 MB.
# shortcut: an uninstall leaves this file behind, remove it there if users complain.
FOLDER = app_paths.RUNTIME_DIR.parent / "models"
FLAG = app_paths.DATA_DIR / "data" / "accurate_voice.on"
_fetching = threading.Lock()


def model_path():
    return FOLDER / FILE


def enabled():
    return FLAG.exists()


def set_enabled(on):
    if on:
        FLAG.parent.mkdir(parents=True, exist_ok=True)
        FLAG.touch()
    else:
        FLAG.unlink(missing_ok=True)


def ready():
    # Size only: this runs at every session start. Re-ticking the box runs fetch(), which hashes.
    path = model_path()
    return path.is_file() and path.stat().st_size == BYTES


def recognizer():
    """The model for one session: Accurate only once its verified file is here."""
    return vt.Recognizer(model_path()) if enabled() and ready() else vt.Recognizer()


def fetch(progress):
    """Download unless a good copy is here; a bad file is never kept. False if already busy."""
    if not _fetching.acquire(blocking=False):
        return False
    part = model_path().with_suffix(".part")
    try:
        if ready():
            with open(model_path(), "rb") as existing:
                if hashlib.file_digest(existing, "sha1").hexdigest() == SHA1:
                    return True
        FOLDER.mkdir(parents=True, exist_ok=True)
        digest, done = hashlib.sha1(), 0
        with urllib.request.urlopen(URL, timeout=30) as source, open(part, "wb") as out:
            while chunk := source.read(1 << 20):
                out.write(chunk)
                digest.update(chunk)
                done += len(chunk)
                progress(done * 100 // BYTES)
        if digest.hexdigest() != SHA1:
            raise ValueError("the download failed its safety check")
        part.replace(model_path())
        return True
    finally:
        part.unlink(missing_ok=True)
        _fetching.release()
