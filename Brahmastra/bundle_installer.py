"""Per-user self-installation for the single-file EyeSync Windows application.

No registry, browser, process or filesystem changes occur at import time.
The GUI's Install/Update button performs the same operation as --quiet.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import webbrowser

import app_paths

DEFAULT_DEST = Path(os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local"))) / "Programs" / "EyeSync-Brahmastra"
EXE_NAME = "EyeSync.exe"
SITE = "https://aniruddh9140.github.io/face-control-website/#dashboard"
RUN = r"Software\Microsoft\Windows\CurrentVersion\Run"
PROTOCOL = r"Software\Classes\eyesync"
REGISTRY_ENTRIES = [(RUN, "EyeSync"), (PROTOCOL, ""), (PROTOCOL, "URL Protocol"),
                    (PROTOCOL + r"\DefaultIcon", ""), (PROTOCOL + r"\shell\open\command", "")]
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
log = logging.getLogger("eyesync.install")


def _same_path(first: Path, second: Path) -> bool:
    return os.path.normcase(str(Path(first).resolve())) == os.path.normcase(str(Path(second).resolve()))


def is_installed_copy(exe: Path | None = None) -> bool:
    """Only the managed default path counts as the installed copy."""
    return _same_path(Path(exe or sys.executable), DEFAULT_DEST / EXE_NAME)


def _read_value(registry, path, name):
    try:
        with registry.OpenKey(registry.HKEY_CURRENT_USER, path) as key:
            value, kind = registry.QueryValueEx(key, name)
            return {"value": value, "kind": kind}
    except FileNotFoundError:
        return None


def _set_value(registry, path, name, value, kind=None):
    with registry.CreateKeyEx(registry.HKEY_CURRENT_USER, path, 0, registry.KEY_SET_VALUE) as key:
        registry.SetValueEx(key, name, 0, registry.REG_SZ if kind is None else kind, value)


def register_install(exe: Path) -> None:
    """Register only HKCU launch entries, retaining the first previous values."""
    import winreg

    exe = Path(exe).resolve()
    backup = exe.parent / "data" / "dashboard-registration-backup.json"
    before = [{"path": path, "name": name, "previous": _read_value(winreg, path, name)}
              for path, name in REGISTRY_ENTRIES]
    if not backup.exists():
        backup.parent.mkdir(parents=True, exist_ok=True)
        # A successful first backup is never overwritten by later updates.
        with backup.open("x", encoding="utf-8") as output:
            json.dump(before, output, indent=2)
            output.write("\n")
    values = [f'"{exe}" --standby', "URL:EyeSync", "", f'"{exe}",0', f'"{exe}" --url "%1"']
    applied = []
    try:
        for entry, value in zip(before, values):
            _set_value(winreg, entry["path"], entry["name"], value)
            applied.append(entry)
    except OSError:
        # Restore this attempt's prior strings if registration only half worked.
        for entry in reversed(applied):
            old = entry["previous"]
            try:
                if old is not None:
                    _set_value(winreg, entry["path"], entry["name"], old["value"], old["kind"])
                else:
                    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, entry["path"], 0, winreg.KEY_SET_VALUE) as key:
                        winreg.DeleteValue(key, entry["name"])
            except OSError:
                log.warning("Could not roll back an EyeSync launch entry", exc_info=True)
        raise


def _digest(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _stop_existing(timeout=30.) -> None:
    """Request this distribution's supervisor to release its installed EXE."""
    import desktop
    from bundle_signals import LauncherExit

    name = "EyeSync.Brahmastra.Launcher"
    if not desktop.is_running(name):
        return
    if not LauncherExit.send():
        raise OSError("An older EyeSync background instance is running. Close that background instance and retry the installation; no process was force-closed.")
    deadline = time.monotonic() + timeout
    while desktop.is_running(name):
        if time.monotonic() >= deadline:
            raise OSError("EyeSync is still closing. Quit its background instance and retry the update.")
        time.sleep(.1)


def install(source, dest=DEFAULT_DEST, *, register=True, launch=True, open_site=True) -> Path:
    """Copy the executable atomically; preserve all existing data and calibration.

    `dest` is a directory, not an executable filename. Quiet verification should
    pass register=False, launch=False, open_site=False and a temporary destination.
    """
    source, dest = Path(source).resolve(), Path(dest).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"The source executable was not found: {source}")
    if register or launch:
        _stop_existing()
    dest.mkdir(parents=True, exist_ok=True)
    target = dest / EXE_NAME
    if not _same_path(source, target):
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(prefix="EyeSync-install-", suffix=".tmp", dir=dest, delete=False) as staged:
                temporary = Path(staged.name)
                with source.open("rb") as incoming:
                    shutil.copyfileobj(incoming, staged, length=1024 * 1024)
                staged.flush()
                os.fsync(staged.fileno())
            if _digest(temporary) != _digest(source):
                raise OSError("The copied executable failed its checksum check. Download EyeSync again.")
            try:
                os.replace(temporary, target)
                temporary = None
            except PermissionError as exc:
                raise OSError("EyeSync is still running or its file is locked. Quit EyeSync from the dashboard, close its standby process, then retry the update.") from exc
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    if register:
        register_install(target)
    if launch:
        # An installed onefile copy must own its own extracted runtime, rather
        # than depending on the downloaded installer's temporary files.
        subprocess.Popen([str(target), "--standby"], cwd=dest, creationflags=NO_WINDOW,
                         env={**os.environ, "PYINSTALLER_RESET_ENVIRONMENT": "1"})
    if open_site:
        webbrowser.open(SITE)
    return target


def _show_installer(source: Path, dest: Path, *, register, launch, open_site) -> int:
    import tkinter as tk
    from tkinter import ttk

    root = tk.Tk()
    root.title("Install EyeSync")
    root.geometry("600x310")
    root.minsize(520, 280)
    root.configure(bg="#0b1020")
    frame = ttk.Frame(root, padding=24)
    frame.pack(fill="both", expand=True)
    ttk.Label(frame, text="EyeSync", font=("Segoe UI", 24, "bold")).pack(anchor="w")
    ttk.Label(frame, text="Install the face and voice controller for your Windows account.",
              wraplength=540).pack(anchor="w", pady=(10, 8))
    ttk.Label(frame, text=str(dest), wraplength=540).pack(anchor="w")
    status = tk.StringVar(value="Your existing calibration and settings will be kept.")
    ttk.Label(frame, textvariable=status, wraplength=540).pack(anchor="w", pady=16)
    events = queue.Queue()
    busy, installed, exit_code = False, False, 0

    def close():
        if busy:
            status.set("Please wait for the copy to finish.")
            return
        root.destroy()

    def begin():
        nonlocal busy
        if installed:
            close()
            return
        busy = True
        action.configure(state="disabled")
        status.set("Installing EyeSync…")
        def worker():
            try:
                target = install(source, dest, register=register, launch=launch, open_site=open_site)
                events.put((True, target))
            except Exception as exc:
                log.exception("EyeSync installation failed")
                events.put((False, str(exc)))
        threading.Thread(target=worker, name="eyesync-install", daemon=True).start()

    def poll():
        nonlocal busy, installed, exit_code
        try:
            success, result = events.get_nowait()
        except queue.Empty:
            pass
        else:
            busy = False
            installed = success
            exit_code = 0 if success else 1
            status.set("EyeSync is installed. Use Connect on the dashboard to activate it." if success else str(result))
            action.configure(state="normal", text="Finish" if success else "Retry")
        if root.winfo_exists():
            root.after(100, poll)

    action = ttk.Button(frame, text="Update" if (dest / EXE_NAME).exists() else "Install", command=begin)
    action.pack(anchor="e")
    root.protocol("WM_DELETE_WINDOW", close)
    root.after(100, poll)
    root.mainloop()
    return exit_code


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    parser.add_argument("--no-register", action="store_true")
    parser.add_argument("--no-launch", action="store_true")
    parser.add_argument("--no-site", action="store_true")
    args = parser.parse_args(argv)
    source = Path(sys.executable)
    options = dict(register=not args.no_register, launch=not args.no_launch, open_site=not args.no_site)
    if args.quiet:
        try:
            install(source, args.dest, **options)
            return 0
        except Exception:
            log.exception("EyeSync installation failed")
            return 1
    return _show_installer(source, args.dest, **options)


if __name__ == "__main__":
    raise SystemExit(main())
