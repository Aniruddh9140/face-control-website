"""Keep a lightweight EyeSync listener ready for the dashboard's Connect button."""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import subprocess
import sys
import time

import desktop
import site_access
import app_paths
from bundle_signals import LauncherExit, EngineExit

ROOT = app_paths.APP_DIR


class OpenSignal(desktop.OpenSignal):
    NAME = "Local\\EyeSync.Brahmastra.Open"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--standby", action="store_true")
    parser.add_argument("--url")
    parser.add_argument("--camera", type=int, default=None)
    parser.add_argument("--ws-ports", type=int, nargs="+", default=[8765, 8080])
    parser.add_argument("--dashboard-test-mode", action="store_true")
    args = parser.parse_args()
    if args.dashboard_test_mode:
        OpenSignal.NAME += ".Test"
    if not desktop.acquire_single_instance("EyeSync.Brahmastra.Launcher" + (".Test" if args.dashboard_test_mode else "")):
        OpenSignal.send()
        desktop.bring_window_to_front("EyeSync", "TkTopLevel")
        return 0
    data = app_paths.DATA_DIR / "data"
    data.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=os.environ.get("EYESYNC_LAUNCHER_LOG", str(data / "dashboard-launcher.log")), level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", encoding="utf-8")
    signal = OpenSignal()
    if args.dashboard_test_mode:
        LauncherExit.NAME += ".Test"
        EngineExit.NAME += ".Test"
    quit_signal = LauncherExit()
    shutdown = False
    waiting = args.standby and not args.url
    minimized = bool(args.url)
    after_quit = False
    while True:
        if waiting:
            signal.clear()
            def wake():
                nonlocal shutdown
                shutdown = shutdown or quit_signal.is_set()
                return shutdown or signal.is_set()
            how = site_access.wait_for_visit(ports=args.ws_ports,
                    key_path=data / "unused-dashboard-key.txt",
                    wake=wake, after_quit=after_quit)
            if shutdown:
                return 0
            minimized = how == "website"
        pythonw = ROOT / ".venv" / "Scripts" / "pythonw.exe"
        command = ([sys.executable, "--engine"] if app_paths.FROZEN else [str(pythonw), str(ROOT / "eyesync_app.py")]) + ["--ws-ports", *(str(p) for p in args.ws_ports)]
        if args.camera is not None:
            command += ["--camera", str(args.camera)]
        if minimized:
            command.append("--minimized")
        if args.dashboard_test_mode:
            command.append("--dashboard-test-mode")
        logging.info("Starting EyeSync from %s", ROOT)
        try:
            child = subprocess.Popen(command, cwd=app_paths.DATA_DIR, creationflags=subprocess.CREATE_NO_WINDOW)
        except OSError:
            logging.exception("Could not start EyeSync; waiting for dashboard activation")
            # A launch failure must not remove the listener needed to retry.
            # Require a fresh activation, rather than repeatedly spawning.
            waiting = True
            after_quit = False
            time.sleep(2)
            continue
        while True:
            try:
                code = child.wait(timeout=.5)
                break
            except subprocess.TimeoutExpired:
                if quit_signal.is_set():
                    shutdown = True
                    EngineExit.send()
        if shutdown:
            return 0
        after_quit = code == 0
        if code:
            logging.warning("EyeSync exited with code %s; waiting for dashboard activation", code)
            time.sleep(2)
        waiting = True


if __name__ == "__main__":
    raise SystemExit(main())
