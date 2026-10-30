"""Real supervisor -> UI -> standby lifecycle, with all hardware disabled."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from websockets.sync.client import connect
from websockets.exceptions import ConnectionClosed
from dashboard_bridge import ORIGIN


def ask(query):
    with connect("ws://127.0.0.1:8799/?" + query + "=1", origin=ORIGIN,
                 open_timeout=1, close_timeout=1, proxy=None) as ws:
        return json.loads(ws.recv(timeout=2))


def wait_for(state, timeout=18):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        try:
            m = ask("probe")
            if m.get("state") == state:
                return m
        except (OSError, TimeoutError, ConnectionClosed):
            pass
        time.sleep(0.2)
    raise AssertionError("Never reached " + state)


def close_app():
    with connect("ws://127.0.0.1:8799", origin=ORIGIN, proxy=None, close_timeout=1) as ws:
        ws.send('{"action":"quit"}')
        try:
            while True:
                ws.recv(timeout=3)
        except ConnectionClosed as exc:
            assert exc.rcvd.code == 4000, exc


def main():
    with tempfile.TemporaryDirectory() as folder:
        env = {**os.environ, "EYESYNC_LOG": str(Path(folder) / "app.log"),
               "EYESYNC_LAUNCHER_LOG": str(Path(folder) / "launcher.log")}
        child = subprocess.Popen([sys._base_executable, str(ROOT / "dashboard_launcher.py"),
                "--standby", "--ws-ports", "8799", "--dashboard-test-mode"],
                cwd=ROOT, env=env, creationflags=subprocess.CREATE_NO_WINDOW)
        try:
            m = wait_for("standby")
            assert not m["closed_by_user"]
            assert ask("start")["state"] == "starting"
            m = wait_for("running")
            assert m["implementation"] == "brahmastra"
            assert not m["camera"] and not m["active"]
            print("PASS: probe stays idle; dashboard start launches the combined UI")
            with connect("ws://127.0.0.1:8799", origin=ORIGIN, proxy=None, close_timeout=1) as ws:
                ws.send('{"action":"speed","value":4}')
                while True:
                    m = json.loads(ws.recv(timeout=3))
                    if m.get("type") == "status" and m.get("speed") == 4:
                        break
                assert not m["camera"]
            close_app()
            assert wait_for("standby")["closed_by_user"]
            print("PASS: quit closes with 4000 and restores camera-free standby")
            assert ask("start")["state"] == "starting"
            assert wait_for("running")["implementation"] == "brahmastra"
            close_app()
            assert wait_for("standby")["closed_by_user"]
            print("PASS: dashboard starts EyeSync again after quitting")
        finally:
            # Only terminate the exact test supervisor created above, after its UI closes.
            child.terminate()
            child.wait(timeout=5)


if __name__ == "__main__":
    main()
