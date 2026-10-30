"""Install or restore this copy's per-user Windows dashboard launch integration."""
import argparse
import json
from pathlib import Path
import sys
import winreg

ROOT = Path(__file__).resolve().parent
BACKUP = ROOT / "data" / "dashboard-registration-backup.json"
RUN = r"Software\Microsoft\Windows\CurrentVersion\Run"
PROTOCOL = r"Software\Classes\eyesync"
ENTRIES = [(RUN, "EyeSync"), (PROTOCOL, ""), (PROTOCOL, "URL Protocol"),
           (PROTOCOL + r"\DefaultIcon", ""), (PROTOCOL + r"\shell\open\command", "")]


def read_value(path, name):
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, path) as key:
            value, kind = winreg.QueryValueEx(key, name)
            return {"value": value, "kind": kind}
    except FileNotFoundError:
        return None


def install():
    if sys.version_info[:2] != (3, 15):
        raise SystemExit("Run this with Brahmastra's Python 3.15 environment.")
    pythonw = ROOT / ".venv" / "Scripts" / "pythonw.exe"
    if not pythonw.is_file():
        raise SystemExit("The local Python environment is missing.")
    if not BACKUP.exists():
        BACKUP.parent.mkdir(exist_ok=True)
        BACKUP.write_text(json.dumps([
            {"path": path, "name": name, "previous": read_value(path, name)}
            for path, name in ENTRIES], indent=2), encoding="utf-8")
    base = f'"{pythonw}" "{ROOT / "dashboard_launcher.py"}"'
    values = [base + " --standby", "URL:EyeSync", "",
              f'"{ROOT / "models" / "eyesync.ico"}"', base + ' --url "%1"']
    for (path, name), value in zip(ENTRIES, values):
        with winreg.CreateKeyEx(winreg.HKEY_CURRENT_USER, path, 0, winreg.KEY_SET_VALUE) as key:
            winreg.SetValueEx(key, name, 0, winreg.REG_SZ, value)
    print("EyeSync dashboard launch and login standby now use Desktop/Brahmastra.")
    print("Previous registration backed up in data/dashboard-registration-backup.json.")


def restore():
    entries = json.loads(BACKUP.read_text(encoding="utf-8"))
    for entry in entries:
        path, name, previous = entry["path"], entry["name"], entry["previous"]
        with winreg.CreateKeyEx(winreg.HKEY_CURRENT_USER, path, 0, winreg.KEY_SET_VALUE) as key:
            if previous is None:
                try:
                    winreg.DeleteValue(key, name)
                except FileNotFoundError:
                    pass
            else:
                winreg.SetValueEx(key, name, 0, previous["kind"], previous["value"])
    print("Previous EyeSync registration restored.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["install", "restore"])
    args = parser.parse_args()
    (install if args.action == "install" else restore)()
