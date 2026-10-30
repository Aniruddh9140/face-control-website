"""Double-click entry point; Start EyeSync.bat selects the bundled environment."""
from pathlib import Path
import subprocess
import sys

folder = Path(__file__).resolve().parent
subprocess.Popen([str(folder / ".venv" / "Scripts" / "pythonw.exe"),
                  str(folder / "dashboard_launcher.py"), *sys.argv[1:]], cwd=folder)
