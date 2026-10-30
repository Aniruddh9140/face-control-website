"""Per-user graceful shutdown signals for updating this EyeSync distribution."""
import desktop

class LauncherExit(desktop.OpenSignal):
    NAME = r"Local\EyeSync.Brahmastra.LauncherExit"

class EngineExit(desktop.OpenSignal):
    NAME = r"Local\EyeSync.Brahmastra.EngineExit"
