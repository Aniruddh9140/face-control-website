"""Windows mouse input from Face Controlled OS."""
import ctypes
import sys
from ctypes import wintypes

class _MouseInput(ctypes.Structure):
    _fields_ = [
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.c_size_t),
    ]

class _InputUnion(ctypes.Union):
    _fields_ = [("mi", _MouseInput)]

class _Input(ctypes.Structure):
    _anonymous_ = ("value",)
    _fields_ = [("type", wintypes.DWORD), ("value", _InputUnion)]

class LocalMouseController:
    """Inject pointer movement and button events directly into Windows."""

    INPUT_MOUSE = 0
    LEFT_DOWN, LEFT_UP = 0x0002, 0x0004
    RIGHT_DOWN, RIGHT_UP = 0x0008, 0x0010

    def __init__(self) -> None:
        if sys.platform != "win32":
            raise OSError("Direct local mouse control is currently supported on Windows.")
        self._user32 = ctypes.WinDLL("user32", use_last_error=True)
        self._user32.SetCursorPos.argtypes = (ctypes.c_int, ctypes.c_int)
        self._user32.SetCursorPos.restype = wintypes.BOOL
        self._user32.SendInput.argtypes = (
            wintypes.UINT, ctypes.POINTER(_Input), ctypes.c_int
        )
        self._user32.SendInput.restype = wintypes.UINT

    def move(self, x: int, y: int) -> None:
        if not self._user32.SetCursorPos(x, y):
            error = ctypes.get_last_error()
            raise ctypes.WinError(error) if error else OSError(
                f"Windows refused to move the cursor to ({x}, {y})."
            )

    def click(self, button: str) -> None:
        if button == "left":
            flags = (self.LEFT_DOWN, self.LEFT_UP)
        elif button == "right":
            flags = (self.RIGHT_DOWN, self.RIGHT_UP)
        else:
            raise ValueError(f"Unsupported mouse button: {button}")

        inputs = (_Input * len(flags))()
        for item, flag in zip(inputs, flags):
            item.type = self.INPUT_MOUSE
            item.mi.dwFlags = flag
        sent = self._user32.SendInput(
            len(inputs), inputs, ctypes.sizeof(_Input)
        )
        if sent != len(inputs):
            error = ctypes.get_last_error()
            # A partial injection may have sent the down event without its up.
            if sent:
                self._button_event(button, False)
            raise ctypes.WinError(error) if error else OSError(
                f"Windows accepted {sent} of {len(inputs)} mouse button events."
            )

    def _button_event(self, button: str, down: bool) -> None:
        if button == "left":
            flag = self.LEFT_DOWN if down else self.LEFT_UP
        elif button == "right":
            flag = self.RIGHT_DOWN if down else self.RIGHT_UP
        else:
            raise ValueError(f"Unsupported mouse button: {button}")
        event = _Input()
        event.type = self.INPUT_MOUSE
        event.mi.dwFlags = flag
        sent = self._user32.SendInput(
            1, ctypes.byref(event), ctypes.sizeof(_Input)
        )
        if sent != 1:
            error = ctypes.get_last_error()
            action = "press" if down else "release"
            raise ctypes.WinError(error) if error else OSError(
                f"Windows did not accept the mouse button {action}."
            )

    def button_down(self, button: str) -> None:
        self._button_event(button, True)

    def scroll(self, units: int) -> None:
        event = _Input()
        event.type = self.INPUT_MOUSE
        event.mi.dwFlags = 0x0800
        event.mi.mouseData = int(units) & 0xffffffff
        if self._user32.SendInput(1, ctypes.byref(event), ctypes.sizeof(_Input)) != 1:
            raise ctypes.WinError(ctypes.get_last_error())

    def button_up(self, button: str) -> None:
        self._button_event(button, False)

def make_dpi_aware() -> None:
    if sys.platform == "win32":
        ctypes.windll.user32.SetProcessDPIAware()

def hotkey_kill_pressed() -> bool:
    """Global Ctrl+Alt+E (Windows). Lets the user switch the real cursor off without
    needing to click anything, even if the pointer is misbehaving."""
    if sys.platform != "win32":
        return False
    import ctypes
    down = ctypes.windll.user32.GetAsyncKeyState
    return all(down(vk) & 0x8000 for vk in (0x11, 0x12, 0x45))
