"""Original EyeSync floating voice bar, kept separate from its main window."""
from __future__ import annotations
import math
import tkinter as tk
class OriginalVoiceOverlay:
    """
    The voice typing bar at the bottom of the screen while voice typing is
    on (live_inference.Snapshot.voice): what it's doing - listening, hearing
    you, typing - with the microphone level, the last phrase typed (or a
    problem) and a Stop button. It never takes the keyboard focus from the
    window being typed into (WS_EX_NOACTIVATE) and stays above other windows.
    """
    BG, FG, DIM, WARN = ('#11161c', '#f2f2f2', '#9aa4ae', '#f0aa1e')
    DOT = {'starting': '#7a8590', 'listening': '#28b95a', 'hearing': '#e13c3c', 'working': '#f0aa1e', 'error': '#7a8590'}
    TEXT = {'starting': 'Starting voice typing...', 'listening': 'Listening - say what to type', 'hearing': 'Hearing you...', 'working': 'Typing...', 'error': 'Voice typing'}
    TIP = 'Stop: close your eyes for 2.5 s, or say "stop listening"'

    def __init__(self, root: tk.Tk, on_stop):
        self.root, self.on_stop = (root, on_stop)
        self.top: tk.Toplevel | None = None
        self.canvas: tk.Canvas | None = None
        self.stop_btn: tk.Button | None = None
        self.drawn = None

    def update(self, snap) -> None:
        if not snap.voice:
            self._hide()
            return
        if self.top is None:
            self._show()
        countdown = math.ceil(snap.voice_left) if snap.voice_left > 0 else 0
        key = (snap.voice, snap.voice_text, snap.voice_hint, round(snap.voice_level, 1), countdown)
        if key != self.drawn:
            self.drawn = key
            self._draw(snap, countdown)

    def _show(self) -> None:
        zoom = max(1.0, float(self.root.tk.call('tk', 'scaling')) / (96 / 72))
        self.zoom = zoom
        w, h = (round(600 * zoom), round(64 * zoom))
        left, _, right, bottom = _work_area(self.root)
        top = tk.Toplevel(self.root, bg=self.BG)
        top.withdraw()
        top.overrideredirect(True)
        top.attributes('-topmost', True)
        top.geometry(f'{w}x{h}+{left + (right - left - w) // 2}+{bottom - h - round(14 * zoom)}')
        self.canvas = tk.Canvas(top, width=w, height=h, bg=self.BG, highlightthickness=0)
        self.canvas.pack(fill='both', expand=True)
        self.stop_btn = tk.Button(top, text='Stop', command=self.on_stop, font=('Segoe UI', 11), bg='#2a323b', fg=self.FG, activebackground='#3a444f', activeforeground=self.FG, relief='flat', padx=14, pady=4, takefocus=0)
        self.stop_btn.place(relx=1.0, rely=0.5, x=-round(12 * zoom), anchor='e')
        self.top, self.drawn = (top, None)
        show_without_focus(top)

    def _hide(self) -> None:
        if self.top is not None:
            self.top.destroy()
        self.top = self.canvas = self.stop_btn = self.drawn = None

    def _draw(self, snap, countdown: int) -> None:
        c, z = (self.canvas, self.zoom)
        c.delete('all')
        h = int(c.cget('height'))
        cx, cy, r = (round(32 * z), h // 2, round(18 * z))
        c.create_oval(cx - r, cy - r, cx + r, cy + r, fill=self.DOT.get(snap.voice, '#7a8590'), outline='')
        mw, mh = (round(5 * z), round(8 * z))
        c.create_rectangle(cx - mw, cy - mh - round(2 * z), cx + mw, cy + round(3 * z), fill='white', outline='')
        c.create_arc(cx - round(9 * z), cy - round(8 * z), cx + round(9 * z), cy + round(8 * z), start=200, extent=140, style='arc', outline='white', width=max(1, round(2 * z)))
        c.create_line(cx, cy + round(8 * z), cx, cy + round(12 * z), fill='white', width=max(1, round(2 * z)))
        x = round(64 * z)
        c.create_text(x, round(h * 0.34), text=self.TEXT.get(snap.voice, ''), anchor='w', fill=self.FG, font=('Segoe UI', 12, 'bold'))
        if snap.voice_hint:
            line, color = (snap.voice_hint, self.WARN)
        elif countdown:
            line, color = (f'Stopping in {countdown} s - keep talking to go on', self.WARN)
        elif snap.voice_text:
            text = snap.voice_text if len(snap.voice_text) <= 60 else '...' + snap.voice_text[-57:]
            line, color = (f'“{text}”', self.DIM)
        else:
            line, color = (self.TIP, self.DIM)
        room = int(c.cget('width')) - x - round((16 if snap.voice == 'error' else 110) * z)
        c.create_text(x, round(h * 0.72), text=line, anchor='w', fill=color, font=('Segoe UI', 10), width=room)
        if snap.voice in ('listening', 'hearing'):
            bar = round(120 * z * min(1.0, snap.voice_level))
            c.create_rectangle(x + round(330 * z), round(h * 0.3), x + round(450 * z), round(h * 0.38), outline='#2f3943', fill='#1b2128')
            if bar:
                c.create_rectangle(x + round(330 * z), round(h * 0.3), x + round(330 * z) + bar, round(h * 0.38), outline='', fill=self.DOT['listening'])
        if snap.voice == 'error':
            self.stop_btn.place_forget()

def show_without_focus(top: tk.Toplevel) -> None:
    """
    Show a borderless Tk window WITHOUT taking the keyboard focus (and never
    take it when clicked): voice typing types into the window that has it.
    """
    top.update_idletasks()
    try:
        import ctypes
        from ctypes import wintypes
        user32 = ctypes.windll.user32
        hwnd = wintypes.HWND(int(top.wm_frame(), 16))
        GWL_EXSTYLE, WS_EX_TOOLWINDOW, WS_EX_TOPMOST, WS_EX_NOACTIVATE = (-20, 128, 8, 134217728)
        user32.GetWindowLongW.argtypes = (wintypes.HWND, ctypes.c_int)
        user32.SetWindowLongW.argtypes = (wintypes.HWND, ctypes.c_int, wintypes.LONG)
        style = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
        user32.SetWindowLongW(hwnd, GWL_EXSTYLE, style | WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW | WS_EX_TOPMOST)
    except (AttributeError, OSError, ValueError, tk.TclError):
        pass
    top.deiconify()

def _work_area(root: tk.Tk) -> tuple[int, int, int, int]:
    """The desktop minus the taskbar, in pixels (whole screen if unknown)."""
    try:
        import ctypes
        import ctypes.wintypes
        r = ctypes.wintypes.RECT()
        if ctypes.windll.user32.SystemParametersInfoW(48, 0, ctypes.byref(r), 0):
            return (r.left, r.top, r.right, r.bottom)
    except (AttributeError, OSError):
        pass
    return (0, 0, root.winfo_screenwidth(), root.winfo_screenheight())


class VoiceOverlay(OriginalVoiceOverlay):
    """Same no-focus bar, with separate dictation and command-listening labels."""
    def update(self, snap):
        command = getattr(snap, 'voice_mode', 'dictation') == 'command'
        if command != getattr(self, '_command_mode', None):
            self.drawn = None
        self._command_mode = command
        self.TEXT = dict(OriginalVoiceOverlay.TEXT)
        self.TEXT['working'] = 'Recognizing command...' if command else 'Recognizing speech...'
        if command:
            self.TEXT.update(starting='Starting voice commands...', listening='Listening for a keyboard command',
                             hearing='Hearing your command...')
            self.TIP = 'Say a listed keyboard command, or "stop listening"'
        else:
            self.TIP = OriginalVoiceOverlay.TIP
        super().update(snap)

    def _draw(self, snap, countdown):
        super()._draw(snap, countdown)
        # The original bar hides Stop on errors. Restore it on retry even when
        # the same top-level window is reused during the error display period.
        if snap.voice != 'error':
            self.stop_btn.place(relx=1., rely=.5, x=-round(12*self.zoom), anchor='e')
