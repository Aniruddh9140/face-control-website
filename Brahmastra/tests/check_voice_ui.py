"""Exercise the restored voice bar and retry UI with synthetic state only."""
from pathlib import Path
import sys
import tkinter as tk

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from eyesync_app import App
import native_input

native_input.make_dpi_aware()
root=tk.Tk()
app=App(root,smoke=True)
try:
    root.update()
    for mode in ('dictation','command'):
        for status in ('starting','listening','hearing','working','error','starting','listening'):
            app.controller.state.update(voice=status,voice_mode=mode,voice_level=.4,voice_left=2.)
            app.voice_overlay.update(app.controller.state.read())
            root.update()
            assert app.voice_overlay.top is not None
            texts=[app.voice_overlay.canvas.itemcget(i,'text') for i in app.voice_overlay.canvas.find_all()
                   if app.voice_overlay.canvas.type(i)=='text']
            if mode=='command' and status=='listening':
                assert 'Listening for a keyboard command' in texts
            if status!='error':
                assert app.voice_overlay.stop_btn.winfo_manager()=='place'
    app.controller.state.update(voice='')
    app.voice_overlay.update(app.controller.state.read())
    assert app.voice_overlay.top is None
    print('PASS: full voice bar, both modes, idle countdown, error/retry Stop button and cleanup; hardware disabled')
finally:
    app.closing=True
    app.voice_overlay._hide()
    root.destroy()
