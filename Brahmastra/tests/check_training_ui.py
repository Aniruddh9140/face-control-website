"""Exercise every training overlay position without camera, mic or mouse I/O."""
from pathlib import Path
import sys
import tkinter as tk

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eyesync_app import App
from control_engine import POINTS
import native_input

native_input.make_dpi_aware()
root = tk.Tk()
app = App(root, smoke=True)
try:
    root.update()
    assert app.enable_button.cget("text") == "Train & enable pointer"
    for index, point in enumerate(POINTS):
        app.show_dot(index, point)
        app.controller.state.update(calibration_text="Hold steady — 50% captured.", capture_progress=.5, face=True)
        app.tick()
        root.update()
        assert app.canvas.itemcget("feedback", "text") == "Hold steady — 50% captured."
        assert abs(float(app.canvas.itemcget("capture", "extent"))) > 170
        app.controller.state.update(calibration_text="Face not visible. Face the camera in good light.", face=False)
        app.tick()
        root.update()
        assert app.canvas.itemcget("feedback", "fill") == "#ffbc78"
    app.hide_calibration()
    assert app.overlay is None
    print("PASS: nine training targets, live feedback, capture ring and overlay cleanup; hardware disabled")
finally:
    app.closing = True
    root.destroy()
