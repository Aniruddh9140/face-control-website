import json
import queue
import time
import unittest
from unittest.mock import Mock

import numpy as np
from websockets.sync.client import connect
from websockets.exceptions import ConnectionClosed

from control_engine import Controller
from dashboard_bridge import DashboardBridge, ORIGIN
from eyesync_app import App


class DashboardTests(unittest.TestCase):
    def setUp(self):
        self.controller = Controller((1920, 1080), mouse_factory=Mock)
        self.bridge = DashboardBridge(self.controller, ports=(0,), hosts=["127.0.0.1"]).start()
        self.assertTrue(self.bridge.ready.wait(3))
        self.assertTrue(self.bridge.bound_ports, self.bridge.errors)
        self.url = f"ws://127.0.0.1:{self.bridge.bound_ports[0]}/eye-tracker"

    def tearDown(self):
        self.bridge.close()
        self.assertFalse(self.bridge.thread.is_alive())

    def connection(self, suffix="", origin=ORIGIN):
        return connect(self.url + suffix, origin=origin, proxy=None, open_timeout=2, close_timeout=1)

    def receive(self, ws, predicate):
        end = time.monotonic() + 3
        while time.monotonic() < end:
            m = json.loads(ws.recv(timeout=2))
            if predicate(m):
                return m
        self.fail("Expected dashboard message was not received")

    def test_probe_identifies_new_copy_without_starting_camera(self):
        with self.connection("?probe=1") as ws:
            m = json.loads(ws.recv(timeout=2))
            self.assertEqual((m["type"], m["state"], m["implementation"]), ("eyesync", "running", "brahmastra"))
            with self.assertRaises(ConnectionClosed) as caught:
                ws.recv(timeout=2)
            self.assertEqual(caught.exception.rcvd.code, 4003)
        self.assertTrue(self.controller.events.empty())
        self.assertFalse(self.controller.state.read().running)

    def test_other_origins_and_missing_origin_are_refused(self):
        for origin in ("https://example.com", "https://aniruddh9140.github.io.evil.example", None):
            with self.connection(origin=origin) as ws:
                with self.assertRaises(ConnectionClosed) as caught:
                    ws.recv(timeout=2)
                self.assertEqual(caught.exception.rcvd.code, 4001)
        self.assertTrue(self.controller.events.empty())

    def test_connect_activates_and_streams_800_by_600_coordinates(self):
        self.controller.state.update(x=960, y=540, face=True, calibrated=True)
        with self.connection("?start=1") as ws:
            self.receive(ws, lambda m: m.get("type") == "status")
            self.assertEqual(self.controller.events.get(timeout=1), ("dashboard_activate", None))
            sample = self.receive(ws, lambda m: "x" in m)
            self.assertEqual((sample["x"], sample["y"], sample["face"]), (400, 300, True))

    def test_second_client_does_not_reset_pause_or_activate_again(self):
        with self.connection() as first:
            self.receive(first, lambda m: m.get("type") == "status")
            self.controller.events.get(timeout=1)
            self.controller.pause()
            with self.connection() as second:
                status = self.receive(second, lambda m: m.get("type") == "status")
                self.assertTrue(status["paused"])
                self.assertTrue(self.controller.events.empty())

    def test_commands_control_the_combined_engine(self):
        with self.connection() as ws:
            self.receive(ws, lambda m: m.get("type") == "status")
            for action, value, field, expected in [
                ("speed", 5, "speed", 5), ("mouse", True, "mouse_enabled", True),
                ("pause", True, "paused", True), ("pause", False, "paused", False),
                ("click_mode", "dwell", "click_mode", "dwell"),
                ("gestures", False, "gestures", False), ("mouse", False, "mouse_enabled", False),
            ]:
                ws.send(json.dumps({"action": action, "value": value}))
                self.receive(ws, lambda m: m.get("type") == "status" and m.get(field) == expected)
            ws.send(json.dumps({"action": "voice", "value": True}))
            self.assertEqual(self.controller.state.voice_requests.get(timeout=2), (True, "website"))

    def test_reconnect_during_grace_preserves_explicit_pause(self):
        with self.connection() as first:
            self.receive(first, lambda m: m.get("type") == "status")
            self.assertEqual(self.controller.events.get(timeout=1), ("dashboard_activate", None))
            first.send(json.dumps({"action": "pause", "value": True}))
            self.receive(first, lambda m: m.get("type") == "status" and m["paused"])
        # Wait for the server to observe disconnection before reconnecting.
        end = time.monotonic() + 1
        while self.bridge.last_client_left is None and time.monotonic() < end:
            time.sleep(.01)
        self.assertIsNotNone(self.bridge.last_client_left)
        with self.connection() as second:
            status = self.receive(second, lambda m: m.get("type") == "status")
            self.assertTrue(status["paused"])
            self.assertTrue(self.controller.events.empty(), "Reconnect must not request pointer activation")

    def test_reconnect_after_grace_activates_new_session(self):
        with self.connection() as first:
            self.receive(first, lambda m: m.get("type") == "status")
            self.assertEqual(self.controller.events.get(timeout=1), ("dashboard_activate", None))
        self.assertEqual(self.controller.events.get(timeout=3), ("dashboard_disconnect", None))
        with self.connection() as second:
            self.receive(second, lambda m: m.get("type") == "status")
            self.assertEqual(self.controller.events.get(timeout=1), ("dashboard_activate", None))

    def test_camera_preview_is_opt_in(self):
        self.controller.state.preview = np.zeros((240, 320, 3), np.uint8)
        with self.connection() as ws:
            self.receive(ws, lambda m: m.get("type") == "status")
            self.assertFalse(self.bridge.preview_clients)
            ws.send(json.dumps({"action": "preview", "value": True}))
            m = self.receive(ws, lambda m: m.get("type") == "preview")
            self.assertTrue(m["image"].startswith("data:image/jpeg;base64,"))
            ws.send(json.dumps({"action": "preview", "value": False}))
            self.receive(ws, lambda m: m.get("type") == "status")

    def test_status_never_sends_transcripts_or_private_voice_details(self):
        self.controller.state.update(voice_text="private dictation", voice_hint="private microphone device")
        wire = json.dumps(self.bridge.status())
        self.assertNotIn("private", wire)
        self.assertNotIn("voice_text", wire)
        self.assertNotIn("gaze", self.bridge.status())

    def test_malformed_commands_and_unsupported_gaze_are_explicit_errors(self):
        for message in ("garbage", "[]", '{"action":"speed","value":true}',
                        '{"action":"voice","value":"yes"}', '{"action":"gaze","value":true}',
                        '{"action":"calibrate","value":"gaze"}'):
            self.assertEqual(self.bridge.command(message)["type"], "error")

    def test_calibration_window_and_quit_are_dispatched_to_ui_thread(self):
        for action, value, event in [("calibrate", None, "dashboard_calibrate"),
                                      ("show_window", None, "dashboard_show"),
                                      ("quit", None, "dashboard_quit")]:
            self.bridge.command(json.dumps({"action": action, "value": value}))
            self.assertEqual(self.controller.events.get_nowait(), (event, value))

    def test_disconnection_requests_camera_and_microphone_release(self):
        with self.connection() as ws:
            self.receive(ws, lambda m: m.get("type") == "status")
            self.controller.events.get(timeout=1)
        self.assertEqual(self.controller.events.get(timeout=3), ("dashboard_disconnect", None))


class DashboardUiTests(unittest.TestCase):
    def app(self):
        app = App.__new__(App)
        app.dashboard_test = False
        app.dashboard_owned = False
        app.controller = Mock()
        app.controller.thread = None
        app.root = Mock()
        return app

    def test_connect_starts_new_engine_and_requests_pointer_enable(self):
        app = self.app()
        app.dashboard_event("dashboard_activate", None)
        app.controller.start.assert_called_once()
        app.controller.request_enable.assert_called_once()
        self.assertTrue(app.dashboard_owned)

    def test_disconnect_stops_dashboard_owned_camera(self):
        app = self.app()
        app.dashboard_owned = True
        app.dashboard_event("dashboard_disconnect", None)
        app.controller.stop_camera.assert_called_once()

    def test_probe_test_mode_never_opens_hardware(self):
        app = self.app()
        app.dashboard_test = True
        app.dashboard_event("dashboard_activate", None)
        app.controller.start.assert_not_called()
        app.controller.request_enable.assert_not_called()


if __name__ == "__main__":
    unittest.main()
