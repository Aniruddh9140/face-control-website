"""The live face-control-website protocol, served only on loopback interfaces."""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import threading
import time
from urllib.parse import parse_qs, urlsplit

import cv2
from websockets.asyncio.server import serve
from websockets.exceptions import ConnectionClosed

import voice_typing as vt
from version import VERSION

ORIGIN = "https://aniruddh9140.github.io"
log = logging.getLogger("eyesync.dashboard")


class DashboardBridge:
    STREAM_SEND_TIMEOUT = 1.0

    def __init__(self, controller, ports=(8765, 8080), hosts=None, test_mode=False):
        self.controller = controller
        self.ports = ports
        self.hosts = hosts or ["127.0.0.1", "::1"]
        self.test_mode = test_mode
        self.clients = set()
        self.preview_clients = set()
        self.stopping = threading.Event()
        self.ready = threading.Event()
        self.thread = None
        self.bound_ports = []
        self.errors = []
        self.closed_by_user = False
        self.last_client_left = None
        self._stream_tasks = {}
        self._stalled_clients = set()
        self._client_clicks = {}

    def status(self):
        s = self.controller.state.read()
        return {
            "type": "status", "version": VERSION, "implementation": "brahmastra",
            "python": "3.15", "mouse_enabled": s.mouse_requested, "paused": s.paused,
            "active": s.enabled and s.face and s.running and not s.paused,
            "face": s.face, "camera": s.running, "speed": s.speed,
            "pointer_mode": s.pointer_mode,
            "scrolling": s.scrolling,
            "click_mode": s.click_mode, "gestures": s.gestures,
            "calibrating": s.calib_step not in ("", "done"), "calibrated": s.calibrated_date,
            "calibration": {"step": s.calib_step, "text": s.calibration_text,
                            "progress": s.calibration_progress, "capture_progress": s.capture_progress,
                            "quality": s.calibration_quality},
            # Advice only ("" when there is none): the website shows it as a notice and lists the gestures itself.
            "hint": s.hint,
            "hardware_mouse": False,
            # Omit the optional gaze object so this site's unsupported gaze controls stay hidden.
            "voice": {"on": s.voice not in ("", "error"), "state": s.voice or "off",
                      "hint": "Check the EyeSync window for voice details." if s.voice_hint else ""},
        }

    def sample(self, previous_clicks):
        s = self.controller.state.read()
        w, h = self.controller.screen
        clicks = s.click_count + s.right_click_count
        return {"x": round(s.x / max(w, 1) * 800, 2),
                "y": round(s.y / max(h, 1) * 600, 2),
                "blink": s.eyes_closed or clicks != previous_clicks,
                "confidence": 1.0 if s.face and s.calibrated else 0.0, "face": s.face}, clicks

    def activate(self):
        self.closed_by_user = False
        self.controller.events.put(("dashboard_activate", None))

    def command(self, message):
        try:
            m = json.loads(message)
        except (ValueError, TypeError):
            return {"type": "error", "error": "Commands must be JSON text."}
        if isinstance(m, dict) and m.get("type") == "calibration":
            return None  # Website accuracy-test telemetry is not personal head calibration.
        if not isinstance(m, dict) or not isinstance(m.get("action"), str):
            return {"type": "error", "error": "Expected an action."}
        action, value = m["action"], m.get("value")
        c = self.controller
        if action == "status":
            pass
        elif action == "mouse" and type(value) is bool:
            if value:
                c.request_enable()
            else:
                c.pause()
                c.state.update(mouse_requested=False, paused=False)
        elif action == "pause" and type(value) is bool:
            if value:
                c.pause()
            elif c.state.read().mouse_requested:
                c.request_enable()
            else:
                c.state.update(paused=False)
        elif action == "voice" and (value is None or type(value) is bool):
            if not self.test_mode:
                vt.request(c.state, value, "website")
        elif action == "speed" and type(value) is int and 1 <= value <= 5:
            c.state.update(speed=value)
        elif action == "gestures" and type(value) is bool:
            c.state.update(gestures=value)
        elif action == "click_mode" and value in ("blink", "dwell"):
            c.state.update(click_mode=value)
        elif action == "recenter":
            c.request("recenter")
        elif action == "calibrate" and value in (None, "head", "cancel"):
            c.events.put(("dashboard_calibrate", value))
        elif action in ("gaze", "calibrate"):
            return {"type": "error", "error": "This EyeSync build uses head control. Eye-gaze jump is not included."}
        elif action == "show_window":
            c.events.put(("dashboard_show", None))
        elif action == "quit":
            self.closed_by_user = True
            c.events.put(("dashboard_quit", None))
        else:
            return {"type": "error", "error": "Unsupported action or invalid value."}
        return self.status()

    async def handler(self, ws):
        # Never accept arbitrary websites, LAN clients, or unauthenticated native apps.
        if ws.request.headers.get("Origin", "") != ORIGIN:
            await ws.close(4001, "Only the EyeSync website is allowed")
            return
        query = parse_qs(urlsplit(ws.request.path).query)
        if query.get("probe") == ["1"]:
            await ws.send(json.dumps({**self.status(), "type": "eyesync", "state": "running",
                                      "closed_by_user": False}))
            await ws.close(4003, "EyeSync is running")
            return
        # A transient reconnect continues the same session. Activating again
        # requests pointer enable in the UI and would undo an explicit pause.
        reconnecting = (self.last_client_left is not None
                        and time.monotonic() - self.last_client_left < 2)
        self.clients.add(ws)
        state = self.controller.state.read()
        self._client_clicks[ws] = state.click_count + state.right_click_count
        self.controller.state.update(dashboard_clients=len(self.clients))
        self.last_client_left = None
        if len(self.clients) == 1 and not reconnecting:
            self.activate()
        try:
            await ws.send(json.dumps(self.status()))
            async for message in ws:
                if not isinstance(message, str):
                    await ws.send(json.dumps({"type": "error", "error": "Use JSON text."}))
                    continue
                try:
                    m = json.loads(message)
                except ValueError:
                    m = None
                if isinstance(m, dict) and m.get("action") == "preview" and type(m.get("value")) is bool:
                    (self.preview_clients.add if m["value"] else self.preview_clients.discard)(ws)
                    reply = self.status()
                else:
                    reply = self.command(message)
                if reply is not None:
                    await ws.send(json.dumps(reply))
                if self.closed_by_user:
                    await ws.close(4000, "EyeSync closed; press Connect to restart")
                    return
        except ConnectionClosed:
            pass
        finally:
            self.clients.discard(ws)
            self.preview_clients.discard(ws)
            self._stalled_clients.discard(ws)
            self._client_clicks.pop(ws, None)
            self.controller.state.update(dashboard_clients=len(self.clients))
            if not self.clients:
                self.last_client_left = time.monotonic()

    async def _send_stream(self, client, packets, clicks):
        try:
            async with asyncio.timeout(self.STREAM_SEND_TIMEOUT):
                for index, packet in enumerate(packets):
                    await client.send(packet)
                    # The first packet is telemetry. Consume this client's
                    # click indication only after that packet was sent.
                    if index == 0 and client in self.clients:
                        self._client_clicks[client] = clicks
        except ConnectionClosed:
            return
        except TimeoutError:
            self._stalled_clients.add(client)
            log.warning("Disconnecting stalled dashboard consumer")
            try:
                async with asyncio.timeout(1):
                    await client.close(4004, "Dashboard is not receiving updates; reconnect")
            except (TimeoutError, ConnectionClosed):
                pass

    async def run(self):
        servers = []
        for port in self.ports:
            try:
                server = await serve(self.handler, self.hosts, port, max_size=4096,
                                     ping_interval=20, ping_timeout=20, close_timeout=1)
            except OSError:
                try:
                    server = await serve(self.handler, "127.0.0.1", port, max_size=4096, close_timeout=1)
                except OSError as exc:
                    self.errors.append(str(exc))
                    log.warning("Dashboard port %s is busy", port)
                    continue
            servers.append(server)
            self.bound_ports.append(server.sockets[0].getsockname()[1])
        self.ready.set()
        if not servers:
            self.controller.state.update(hint="Dashboard ports are busy. Close the older EyeSync and restart this copy.")
            return
        next_status = next_preview = 0
        try:
            while not self.stopping.is_set() and not self.controller.state.stop.is_set():
                now = time.monotonic()
                if self.closed_by_user:
                    for client in list(self.clients):
                        await client.close(4000, "EyeSync closed; press Connect to restart")
                if self.last_client_left is not None and now - self.last_client_left >= 2:
                    self.last_client_left = None
                    self.controller.events.put(("dashboard_disconnect", None))
                packets = []
                preview_packet = None
                if self.clients:
                    if now >= next_status:
                        next_status = now + 0.25
                        packets.append(json.dumps(self.status()))
                if self.preview_clients and now >= next_preview:
                    next_preview = now + 0.125
                    with self.controller.state.lock:
                        frame = self.controller.state.preview
                    if frame is not None:
                        ok, jpg = cv2.imencode(".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
                                               [cv2.IMWRITE_JPEG_QUALITY, 70])
                        if ok:
                            preview_packet = json.dumps({"type": "preview", "image":
                                "data:image/jpeg;base64," + base64.b64encode(jpg).decode("ascii")})
                for client, task in list(self._stream_tasks.items()):
                    if task.done():
                        # Retrieve errors so failed sends never become unobserved
                        # task exceptions. Connection closure is handled above.
                        if not task.cancelled() and task.exception() is not None:
                            log.error("Dashboard stream task failed", exc_info=task.exception())
                        del self._stream_tasks[client]
                for client in list(self.clients):
                    if client in self._stream_tasks or client in self._stalled_clients:
                        continue
                    # At most one batch per client; obsolete telemetry is dropped
                    # rather than queued behind a slow consumer.
                    sample, clicks = self.sample(self._client_clicks.get(client, 0))
                    batch = [json.dumps(sample)] + packets + ([preview_packet] if preview_packet is not None
                                       and client in self.preview_clients else [])
                    self._stream_tasks[client] = asyncio.create_task(self._send_stream(client, batch, clicks))
                await asyncio.sleep(1 / 30)
        finally:
            for server in servers:
                server.close()
            for task in self._stream_tasks.values():
                task.cancel()
            await asyncio.gather(*self._stream_tasks.values(), return_exceptions=True)
            self._stream_tasks.clear()
            self._client_clicks.clear()
            await asyncio.gather(*(s.wait_closed() for s in servers))

    def start(self):
        def worker():
            try:
                asyncio.run(self.run())
            except Exception:
                log.exception("Dashboard service failed")
            finally:
                self.ready.set()
        self.thread = threading.Thread(target=worker, name="dashboard", daemon=True)
        self.thread.start()
        return self

    def close(self):
        self.stopping.set()
        if self.thread:
            self.thread.join(3)
