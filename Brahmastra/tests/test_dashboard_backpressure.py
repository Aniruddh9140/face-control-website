"""Exercise dashboard streaming with stalled consumers and no hardware."""
import asyncio
import json
import unittest
from unittest.mock import AsyncMock, Mock, patch

from control_engine import Controller
import dashboard_bridge as db


class BackpressureTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.bridge = db.DashboardBridge(Controller((1920, 1080), mouse_factory=Mock), ports=(0,), test_mode=True)
        self.release = asyncio.Event()
        self.slow = Mock()
        self.slow.send = AsyncMock(side_effect=lambda packet: None)
        async def blocked(packet):
            await self.release.wait()
        self.slow.send.side_effect = blocked
        self.slow.close = AsyncMock()
        self.fast = Mock(send=AsyncMock(), close=AsyncMock())
        self.bridge.clients = {self.slow, self.fast}
        server = Mock()
        server.sockets = [Mock()]
        server.sockets[0].getsockname.return_value = ("127.0.0.1", 12345)
        server.wait_closed = AsyncMock()
        patcher = patch.object(db, "serve", AsyncMock(return_value=server))
        patcher.start()
        self.addCleanup(patcher.stop)
        self.task = None

    async def asyncTearDown(self):
        self.bridge.stopping.set()
        self.release.set()
        if self.task is not None:
            await asyncio.wait_for(self.task, 2)
        self.assertFalse(self.bridge._stream_tasks)

    async def test_stalled_client_does_not_freeze_healthy_telemetry(self):
        self.bridge.STREAM_SEND_TIMEOUT = 10
        self.task = asyncio.create_task(self.bridge.run())
        async def wait_for_progress():
            while True:
                samples = [json.loads(call.args[0]) for call in self.fast.send.call_args_list
                           if "x" in json.loads(call.args[0])]
                if len(samples) >= 3:
                    return
                await asyncio.sleep(.01)
        await asyncio.wait_for(wait_for_progress(), 2)
        self.slow.close.assert_not_awaited()
        self.assertFalse(self.release.is_set())
        self.assertEqual(self.slow.send.call_count, 1, "Never build a backlog for a stalled client")

    async def test_stalled_client_is_disconnected_after_send_deadline(self):
        self.bridge.STREAM_SEND_TIMEOUT = .05
        self.task = asyncio.create_task(self.bridge.run())
        await asyncio.sleep(.2)
        self.slow.close.assert_awaited_once()

    async def test_click_during_busy_send_survives_until_next_fresh_sample(self):
        self.bridge.STREAM_SEND_TIMEOUT = 10
        entered = asyncio.Event()
        delivered = []
        async def temporary_delay(packet):
            entered.set()
            await self.release.wait()
            delivered.append(json.loads(packet))
        self.slow.send.side_effect = temporary_delay
        self.task = asyncio.create_task(self.bridge.run())
        await asyncio.wait_for(entered.wait(), 2)
        self.bridge.controller.state.update(click_count=1)
        async def healthy_client_saw_click():
            while not any(json.loads(call.args[0]).get('blink')
                          for call in self.fast.send.call_args_list):
                await asyncio.sleep(.01)
        # Prove another client progressed after the click while this send blocks.
        await asyncio.wait_for(healthy_client_saw_click(), 2)
        self.assertFalse(self.release.is_set())
        self.slow.close.assert_not_awaited()
        self.release.set()
        async def fresh_samples_delivered():
            while len([packet for packet in delivered if 'x' in packet]) < 3:
                await asyncio.sleep(.01)
        await asyncio.wait_for(fresh_samples_delivered(), 2)
        samples = [packet for packet in delivered if "x" in packet]
        self.assertTrue(any(packet["blink"] for packet in samples))
        self.assertEqual(sum(packet["blink"] for packet in samples), 1)
        self.assertFalse(samples[-1]["blink"], "Delivered clicks must not replay forever")
