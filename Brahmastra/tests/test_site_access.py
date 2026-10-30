"""Untrusted standby handshake parsing; no listener or hardware needed."""
import unittest
from unittest.mock import Mock, patch

import site_access as sa


class StandbyRequestTests(unittest.TestCase):
    def test_busy_ports_keep_shortcut_wake_available_without_empty_select(self):
        selector = Mock()
        selector.select.side_effect = OSError(10022, "empty Windows selector")
        wake = Mock(side_effect=[False, True])
        with patch.object(sa.selectors, "DefaultSelector", return_value=selector), \
             patch.object(sa, "_listen", return_value=[]), \
             patch.object(sa.time, "sleep") as sleep:
            self.assertEqual(sa.wait_for_visit(wake=wake), "shortcut")
        selector.select.assert_not_called()
        sleep.assert_called_once_with(.5)
        selector.close.assert_called_once()

    def test_busy_ports_are_reclaimed_after_retry_delay(self):
        selector, sock = Mock(), Mock()
        selector.select.return_value = []
        with patch.object(sa.selectors, "DefaultSelector", return_value=selector), \
             patch.object(sa, "_listen", side_effect=[[], [], [sock], []]) as listen, \
             patch.object(sa.time, "monotonic", side_effect=[0., 5., 5.5]), \
             patch.object(sa.time, "sleep") as sleep:
            self.assertEqual(sa.wait_for_visit(wake=Mock(side_effect=[False, False, True])), "shortcut")
        self.assertEqual(listen.call_count, 4)
        selector.register.assert_called_once_with(sock, sa.selectors.EVENT_READ)
        selector.select.assert_called_once_with(timeout=.5)
        selector.unregister.assert_called_once_with(sock)
        sock.close.assert_called_once()
        sleep.assert_called_once_with(.5)

    def test_only_explicit_full_origin_is_allowed(self):
        site = "https://aniruddh9140.github.io"
        self.assertTrue(sa.is_website_origin(site, sa.DEFAULT_SITE_ORIGINS))
        for origin in (
            "http://aniruddh9140.github.io",
            "http://aniruddh9140.github.io:1234",
            site + ":1234", site + ":443", site + "/",
            site + "/dashboard", site + "?start=1", site + "#dashboard",
            "https://user@aniruddh9140.github.io",
            site + ".evil.example", "https://evil.example",
            "//aniruddh9140.github.io", "aniruddh9140.github.io", "null", "",
        ):
            with self.subTest(origin=origin):
                self.assertFalse(sa.is_website_origin(origin, sa.DEFAULT_SITE_ORIGINS))

    def test_custom_origin_must_be_explicitly_allowed(self):
        origin = "https://dashboard.example:8443"
        self.assertTrue(sa.is_website_origin(origin, [origin]))
        self.assertFalse(sa.is_website_origin("https://dashboard.example", [origin]))
        self.assertFalse(sa.is_website_origin(origin, ["dashboard.example"]))

    def test_malformed_origin_is_rejected_without_stopping_listener(self):
        for origin in ("https://[", "http://[::1", "https://[invalid]/"):
            with self.subTest(origin=origin):
                self.assertFalse(sa.is_website_origin(origin, sa.DEFAULT_SITE_ORIGINS))

    def test_malformed_request_target_is_rejected(self):
        conn = Mock()
        conn.recv.return_value = b"GET http://[ HTTP/1.1\r\nOrigin: https://aniruddh9140.github.io\r\n\r\n"
        self.assertEqual(sa._parse_request(conn), ({}, {}))

    def test_valid_start_request_still_parses(self):
        conn = Mock()
        conn.recv.return_value = b"GET /eye-tracker?start=1 HTTP/1.1\r\nOrigin: https://aniruddh9140.github.io\r\n\r\n"
        headers, query = sa._parse_request(conn)
        self.assertTrue(sa.is_website_origin(headers["origin"], sa.DEFAULT_SITE_ORIGINS))
        self.assertEqual(query, {"start": ["1"]})
