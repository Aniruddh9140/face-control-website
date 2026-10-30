"""Launcher recovery without real processes, Windows signals or hardware."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import dashboard_launcher as launcher


class LauncherRecoveryTests(unittest.TestCase):
    def test_failed_spawn_returns_to_standby_and_later_connect_can_retry(self):
        quit_signal = Mock()
        quit_signal.is_set.return_value = False
        visits = []

        def visit(**kwargs):
            visits.append(kwargs)
            if len(visits) == 3:
                quit_signal.is_set.return_value = True
                self.assertTrue(kwargs['wake']())
                return 'shortcut'
            return 'website'

        child = Mock()
        child.wait.return_value = 0
        with tempfile.TemporaryDirectory() as folder, \
             patch.object(launcher.sys, 'argv', ['launcher', '--standby']), \
             patch.object(launcher.app_paths, 'DATA_DIR', Path(folder)), \
             patch.object(launcher.desktop, 'acquire_single_instance', return_value=True), \
             patch.object(launcher, 'OpenSignal'), \
             patch.object(launcher, 'LauncherExit', return_value=quit_signal), \
             patch.object(launcher.site_access, 'wait_for_visit', side_effect=visit), \
             patch.object(launcher.logging, 'basicConfig'), \
             patch.object(launcher.time, 'sleep') as sleep, \
             patch.object(launcher.subprocess, 'Popen', side_effect=[OSError('temporary launch failure'), child]) as spawn, \
             self.assertLogs(level='ERROR') as logs:
            self.assertEqual(launcher.main(), 0)

        self.assertEqual(spawn.call_count, 2)
        child.wait.assert_called_once_with(timeout=.5)
        sleep.assert_called_once_with(2)
        self.assertEqual([v['after_quit'] for v in visits], [False, False, True])
        self.assertIn('temporary launch failure', '\n'.join(logs.output))


if __name__ == '__main__':
    unittest.main()
