"""Self-install checks use temporary files and an in-memory HKCU, never Windows settings."""
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import bundle_installer as bi


class FakeKey:
    def __init__(self, path): self.path = path
    def __enter__(self): return self
    def __exit__(self, *args): pass


class FakeRegistry:
    HKEY_CURRENT_USER = 'HKCU'
    KEY_SET_VALUE = 2
    REG_SZ = 1
    def __init__(self):
        self.data = {}
        self.writes = []
        self.fail_write = None
    def OpenKey(self, hive, path, *args):
        assert hive == self.HKEY_CURRENT_USER
        if path not in {p for p, n in self.data}: raise FileNotFoundError(path)
        return FakeKey(path)
    def QueryValueEx(self, key, name):
        try: return self.data[(key.path, name)]
        except KeyError: raise FileNotFoundError(name)
    def CreateKeyEx(self, hive, path, reserved=0, access=0):
        assert hive == self.HKEY_CURRENT_USER
        return FakeKey(path)
    def SetValueEx(self, key, name, reserved, kind, value):
        if self.fail_write == (key.path, name):
            self.fail_write = None
            raise OSError('Fake registry failure')
        self.writes.append((key.path, name))
        self.data[(key.path, name)] = (value, kind)
    def DeleteValue(self, key, name):
        try: del self.data[(key.path, name)]
        except KeyError: raise FileNotFoundError(name)


class InstallerTests(unittest.TestCase):
    def setUp(self):
        self.stop_existing = patch.object(bi, '_stop_existing')
        self.stopped = self.stop_existing.start()
        self.addCleanup(self.stop_existing.stop)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / 'Downloads' / 'EyeSync.exe'
        self.source.parent.mkdir()
        self.source.write_bytes(b'MZ fake release')
        self.dest = self.root / 'Local App Data' / 'Programs' / 'EyeSync-Brahmastra'

    def install(self, **kwargs):
        options = dict(register=False, launch=False, open_site=False)
        options.update(kwargs)
        return bi.install(self.source, self.dest, **options)

    def test_copy_is_one_file_and_preserves_existing_data(self):
        data = self.dest / 'data'
        data.mkdir(parents=True)
        personal = data / 'pointer.npz'
        personal.write_bytes(b'personal calibration')
        old = self.dest / 'EyeSync.exe'
        old.write_bytes(b'old executable')
        target = self.install()
        self.assertEqual(target.read_bytes(), self.source.read_bytes())
        self.assertEqual(personal.read_bytes(), b'personal calibration')
        self.assertEqual(sorted(p.name for p in self.dest.iterdir()), ['EyeSync.exe', 'data'])

    def test_source_and_destination_same_file_is_supported(self):
        self.dest.mkdir(parents=True)
        target = self.dest / 'EyeSync.exe'
        target.write_bytes(b'current executable')
        with patch.object(bi.os, 'replace') as replace:
            self.assertEqual(bi.install(target, self.dest, register=False, launch=False, open_site=False), target)
        replace.assert_not_called()

    def test_locked_update_preserves_old_file_and_cleans_staging_file(self):
        self.dest.mkdir(parents=True)
        target = self.dest / 'EyeSync.exe'
        target.write_bytes(b'old executable')
        with patch.object(bi.os, 'replace', side_effect=PermissionError('Fake lock')):
            with self.assertRaisesRegex(OSError, 'still running'):
                self.install()
        self.assertEqual(target.read_bytes(), b'old executable')
        self.assertEqual([p.name for p in self.dest.iterdir()], ['EyeSync.exe'])

    def test_missing_source_makes_no_destination(self):
        with self.assertRaises(FileNotFoundError):
            bi.install(self.root / 'missing.exe', self.dest, register=False, launch=False, open_site=False)
        self.assertFalse(self.dest.exists())

    def test_checksum_failure_does_not_replace_existing_executable(self):
        self.dest.mkdir(parents=True)
        target = self.dest / 'EyeSync.exe'
        target.write_bytes(b'old')
        with patch.object(bi, '_digest', side_effect=['wrong', 'expected']):
            with self.assertRaisesRegex(OSError, 'checksum'):
                self.install()
        self.assertEqual(target.read_bytes(), b'old')
        self.assertEqual([p.name for p in self.dest.iterdir()], ['EyeSync.exe'])

    def test_registration_quotes_exe_and_backs_up_once(self):
        registry = FakeRegistry()
        registry.data[(bi.RUN, 'EyeSync')] = ('old source standby', registry.REG_SZ)
        with patch.dict(sys.modules, winreg=registry):
            target = self.install(register=True)
            self.assertEqual(registry.data[(bi.RUN, 'EyeSync')][0], f'"{target}" --standby')
            command = registry.data[(bi.PROTOCOL + r'\shell\open\command', '')][0]
            self.assertEqual(command, f'"{target}" --url "%1"')
            backup = self.dest / 'data' / 'dashboard-registration-backup.json'
            original_backup = backup.read_bytes()
            saved = json.loads(original_backup)
            self.assertEqual(saved[0]['previous']['value'], 'old source standby')
            self.install(register=True)
            self.assertEqual(backup.read_bytes(), original_backup)

    def test_partial_registry_error_restores_this_attempt(self):
        registry = FakeRegistry()
        registry.data[(bi.RUN, 'EyeSync')] = ('old standby', registry.REG_SZ)
        registry.fail_write = (bi.PROTOCOL, 'URL Protocol')
        before = registry.data.copy()
        with patch.dict(sys.modules, winreg=registry):
            with self.assertRaises(OSError):
                self.install(register=True)
        self.assertEqual(registry.data, before)

    def test_launch_is_hidden_and_uses_installed_executable(self):
        with patch.object(bi.subprocess, 'Popen') as launch, patch.object(bi.webbrowser, 'open') as browser:
            target = self.install(launch=True, open_site=True)
        args, options = launch.call_args
        self.assertEqual(args[0], [str(target), '--standby'])
        self.assertEqual(options['creationflags'], bi.NO_WINDOW)
        self.assertEqual(options['cwd'], self.dest)
        self.assertEqual(options['env']['PYINSTALLER_RESET_ENVIRONMENT'], '1')
        browser.assert_called_once_with(bi.SITE)

    def test_no_side_effect_flags_skip_registration_process_and_browser(self):
        with patch.object(bi, 'register_install') as register, patch.object(bi.subprocess, 'Popen') as launch, \
             patch.object(bi.webbrowser, 'open') as browser:
            self.install()
        register.assert_not_called()
        launch.assert_not_called()
        browser.assert_not_called()
        self.stopped.assert_not_called()

    def test_is_installed_checks_exact_managed_path(self):
        with patch.object(bi, 'DEFAULT_DEST', self.dest):
            self.assertTrue(bi.is_installed_copy(self.dest / 'EyeSync.exe'))
            self.assertFalse(bi.is_installed_copy(self.source))
            self.assertFalse(bi.is_installed_copy(self.dest / 'Another.exe'))

    def test_quiet_main_installs_only_to_requested_destination(self):
        with patch.object(bi.sys, 'executable', str(self.source)):
            code = bi.main(['--quiet', '--dest', str(self.dest), '--no-register', '--no-launch', '--no-site'])
        self.assertEqual(code, 0)
        self.assertEqual((self.dest / 'EyeSync.exe').read_bytes(), self.source.read_bytes())

    def test_gui_is_not_created_until_requested(self):
        with patch.object(bi, '_show_installer', return_value=0) as gui:
            self.assertEqual(bi.main(['--dest', str(self.dest), '--no-register', '--no-launch', '--no-site']), 0)
        gui.assert_called_once()


class ExistingSupervisorTests(unittest.TestCase):
    def test_no_running_supervisor_needs_no_signal(self):
        with patch('desktop.is_running', return_value=False), patch('bundle_signals.LauncherExit.send') as send:
            bi._stop_existing()
        send.assert_not_called()

    def test_old_supervisor_without_exit_signal_is_not_force_killed(self):
        with patch('desktop.is_running', return_value=True), patch('bundle_signals.LauncherExit.send', return_value=False), \
             patch.object(bi.subprocess, 'Popen') as process:
            with self.assertRaisesRegex(OSError, 'older EyeSync'):
                bi._stop_existing()
        process.assert_not_called()

    def test_supported_supervisor_exits_gracefully(self):
        with patch('desktop.is_running', side_effect=[True, True, False]), \
             patch('bundle_signals.LauncherExit.send', return_value=True) as send, patch.object(bi.time, 'sleep'):
            bi._stop_existing()
        send.assert_called_once()

    def test_timeout_reports_waiting_without_force_kill(self):
        with patch('desktop.is_running', return_value=True), patch('bundle_signals.LauncherExit.send', return_value=True), \
             patch.object(bi.time, 'monotonic', side_effect=[0., 31.]), patch.object(bi.time, 'sleep'):
            with self.assertRaisesRegex(OSError, 'still closing'):
                bi._stop_existing()


if __name__ == '__main__':
    unittest.main()
