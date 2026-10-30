"""Single-EXE dispatcher: installer, lightweight standby, AI engine and self-test."""
from __future__ import annotations
import json
import os
from pathlib import Path
import sys
import tempfile

def selftest(path):
    # Tests never use installed settings or register the EXE.
    import unittest
    with tempfile.TemporaryDirectory(prefix='EyeSync-selftest-') as temp:
        os.environ['EYESYNC_DATA_DIR'] = temp
        os.environ['EYESYNC_LOG'] = str(Path(temp)/'test.log')
        import app_paths
        sys.path.insert(0,str(app_paths.APP_DIR/'tests'))
        with Path(path).open('w',encoding='utf-8') as output:
            suite=unittest.defaultTestLoader.discover(str(app_paths.APP_DIR/'tests'),pattern='test_*.py')
            result=unittest.TextTestRunner(stream=output,verbosity=1).run(suite)
            output.write(json.dumps(dict(tests=result.testsRun,errors=len(result.errors),failures=len(result.failures))))
        return 0 if result.wasSuccessful() else 1

def main(argv=None):
    args=list(sys.argv[1:] if argv is None else argv)
    if args and args[0]=='--selftest':
        return selftest(args[1] if len(args)>1 else Path.cwd()/'selftest.txt')
    import app_paths
    if args and args[0]=='--install':
        import bundle_installer
        return bundle_installer.main(args[1:])
    if args and args[0]=='--version':
        from version import VERSION
        if len(args)>1:
            Path(args[1]).write_text(VERSION,encoding='utf-8')
        else:
            print(VERSION)
        return 0
    engine = args and args[0] in ('--engine','--ui-smoke','--dashboard-test-mode')
    if engine:
        if args[0]=='--engine':
            args.pop(0)
        sys.argv=[sys.argv[0],*args]
        import eyesync_app
        return eyesync_app.main()
    if app_paths.FROZEN and not args:
        import bundle_installer
        if not bundle_installer.is_installed_copy():
            return bundle_installer.main([])
    sys.argv=[sys.argv[0],*args]
    import dashboard_launcher
    return dashboard_launcher.main()

if __name__=='__main__':
    # Native libraries expect streams even for a windowed build.
    if sys.stdout is None:
        sys.stdout=open(os.devnull,'w')
    if sys.stderr is None:
        sys.stderr=open(os.devnull,'w')
    import multiprocessing
    multiprocessing.freeze_support()
    try:
        raise SystemExit(main())
    except Exception:
        import traceback
        report=Path(tempfile.gettempdir())/'EyeSync-startup-error.log'
        report.write_text(traceback.format_exc(),encoding='utf-8')
        if not any(a in sys.argv for a in ('--selftest','--dashboard-test-mode','--ui-smoke')):
            import ctypes
            ctypes.windll.user32.MessageBoxW(None,f'EyeSync could not start. Details: {report}','EyeSync',0x10)
        raise SystemExit(1)
