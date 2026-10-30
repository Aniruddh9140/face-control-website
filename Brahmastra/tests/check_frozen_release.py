"""Real one-file EXE tests; isolate all storage, ports and instance names."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from websockets.sync.client import connect
from websockets.exceptions import ConnectionClosed
from bundle_signals import LauncherExit
from version import VERSION

ORIGIN='https://aniruddh9140.github.io'
PORT=8799

def ask(query):
    with connect(f'ws://127.0.0.1:{PORT}/?{query}=1',origin=ORIGIN,proxy=None,open_timeout=1,close_timeout=1) as ws:
        return json.loads(ws.recv(timeout=2))

def wait_for(state, timeout=60):
    end=time.monotonic()+timeout
    while time.monotonic()<end:
        try:
            message=ask('probe')
            if message.get('state')==state: return message
        except (OSError,TimeoutError,ConnectionClosed): pass
        time.sleep(.25)
    raise AssertionError('Frozen EXE never reached '+state)

def quit_app():
    with connect(f'ws://127.0.0.1:{PORT}',origin=ORIGIN,proxy=None,close_timeout=1) as ws:
        ws.send('{"action":"quit"}')
        try:
            while True: ws.recv(timeout=5)
        except ConnectionClosed as exc:
            assert exc.rcvd.code==4000

def main():
    exe=Path(sys.argv[1]).resolve()
    with tempfile.TemporaryDirectory(prefix='EyeSync-frozen-') as temporary:
        dest=Path(temporary)
        env={**os.environ,'EYESYNC_DATA_DIR':str(dest),'EYESYNC_LOG':str(dest/'app.log'),
             'EYESYNC_LAUNCHER_LOG':str(dest/'launcher.log'),'PYINSTALLER_RESET_ENVIRONMENT':'1'}
        # Verify installer copying, without registration, launching or browser.
        subprocess.run([str(exe),'--install','--quiet','--dest',str(dest/'installed'),
                        '--no-register','--no-launch','--no-site'],env=env,check=True,timeout=120)
        installed=dest/'installed/EyeSync.exe'
        assert hashlib.sha256(installed.read_bytes()).digest()==hashlib.sha256(exe.read_bytes()).digest()
        print('PASS: frozen self-install copies identical EXE without registry changes',flush=True)
        child=subprocess.Popen([str(installed),'--standby','--ws-ports',str(PORT),'--dashboard-test-mode'],
                              env=env,cwd=dest,creationflags=subprocess.CREATE_NO_WINDOW)
        try:
            wait_for('standby')
            for cycle in range(2):
                assert ask('start')['state']=='starting'
                message=wait_for('running')
                assert message['implementation']=='brahmastra' and message['version']==VERSION
                assert not message['camera'] and not message['active']
                quit_app()
                assert wait_for('standby')['closed_by_user']
                print(f'PASS: frozen dashboard start/quit/standby cycle {cycle+1}; hardware disabled',flush=True)
        finally:
            # Cooperative exit reaches the actual onefile child, not just its bootloader.
            original=LauncherExit.NAME
            LauncherExit.NAME=original+'.Test'
            try:
                LauncherExit.send()
                child.wait(timeout=30)
            finally:
                LauncherExit.NAME=original
            if child.returncode:
                raise AssertionError('Frozen supervisor exit code '+str(child.returncode))
        direct_env={k:v for k,v in env.items() if k not in ('EYESYNC_DATA_DIR','EYESYNC_LOG','EYESYNC_LAUNCHER_LOG')}
        subprocess.run([str(installed),'--engine','--dashboard-test-mode','--ws-ports','8798',
                        '--smoke-seconds','2'],env=direct_env,check=True,timeout=90)
        assert (installed.parent/'data/eyesync.log').is_file()
        print('PASS: installed EXE keeps writable data beside itself, outside the extraction directory',flush=True)

if __name__=='__main__': main()
