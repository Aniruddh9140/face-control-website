"""Build the complete Windows x64 Python 3.15 EyeSync release into one EXE."""
from __future__ import annotations
import argparse
from datetime import date
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile
import time

from version import VERSION
ROOT=Path(__file__).resolve().parent

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'release')
    args=parser.parse_args()
    if sys.platform!='win32' or sys.version_info[:2]!=(3,15):
        raise SystemExit('Build with Windows x64 Python 3.15.')
    output=args.output.resolve()
    output.mkdir(parents=True,exist_ok=True)
    # Fresh scratch avoids OneDrive/antivirus locks on a previous run's cache.
    build=ROOT/'build'/'single-exe'/str(time.time_ns())
    build.mkdir(parents=True,exist_ok=True)
    parts=tuple(map(int,VERSION.split('.')))+(0,)
    version=build/'version.txt'
    version.write_text(f"VSVersionInfo(ffi=FixedFileInfo(filevers={parts}, prodvers={parts}, mask=0x3f, flags=0, OS=0x40004, fileType=1, subtype=0, date=(0,0)), kids=[StringFileInfo([StringTable('040904B0', [StringStruct('ProductName', 'EyeSync'), StringStruct('FileDescription', 'EyeSync face and voice control'), StringStruct('FileVersion', '{VERSION}'), StringStruct('ProductVersion', '{VERSION}')])]), VarFileInfo([VarStruct('Translation', [1033,1200])])])",encoding='utf-8')
    # Explicit allowlist: no settings, logs, recordings or training dataset.
    data=[('README.md','.'),('THIRD_PARTY_NOTICES.md','.'),('release.md','.'),
          ('voice_typing.py','.'),('docs/SOURCE_MANIFEST.json','docs')]
    data += [(str(p.relative_to(ROOT)),'models') for p in (ROOT/'models').iterdir() if p.suffix in ('.bin','.task','.npz','.ico')]
    data += [(str(p.relative_to(ROOT)),'tests') for p in (ROOT/'tests').glob('test_*.py')]
    data += [(str(p.relative_to(ROOT)),'licenses') for p in (ROOT/'licenses').glob('*.txt')]
    for src,_ in data:
        if not (ROOT/src).is_file(): raise FileNotFoundError(src)
    import _pywhispercpp
    dlls=sorted(Path(_pywhispercpp.__file__).parent.glob('*.dll'))
    hidden=['_pywhispercpp','sounddevice','numpy.testing','websockets.sync.client',
            'websockets.asyncio.client','unittest.mock','bundle_installer','eyesync_app','dashboard_launcher']
    command=[sys.executable,'-m','PyInstaller','--noconfirm','--clean','--onefile','--windowed',
             '--name','EyeSync','--icon',str(ROOT/'models/eyesync.ico'),
             '--version-file',str(version),'--distpath',str(output),'--workpath',str(build/'work'),
             '--specpath',str(build),'--paths',str(ROOT),'--paths',str(ROOT/'tests'),
             '--collect-all','mediapipe','--exclude-module','matplotlib','--exclude-module','pywhispercpp']
    for distribution in ('pywhispercpp','opencv-contrib-python','mediapipe','numpy','pillow','sounddevice','websockets','cffi'):
        command+=['--copy-metadata',distribution]
    python_license=Path(sys.base_prefix)/'LICENSE.txt'
    if python_license.is_file(): command+=['--add-data',f'{python_license};licenses/Python']
    for module in hidden: command+=['--hidden-import',module]
    for src,dest in data: command+=['--add-data',f'{ROOT/src};{dest}']
    for dll in dlls: command+=['--add-binary',f'{dll};.']
    command.append(str(ROOT/'package_main.py'))
    subprocess.run(command,cwd=ROOT,check=True)
    exe=output/'EyeSync.exe'
    with exe.open('rb') as file: digest=hashlib.file_digest(file,'sha256').hexdigest()
    info=dict(version=VERSION,file=exe.name,size=exe.stat().st_size,sha256=digest,date=date.today().isoformat())
    (output/'latest.json').write_text(json.dumps(info,indent=2)+'\n',encoding='utf-8')
    (output/'SHA256SUMS.txt').write_text(f'{digest}  EyeSync.exe\n',encoding='utf-8')
    shutil.copy2(ROOT/'release.md',output/'release.md')
    with (output/'release.md').open('a',encoding='utf-8') as notes:
        notes.write(f'\n## Generated build information\n\nVersion: {VERSION}\n\nFile: `EyeSync.exe` ({exe.stat().st_size:,} bytes)\n\nSHA-256: `{digest}`\n')
    # A small source upload, without .venv, recordings, settings or large models.
    sources=list(ROOT.glob('*.py'))+list(ROOT.glob('*.pyw'))+list(ROOT.glob('*.bat'))
    sources += [ROOT/name for name in ('README.md','release.md','THIRD_PARTY_NOTICES.md',
                'requirements.txt','requirements-python315.txt','requirements-build.txt','.gitignore')]
    sources += list((ROOT/'.github').rglob('*.yml'))
    sources += list((ROOT/'tests').glob('*.py'))+list((ROOT/'training').glob('*.py'))
    sources += [p for p in (ROOT/'models').iterdir() if p.suffix in ('.npz','.ico','.json')]
    sources += [p for p in (ROOT/'docs').iterdir() if p.suffix in ('.md','.json')]
    sources += list((ROOT/'licenses').glob('*.txt'))
    with zipfile.ZipFile(output/'EyeSync-source.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(set(sources)):
            archive.write(path,path.relative_to(ROOT).as_posix())
    sums=[]
    for name in ('EyeSync.exe','EyeSync-source.zip','latest.json','release.md'):
        with (output/name).open('rb') as file:
            sums.append(f"{hashlib.file_digest(file,'sha256').hexdigest()}  {name}\n")
    (output/'SHA256SUMS.txt').write_text(''.join(sums),encoding='utf-8')
    print(json.dumps(info,indent=2),flush=True)

if __name__=='__main__': main()
