"""Fetch only the release's pretrained models, checking exact size and SHA256."""
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT=Path(__file__).resolve().parents[1]

def matches(path,item):
    if not path.exists() or path.stat().st_size!=item['size']: return False
    with path.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()==item['sha256']

def main():
    for item in json.loads((ROOT/'models/manifest.json').read_text()):
        path=ROOT/'models'/item['file']
        if not matches(path,item):
            temporary=path.with_suffix(path.suffix+'.download')
            with urllib.request.urlopen(item['url'],timeout=120) as source,temporary.open('wb') as target:
                while chunk:=source.read(1024*1024): target.write(chunk)
            if not matches(temporary,item): raise RuntimeError('Model verification failed: '+item['file'])
            temporary.replace(path)
        print('Verified '+item['file'],flush=True)

if __name__=='__main__': main()
