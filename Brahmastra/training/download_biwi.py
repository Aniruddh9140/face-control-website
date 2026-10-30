"""Download the public fast.ai RGB repack of BIWI; retain provenance locally."""
import hashlib
import json
from pathlib import Path
import tarfile
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data" / "public-training"
URL = "https://s3.amazonaws.com/fast-ai-imagelocal/biwi_head_pose.tgz"

def main():
    DEST.mkdir(parents=True, exist_ok=True)
    archive = DEST / "biwi_head_pose.tgz"
    if not archive.exists():
        temporary = archive.with_suffix(".part")
        with urllib.request.urlopen(URL, timeout=60) as response, temporary.open("wb") as output:
            total = int(response.headers["Content-Length"])
            size, last = 0, time.monotonic()
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
                size += len(chunk)
                if time.monotonic() - last > 10:
                    print(f"Downloaded {size / 1e6:.0f}/{total / 1e6:.0f} MB", flush=True)
                    last = time.monotonic()
        if size != total:
            raise RuntimeError("Incomplete dataset download")
        temporary.replace(archive)
    digest = hashlib.file_digest(archive.open("rb"), "sha256").hexdigest()
    print("Archive SHA256:", digest, flush=True)
    with tarfile.open(archive) as source:
        source.extractall(DEST, filter="data")
    manifest = {
        "dataset": "BIWI Kinect Head Pose, fast.ai RGB repack",
        "url": URL, "sha256": digest, "bytes": archive.stat().st_size,
        "source": "https://huggingface.co/datasets/ETHZurich/biwi_kinect_head_pose",
        "mirror_catalog": "https://github.com/fastai/fastai/blob/master/fastai/data/external.py",
        "terms": "Non-commercial use such as university research and education.",
        "citation": "Fanelli et al., Random Forests for Real Time 3D Face Analysis, IJCV 101(3), 437–458, 2013."
    }
    (DEST / "provenance.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Dataset extracted:", DEST, flush=True)

if __name__ == "__main__":
    main()
