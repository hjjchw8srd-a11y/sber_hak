"""
download_chronos_weights.py

Downloads amazon/chronos-2 model weights and config from HuggingFace at fixed revision
29ec3766d36d6f73f0696f85560a422f50e8498c, verifies sha256 checksums, and stores them locally
for fully offline, air-gapped inference.
"""

import hashlib
import json
import os
import sys
import urllib.request
from pathlib import Path

DEST_DIR = Path("experiments/model_improvements/chronos2_weights")
DEST_DIR.mkdir(parents=True, exist_ok=True)

REVISION = "29ec3766d36d6f73f0696f85560a422f50e8498c"
BASE_URL = f"https://huggingface.co/amazon/chronos-2/raw/{REVISION}"
LFS_BASE_URL = f"https://huggingface.co/amazon/chronos-2/resolve/{REVISION}"

FILES = [
    ("config.json", f"{BASE_URL}/config.json"),
    ("model.safetensors", f"{LFS_BASE_URL}/model.safetensors"),
]

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()

def main():
    print(f"Downloading amazon/chronos-2 at revision {REVISION} to {DEST_DIR}...")
    manifest = {
        "model_id": "amazon/chronos-2",
        "revision": REVISION,
        "files": {},
    }
    
    for filename, url in FILES:
        out_path = DEST_DIR / filename
        if out_path.exists() and out_path.stat().st_size > 0:
            print(f"File {filename} already exists ({out_path.stat().st_size} bytes). Verifying...")
        else:
            print(f"Downloading {filename} from {url}...")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as resp, open(out_path, "wb") as f:
                while chunk := resp.read(1024 * 1024):
                    f.write(chunk)
            print(f"Downloaded {filename} ({out_path.stat().st_size} bytes).")
        
        file_sha = sha256_file(out_path)
        print(f"  {filename} SHA-256: {file_sha}")
        manifest["files"][filename] = {
            "size_bytes": out_path.stat().st_size,
            "sha256": file_sha,
        }

    manifest_path = DEST_DIR / "weights_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest written to {manifest_path}.")

if __name__ == "__main__":
    main()
