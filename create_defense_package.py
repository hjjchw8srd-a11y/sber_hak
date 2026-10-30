#!/usr/bin/env python3
"""
create_defense_package.py

Creates a self-contained distribution zip package for the defense version:
defense_package_candidate_lgbm52.zip
"""

from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
PACKAGE_NAME = "defense_package_candidate_lgbm52.zip"
OUTPUT_ZIP = PROJECT_ROOT / PACKAGE_NAME

FILES_AND_DIRS = [
    "forecast-container/model",
    "forecast-container/app",
    "forecast-container/tests",
    "forecast-container/data",
    "forecast-container/Dockerfile",
    "forecast-container/.dockerignore",
    "forecast-container/requirements.txt",
    "forecast-container/README.md",
    "forecast-container/test_api.sh",
    "export_model_bundle.py",
    "audit_candidate_match.py",
    "DEFENSE_VERSION.md",
    "API_ACCEPTANCE_REPORT.md",
    "MODEL_ACTION_LOG.md",
    "submissions/candidate_lgbm52.csv",
    "submissions/candidate_lgbm52_manifest.json",
    "experiments/reviews/fold_0_real_180d_ablation.json",
    "experiments/reviews/run_real_fold0_ablation.py",
]


def make_package() -> Path:
    print(f"Creating defense distribution package: {OUTPUT_ZIP.name}...")
    file_count = 0
    total_uncompressed_bytes = 0

    with zipfile.ZipFile(OUTPUT_ZIP, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for item_rel in FILES_AND_DIRS:
            item_path = PROJECT_ROOT / item_rel
            if not item_path.exists():
                print(f"WARNING: File/dir not found: {item_rel}")
                continue

            if item_path.is_file():
                zf.write(item_path, arcname=item_rel)
                file_count += 1
                total_uncompressed_bytes += item_path.stat().st_size
            else:
                for sub in sorted(item_path.rglob("*")):
                    if sub.is_file() and "__pycache__" not in sub.parts and not sub.name.endswith(".pyc"):
                        rel_path = sub.relative_to(PROJECT_ROOT).as_posix()
                        zf.write(sub, arcname=rel_path)
                        file_count += 1
                        total_uncompressed_bytes += sub.stat().st_size

    archive_size_mb = OUTPUT_ZIP.stat().st_size / (1024 * 1024)
    raw_size_mb = total_uncompressed_bytes / (1024 * 1024)

    # Compute sha256 of zip
    h = hashlib.sha256()
    with open(OUTPUT_ZIP, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    zip_sha256 = h.hexdigest()

    print(f"Package created successfully!")
    print(f"  Files packed:         {file_count}")
    print(f"  Uncompressed size:    {raw_size_mb:.2f} MB")
    print(f"  Compressed zip size:  {archive_size_mb:.2f} MB")
    print(f"  SHA-256:              {zip_sha256}")

    return OUTPUT_ZIP


if __name__ == "__main__":
    make_package()
