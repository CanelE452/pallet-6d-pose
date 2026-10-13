"""Restore the pinned public MobileSAM dependency for the optional refiner."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import tempfile

from . import common as C


def verify_assets(destination, manifest):
    destination = Path(destination)
    for name, expected in manifest["runtime_files_sha256"].items():
        path = destination / name
        if not path.is_file() or C.sha(path) != expected:
            raise ValueError("Pinned MobileSAM asset missing or changed: " + str(path))
    return destination


def restore_assets(destination, manifest):
    destination = Path(destination).resolve()
    if destination.exists():
        return verify_assets(destination, manifest)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="mobilesam-download-", dir=destination.parent))
    try:
        repository = temporary / "repository"
        subprocess.run(["git", "clone", "--quiet", "--filter=blob:none", "--no-checkout",
                        manifest["repository"], str(repository)], check=True)
        subprocess.run(["git", "-C", str(repository), "checkout", "--quiet", manifest["commit"]], check=True)
        verify_assets(repository, manifest)
        repository.rename(destination)
    finally:
        shutil.rmtree(temporary)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path,
                        default=C.DOC / "foundation_mask_v1/vendor/MobileSAM")
    parser.add_argument("--manifest", type=Path, default=C.DOC / "MOBILESAM_ASSET_MANIFEST.json")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    manifest = C.read(args.manifest)
    destination = (verify_assets(args.destination, manifest) if args.check_only
                   else restore_assets(args.destination, manifest))
    print("PINNED_MOBILESAM_READY", destination, flush=True)


if __name__ == "__main__":
    main()
