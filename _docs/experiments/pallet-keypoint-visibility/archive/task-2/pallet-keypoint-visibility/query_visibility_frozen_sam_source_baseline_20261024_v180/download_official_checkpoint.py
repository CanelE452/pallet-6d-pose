"""Download the one prospectively fixed official SAM checkpoint, no model loading."""
import datetime
import hashlib
import json
from pathlib import Path
import time
import requests

ROOT = Path(__file__).resolve().parent
URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
DEST = ROOT / "checkpoints" / "sam_vit_b_01ec64.pth"
DEST.parent.mkdir(exist_ok=True)
assert not DEST.exists(), "Do not overwrite a previous complete download"
PART = DEST.with_suffix(".pth.part")
assert not PART.exists(), "Inspect previous partial download before retrying"
start = time.monotonic()
sha = hashlib.sha256()
md5 = hashlib.md5()
size = 0
with requests.get(URL, stream=True, timeout=(30, 60)) as response:
    response.raise_for_status()
    assert response.url == URL
    expected = int(response.headers["Content-Length"])
    assert expected == 375042383, "HEAD-verified official length changed"
    assert response.headers.get("Content-Type") == "binary/octet-stream"
    with PART.open("xb") as output:
        for block in response.iter_content(1024 * 1024):
            if block:
                output.write(block)
                sha.update(block)
                md5.update(block)
                size += len(block)
    assert size == expected
    receipt = {
        "UTC": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "URL": URL,
        "HTTP": response.status_code,
        "headers": dict(response.headers),
        "bytes": size,
        "sha256": sha.hexdigest(),
        "md5": md5.hexdigest(),
        "official_published_full_sha_verified": False,
        "integrity_scope": "HTTPS official linked endpoint, exact HEAD length and computed local hashes; multipart ETag is not a published SHA256 or simple MD5.",
        "elapsed_seconds": time.monotonic() - start,
        "model_loaded": False,
    }
PART.rename(DEST)
(ROOT / "OFFICIAL_CHECKPOINT_DOWNLOAD_RECEIPT.json").write_text(
    json.dumps(receipt, indent=2) + "\n"
)
print(json.dumps(receipt, indent=2), flush=True)
