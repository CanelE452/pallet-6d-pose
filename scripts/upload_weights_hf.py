"""Upload selected weights to Hugging Face Hub.

Uploads the 3 key checkpoints + their header.txt files to
`CanelE452/pallet-pose-dope-weights`.
"""
import os
import sys
import time
from huggingface_hub import HfApi

REPO_ID = "CanelE452/pallet-pose-dope-weights"
TOKEN = os.environ.get("HF_TOKEN")
if not TOKEN:
    print("ERROR: set HF_TOKEN env var")
    sys.exit(1)

FILES = [
    ("weights/mixed_v8/final_net_epoch_0060.pth", "mixed_v8/final_net_epoch_0060.pth"),
    ("weights/mixed_v8/header.txt", "mixed_v8/header.txt"),
    ("weights/v8_ablation_C_coord_edge/final_net_epoch_0065.pth", "v8_ablation_C_coord_edge/final_net_epoch_0065.pth"),
    ("weights/v8_ablation_C_coord_edge/header.txt", "v8_ablation_C_coord_edge/header.txt"),
    ("weights/misc/f5_noapril_ransac_loo_realonly/final_net_epoch_0096.pth", "f5_noapril_ransac_loo_realonly/final_net_epoch_0096.pth"),
    ("weights/misc/f5_noapril_ransac_loo_realonly/header.txt", "f5_noapril_ransac_loo_realonly/header.txt"),
]

api = HfApi(token=TOKEN)
print(f"[start] uploading {len(FILES)} files to {REPO_ID}")
for i, (src, dst) in enumerate(FILES, 1):
    if not os.path.exists(src):
        print(f"[{i}/{len(FILES)}] SKIP missing: {src}")
        continue
    sz_mb = os.path.getsize(src) / (1024 * 1024)
    t0 = time.time()
    print(f"[{i}/{len(FILES)}] {src} ({sz_mb:.1f} MB) -> {dst}", flush=True)
    api.upload_file(
        path_or_fileobj=src,
        path_in_repo=dst,
        repo_id=REPO_ID,
        commit_message=f"upload {dst}",
    )
    dt = time.time() - t0
    print(f"  done in {dt:.1f}s ({sz_mb/dt:.1f} MB/s)", flush=True)
print("[done] all uploads finished")
