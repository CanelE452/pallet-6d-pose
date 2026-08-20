"""PHASE 11 -- real capture as an appearance reference, observable axes only.

Real captures carry no pose label, so yaw and elevation are not estimated here;
inventing them would put a prediction-derived number next to a rendered ground
truth and invite a comparison that means nothing.  What is compared is what a
raw image actually shows: resolution and luma.  Object screen size is left out
rather than filled in with a detector output.

This is a reference, not a claim that the synthetic mixture matches real
appearance.
"""
from __future__ import annotations

import json
import pathlib
import sys

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import da_common as DA  # noqa: E402

RAW = DA.ROOT / "data/pallet/raw_data"
SESSIONS = ["night", "outside", "capture0403middle", "capture02", "capture03",
            "wood", "capture0403noapril", "real_pool_all"]
SAMPLE_PER_SESSION = 300
SEED = 20260905
EXTS = {".png", ".jpg", ".jpeg"}


def sample_images(folder, rng):
    if not folder.exists():
        return []
    files = [p for p in folder.rglob("*") if p.suffix.lower() in EXTS]
    if not files:
        return []
    if len(files) > SAMPLE_PER_SESSION:
        idx = rng.choice(len(files), SAMPLE_PER_SESSION, replace=False)
        files = [files[i] for i in sorted(idx)]
    return files


def main():
    rng = np.random.default_rng(SEED)
    rows = []
    for name in SESSIONS:
        for path in sample_images(RAW / name, rng):
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                continue
            grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            rows.append({"session": name,
                         "resolution": f"{image.shape[1]}x{image.shape[0]}",
                         "luma_p10": float(np.percentile(grey, 10)),
                         "luma_p50": float(np.percentile(grey, 50)),
                         "luma_p90": float(np.percentile(grey, 90))})
        print(f"  {name:22} {sum(1 for r in rows if r['session'] == name):5d}",
              flush=True)
    real = pd.DataFrame(rows)
    if real.empty:
        print("  no real captures found -- PHASE 11 skipped")
        return
    real.to_parquet(DA.AUDIT / "real_reference_features.parquet", index=False)

    pos = pd.read_parquet(DA.AUDIT / "positive_frame_features.parquet")
    report = {
        "scope": "observable axes only (resolution, luma). No pose label"
                 " exists for these frames, so yaw/elevation are not estimated"
                 " and no distribution match is claimed.",
        "n_real_sampled": int(len(real)),
        "sample_per_session": SAMPLE_PER_SESSION,
        "real_resolution": real["resolution"].value_counts().head(10).to_dict(),
        "synthetic_resolution": pos["resolution"].value_counts().to_dict(),
        "real_luma": {k: round(float(real[f"luma_{k}"].median()), 2)
                      for k in ("p10", "p50", "p90")},
        "synthetic_luma_frame_median": round(
            float(pos["luma_frame"].median()), 2),
        "real_luma_p50_by_session": {
            k: round(float(v), 2) for k, v in
            real.groupby("session")["luma_p50"].median().items()},
    }
    (DA.AUDIT / "REAL_REFERENCE.json").write_text(json.dumps(report, indent=1))
    print(f"  real resolutions: {list(report['real_resolution'])[:4]}")
    print(f"  real luma p50 median {report['real_luma']['p50']}  vs "
          f"synthetic {report['synthetic_luma_frame_median']}")
    print("-> REAL_REFERENCE.json")


if __name__ == "__main__":
    main()
