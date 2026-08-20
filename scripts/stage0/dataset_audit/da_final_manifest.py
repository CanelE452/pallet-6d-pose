"""PHASE 2 -- FINAL_SYNTH_TRAIN_V1: the whole BROAD 40,000, as a manifest.

This is a different contract from PAPER_CORE_V1 and is named so it cannot be
confused with it:

    PAPER_CORE_V1        33,758   architecture-development contract (MH_TRAIN)
    FINAL_SYNTH_TRAIN_V1 40,000   final-training contract for real evaluation

Folding the historical MH_DEV back in is only legitimate because the final claim
moves to REAL IN-HOUSE DEV/TEST.  The manifest records that explicitly, so a
later reader cannot quote MH_DEV numbers from this checkpoint as held out.

No image is copied.  `_src_shard` comes from the release's own records.jsonl,
not from a directory name.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import da_common as DA  # noqa: E402

BROAD = DA.RELEASE / "v2_prod40k_clean_merged"
FIELDS = ["frame_id", "rgb_path", "label_path", "mask_visible_path",
          "mask_amodal_path", "src_shard", "src_id", "resolution",
          "pallet_type", "elevation_deg_actual", "canonical_frontal_yaw_deg",
          "camera_distance_actual_m", "bbox_diag_norm", "V_vis_actual",
          "n_inframe", "n_supervised", "declared_loss_cause",
          "historical_split"]


def shard_index():
    """stem -> (_src_shard, _src_id), read from the release's own records."""
    out = {}
    with (BROAD / "records.jsonl").open() as fh:
        for line in fh:
            row = json.loads(line)
            # The stem is usable_id, not idx.  Measured, not assumed:
            # f{usable_id:04d} covers 40,000/40,000 stems while f{idx:04d}
            # covers 38,955 -- idx restarts per source shard.  %04d also
            # becomes 5 digits at 10,000+, the naming trap the EDGE README
            # warns about, which applies to BROAD too.
            out[f"f{row['usable_id']:04d}"] = (row.get("_src_shard"),
                                               row.get("_src_id"))
    return out


def main():
    frame = pd.read_parquet(DA.AUDIT / "positive_frame_features.parquet")
    broad = frame[frame["dataset_id"] == "BROAD_40K"].copy()
    if len(broad) != 40000:
        raise SystemExit(f"expected 40,000 BROAD rows, got {len(broad)}")
    if broad["frame_id"].nunique() != 40000:
        raise SystemExit("BROAD frame ids are not unique")

    shards = shard_index()
    missing = [s for s in broad["frame_id"] if s not in shards]
    if missing:
        raise SystemExit(f"{len(missing)} stems absent from records.jsonl, "
                         f"e.g. {missing[:3]}")

    rel = BROAD.relative_to(DA.ROOT)
    items = []
    for row in broad.sort_values("frame_id").itertuples():
        shard, src_id = shards[row.frame_id]
        items.append({
            "frame_id": row.frame_id,
            "rgb_path": f"{rel}/rgb/{row.frame_id}_rgb.png",
            "label_path": f"{rel}/labels/{row.frame_id}_label.json",
            "mask_visible_path": f"{rel}/mask_visible/{row.frame_id}.png",
            "mask_amodal_path": f"{rel}/mask_amodal/{row.frame_id}.png",
            "src_shard": shard,
            "src_id": src_id,
            "resolution": row.resolution,
            "pallet_type": row.pallet_type,
            "elevation_deg_actual": row.elevation_deg_actual,
            "canonical_frontal_yaw_deg": (
                None if row.canonical_frontal_yaw_deg is None
                or (isinstance(row.canonical_frontal_yaw_deg, float)
                    and np.isnan(row.canonical_frontal_yaw_deg))
                else round(float(row.canonical_frontal_yaw_deg), 4)),
            "camera_distance_actual_m": row.camera_distance_actual_m,
            "bbox_diag_norm": round(float(row.bbox_diag_norm), 6),
            "V_vis_actual": row.V_vis_actual,
            "n_inframe": int(row.n_inframe),
            "n_supervised": int(row.n_supervised),
            "declared_loss_cause": row.declared_loss_cause,
            "historical_split": row.mh_split,
            "sampling_weight": 1.0 / 40000,
            "branch": "corner+line",
        })

    payload = {
        "manifest": "FINAL_SYNTH_TRAIN_V1",
        "purpose": "final neural training pool for REAL IN-HOUSE evaluation",
        "n_unique": len(items),
        "composition": {"BROAD_40K": len(items)},
        "historical_split_counts":
            broad["mh_split"].value_counts(dropna=False).to_dict(),
        "supersedes_for_final_training": "PAPER_CORE_V1 (33,758)",
        "not_the_same_as": {
            "PAPER_CORE_V1": "architecture-development contract, MH_TRAIN"
                             " 33,758. Keep for reproducing past experiments."},
        "MH_DEV_WARNING":
            "the historical MH_DEV 6,242 is INSIDE this pool. Any checkpoint"
            " trained on it must never report MH_DEV as unseen or held out,"
            " and must not use it for checkpoint selection or early stopping.",
        "excluded": {
            "CORNER_LA_Y15_30": "targeted enrichment benefit NOT_ESTABLISHED",
            "CORNER_LA_Y30_PLUS": "targeted enrichment benefit NOT_ESTABLISHED",
            "CORNER_LA_FRONTAL": "0 frames rendered",
            "EDGE_HARD_TRUNC": "V_vis<4, outside point-valid support; F3 needs a"
                               " Point-PnP initialisation first",
            "EDGE_HARD_CLEAN_UNTOUCHED": "QA / control holdout",
            "NEGATIVE_SYNTH_V1": "dense negative pose training REJECTED;"
                                 " role is score_4kp calibration only"},
        "fields": FIELDS + ["sampling_weight", "branch"],
        "items": items,
    }
    out = DA.RELEASE_OUT / "FINAL_SYNTH_TRAIN_V1.json"
    text = json.dumps(payload, indent=1, default=str)
    out.write_text(text)
    digest = hashlib.sha256(text.encode()).hexdigest()
    (DA.RELEASE_OUT / "FINAL_SYNTH_TRAIN_V1.sha256").write_text(
        f"{digest}  FINAL_SYNTH_TRAIN_V1.json\n")

    print(f"  n_unique {payload['n_unique']}")
    print(f"  historical split {payload['historical_split_counts']}")
    print(f"  shards {broad['frame_id'].map(lambda s: shards[s][0]).nunique()}"
          f" (None 포함)")
    print(f"  sha256 {digest}")
    print("-> FINAL_SYNTH_TRAIN_V1.json / .sha256")


if __name__ == "__main__":
    main()
