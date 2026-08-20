"""PHASE 3-4 -- one canonical row per frame, positives and negatives apart.

Line-role validity is not a proxy invented here: it is the same
`visible_segments(...)["hit"]` that the line loss masks with, over the same 12
structural edges, on the same 50-grid.  Anything else would report a coverage
number the training objective does not share.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "multihead"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "line"))

import da_common as DA                       # noqa: E402
import line_feature_capacity_v2 as V2        # noqa: E402
from mh_cigm import EDGES                    # noqa: E402

GRID = DA.GRID


def line_roles(rows_corners, widths, heights):
    """(N, 12) boolean: does each structural role intersect the frame."""
    pts = np.asarray(rows_corners, float)
    scale = np.stack([np.asarray(widths, float),
                      np.asarray(heights, float)], -1)[:, None, :]
    grid = pts * GRID / np.clip(scale, 1, None)
    _, _, p0, p1, length = V2.gt_lines(grid, EDGES)
    return V2.visible_segments(p0, p1, length)["hit"]


def build_positive():
    rows, corners, widths, heights = [], [], [], []
    for src in DA.POSITIVE_SOURCES:
        if not src.exists:
            print(f"  {src.dataset_id:28} MISSING -- skipped", flush=True)
            continue
        n = 0
        for stem, payload in src.labels():
            row = DA.positive_row(src.dataset_id, stem, payload)
            cuboid = (payload.get("objects") or [{}])[0].get("projected_cuboid")
            if cuboid and len(cuboid) == 8:
                row["_geom_index"] = len(corners)
                corners.append(cuboid)
                widths.append(row["width"])
                heights.append(row["height"])
            else:
                row["_geom_index"] = np.nan
            rows.append(row)
            n += 1
        print(f"  {src.dataset_id:28} {n:6d} frames", flush=True)
    return rows, corners, widths, heights


def main():
    rows, corners, widths, heights = build_positive()
    frame = pd.DataFrame(rows)
    if corners:
        hit = line_roles(corners, widths, heights)
        counts = hit.sum(1).astype(int)
        mapped = np.full(len(frame), -1, int)
        idx = frame["_geom_index"].to_numpy(dtype="float64")
        have = ~np.isnan(idx)
        mapped[have] = counts[idx[have].astype(int)]
        frame["line_valid_roles"] = np.where(mapped >= 0, mapped, np.nan)
        for role in range(len(EDGES)):
            col = np.full(len(frame), np.nan)
            col[have] = hit[idx[have].astype(int), role]
            frame[f"role{role:02d}_valid"] = col
    frame = frame.drop(columns=[c for c in ("_geom_index",) if c in frame])

    # split membership for BROAD comes from the frozen contract, not a guess
    split_csv = DA.OUT / "mh_split.csv"
    if split_csv.exists():
        split = pd.read_csv(split_csv)
        col = "stem" if "stem" in split.columns else split.columns[0]
        mapping = dict(zip(split[col], split["split"]))
        frame["mh_split"] = [
            mapping.get(f) if d == "BROAD_40K" else None
            for d, f in zip(frame["dataset_id"], frame["frame_id"])]
    else:
        frame["mh_split"] = None

    DA.AUDIT.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(DA.AUDIT / "positive_frame_features.parquet", index=False)
    print(f"  positive rows {len(frame)}  cols {len(frame.columns)}", flush=True)

    neg = []
    for src in DA.NEGATIVE_SOURCES:
        if not src.exists:
            print(f"  {src.dataset_id:28} MISSING -- skipped", flush=True)
            continue
        n = 0
        for stem, payload in src.labels():
            neg.append(DA.negative_row(src.dataset_id, stem, payload))
            n += 1
        print(f"  {src.dataset_id:28} {n:6d} frames", flush=True)
    negative = pd.DataFrame(neg)
    negative.to_parquet(DA.AUDIT / "negative_frame_features.parquet", index=False)
    print(f"  negative rows {len(negative)}  cols {len(negative.columns)}",
          flush=True)

    summary = {
        "positive_rows": int(len(frame)),
        "negative_rows": int(len(negative)),
        "positive_by_dataset": frame["dataset_id"].value_counts().to_dict(),
        "negative_by_dataset": negative["dataset_id"].value_counts().to_dict()
        if len(negative) else {},
        "missing_sources": [s.dataset_id for s in
                            DA.POSITIVE_SOURCES + DA.NEGATIVE_SOURCES
                            if not s.exists],
    }
    (DA.AUDIT / "feature_build_summary.json").write_text(
        json.dumps(summary, indent=1, default=str))
    print("-> positive_frame_features.parquet / negative_frame_features.parquet")


if __name__ == "__main__":
    main()
