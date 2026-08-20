"""PHASE 5 / 7 / 12 -- 1D tables, cell coverage, diversity.

Bin edges are fixed here, before any coverage number is looked at, and the
UNDERCOVERED / OVERCONCENTRATED labels are relative to BROAD's own share rather
than to an absolute count.  That matters because the brief forbids flattening
every cell to the same N: the question is never "is this cell small" but "is
this cell smaller than the prior it was drawn from".
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import da_common as DA  # noqa: E402

ELEV_BINS = [-np.inf, 8, 15, 30, 45, np.inf]
ELEV_NAMES = ["<8", "8-15", "15-30", "30-45", ">=45"]
YAW_BINS = [-np.inf, 15, 30, np.inf]
YAW_NAMES = ["<15", "15-30", ">=30"]
VVIS_BINS = [-np.inf, 3, 4, 5, 6, np.inf]
VVIS_NAMES = ["<=3", "4", "5", "6", ">=7"]
SIZE_BINS = [-np.inf, 0.25, 0.40, 0.60, 0.85, np.inf]
SIZE_NAMES = ["<0.25", "0.25-0.40", "0.40-0.60", "0.60-0.85", ">=0.85"]
DIST_BINS = [-np.inf, 2.0, 3.0, 4.5, 7.0, np.inf]
DIST_NAMES = ["<2m", "2-3m", "3-4.5m", "4.5-7m", ">=7m"]

# relative to BROAD's share of the same cell, decided before seeing results
UNDER_RATIO = 0.5
OVER_RATIO = 2.0
MIN_CELL_FOR_JUDGEMENT = 30

NUMERIC_AXES = ["elevation_deg_actual", "canonical_frontal_yaw_deg",
                "camera_distance_actual_m", "bbox_diag_norm",
                "bbox_area_fraction", "projected_size_actual",
                "V_vis_actual", "n_inframe", "n_supervised",
                "luma_frame", "luma_pallet", "line_valid_roles"]
CATEGORICAL_AXES = ["pallet_type", "resolution", "background_asset",
                    "scene_preset", "noise_tier", "truncation",
                    "self_occlusion", "external_occlusion"]


def add_bins(frame):
    frame = frame.copy()
    frame["elev_bin"] = pd.cut(frame["elevation_deg_actual"], ELEV_BINS,
                               labels=ELEV_NAMES, right=False)
    frame["yaw_bin"] = pd.cut(frame["canonical_frontal_yaw_deg"], YAW_BINS,
                              labels=YAW_NAMES, right=False)
    frame["vvis_bin"] = pd.cut(frame["V_vis_actual"], VVIS_BINS,
                               labels=VVIS_NAMES, right=True)
    frame["size_bin"] = pd.cut(frame["bbox_diag_norm"], SIZE_BINS,
                               labels=SIZE_NAMES, right=False)
    frame["dist_bin"] = pd.cut(frame["camera_distance_actual_m"], DIST_BINS,
                               labels=DIST_NAMES, right=False)
    return frame


def summary_table(frame, group="dataset_id"):
    rows = []
    for key, block in frame.groupby(group, dropna=False):
        for axis in NUMERIC_AXES:
            if axis not in block:
                continue
            values = pd.to_numeric(block[axis], errors="coerce").dropna()
            if values.empty:
                continue
            q = values.quantile([.05, .10, .25, .50, .75, .90, .95])
            rows.append({"group": key, "axis": axis, "N": int(len(values)),
                         "mean": float(values.mean()),
                         "std": float(values.std()),
                         "p05": float(q.loc[.05]), "p10": float(q.loc[.10]),
                         "p25": float(q.loc[.25]), "median": float(q.loc[.50]),
                         "p75": float(q.loc[.75]), "p90": float(q.loc[.90]),
                         "p95": float(q.loc[.95])})
    return pd.DataFrame(rows)


def effective_sources(series):
    """exp(Shannon entropy) -- how many sources a cell behaves like it has."""
    counts = series.dropna().astype(str).value_counts()
    if counts.empty:
        return 0.0, 0.0, None
    share = counts / counts.sum()
    entropy = float(-(share * np.log(share)).sum())
    return float(np.exp(entropy)), float(share.iloc[0]), str(counts.index[0])


def cell_table(frame, keys=("elev_bin", "yaw_bin", "vvis_bin")):
    broad = frame[frame["dataset_id"] == "BROAD_40K"]
    broad_share = (broad.groupby(list(keys), observed=True).size()
                   / max(len(broad), 1))
    rows = []
    for key, block in frame.groupby(["dataset_id", *keys], observed=True):
        dataset_id, cell = key[0], key[1:]
        pool = frame[frame["dataset_id"] == dataset_id]
        share = len(block) / max(len(pool), 1)
        prior = float(broad_share.get(cell, 0.0))
        if len(block) < MIN_CELL_FOR_JUDGEMENT or prior == 0.0:
            status = "INSUFFICIENT_N" if len(block) < MIN_CELL_FOR_JUDGEMENT \
                else "NOT_IN_PRIOR"
        else:
            ratio = share / prior
            status = ("UNDERCOVERED" if ratio < UNDER_RATIO else
                      "OVERCONCENTRATED" if ratio > OVER_RATIO else "ADEQUATE")
        eff_src, top_src_share, top_src = effective_sources(
            block.get("seed_background", pd.Series(dtype=object)))
        eff_bg, top_bg_share, top_bg = effective_sources(
            block.get("background_asset", pd.Series(dtype=object)))
        rows.append({
            "dataset_id": dataset_id,
            **{k: str(v) for k, v in zip(keys, cell)},
            "N_unique": int(len(block)),
            "share_in_pool": round(share, 6),
            "broad_prior_share": round(prior, 6),
            "ratio_vs_broad": round(share / prior, 4) if prior else None,
            "status": status,
            "n_pallet_type": int(block["pallet_type"].nunique()),
            "n_background": int(block["background_asset"].nunique()),
            "n_resolution": int(block["resolution"].nunique()),
            "effective_backgrounds": round(eff_bg, 3),
            "top_background": top_bg,
            "top_background_share": round(top_bg_share, 4),
            "effective_seed_sources": round(eff_src, 3),
            "source_dominance_warning": bool(top_bg_share > 0.5),
        })
    return pd.DataFrame(rows)


def main():
    DA.AUDIT.mkdir(parents=True, exist_ok=True)
    frame = add_bins(pd.read_parquet(DA.AUDIT / "positive_frame_features.parquet"))
    frame.to_parquet(DA.AUDIT / "positive_frame_features_binned.parquet",
                     index=False)

    summary_table(frame).to_csv(DA.AUDIT / "AXIS_SUMMARY.csv", index=False)

    pool = frame.copy()
    pool["dataset_id"] = np.where(
        pool["dataset_id"] == "BROAD_40K",
        "BROAD_" + pool["mh_split"].fillna("UNSPLIT").astype(str),
        pool["dataset_id"])
    summary_table(pool).to_csv(DA.AUDIT / "AXIS_SUMMARY_BY_SPLIT.csv",
                               index=False)

    cells = cell_table(frame)
    cells.to_csv(DA.AUDIT / "CELL_COVERAGE.csv", index=False)

    categorical = []
    for dataset_id, block in frame.groupby("dataset_id"):
        for axis in CATEGORICAL_AXES:
            if axis not in block:
                continue
            counts = block[axis].astype(str).value_counts()
            eff, top_share, top = effective_sources(block[axis])
            categorical.append({"dataset_id": dataset_id, "axis": axis,
                                "n_unique": int(counts.size),
                                "effective_count": round(eff, 3),
                                "top_value": top,
                                "top_share": round(top_share, 4)})
    pd.DataFrame(categorical).to_csv(DA.AUDIT / "CATEGORICAL_DIVERSITY.csv",
                                     index=False)

    status_counts = (cells.groupby(["dataset_id", "status"]).size()
                     .unstack(fill_value=0))
    stats = {
        "bins": {"elevation": ELEV_NAMES, "yaw": YAW_NAMES,
                 "vvis": VVIS_NAMES, "size": SIZE_NAMES, "dist": DIST_NAMES},
        "rule": {"under_ratio": UNDER_RATIO, "over_ratio": OVER_RATIO,
                 "min_cell_for_judgement": MIN_CELL_FOR_JUDGEMENT,
                 "reference": "BROAD_40K share of the same cell"},
        "status_counts": json.loads(status_counts.to_json(orient="index")),
        "n_cells": int(len(cells)),
    }
    (DA.AUDIT / "CELL_COVERAGE_SUMMARY.json").write_text(
        json.dumps(stats, indent=1))
    print(status_counts.to_string())
    print("-> AXIS_SUMMARY.csv / CELL_COVERAGE.csv / CATEGORICAL_DIVERSITY.csv")


if __name__ == "__main__":
    main()
