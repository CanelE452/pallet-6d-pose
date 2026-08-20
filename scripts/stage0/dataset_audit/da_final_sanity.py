"""PHASE 3 -- sanity on FINAL_SYNTH_TRAIN_V1, plus the EDGE ablation manifest.

The point is not to re-describe the data; the full EDA already exists.  The
point is to catch a manifest that silently dropped a pool or skewed an axis
during construction.  Every distribution here is compared against the parquet
the manifest was built from, so a mismatch is a build bug, not a finding.

Nothing here is allowed to trigger oversampling of a cell.
"""
from __future__ import annotations

import hashlib, json, pathlib, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import da_common as DA      # noqa: E402
import da_coverage as CV    # noqa: E402

EDA = DA.OUT / "eda_final"
NUMERIC = ["elevation_deg_actual", "canonical_frontal_yaw_deg",
           "camera_distance_actual_m", "bbox_diag_norm", "bbox_area_fraction",
           "V_vis_actual", "n_inframe", "n_supervised", "luma_frame"]
CATEGORICAL = ["resolution", "pallet_type", "background_asset", "scene_preset",
               "truncation", "self_occlusion", "external_occlusion",
               "src_shard"]


def edge_ablation_manifest(frame):
    edge = frame[frame["dataset_id"] == "EDGE_HARD_TRUNC_TRAIN"]
    src = next(s for s in DA.POSITIVE_SOURCES
               if s.dataset_id == "EDGE_HARD_TRUNC_TRAIN")
    rel = src.path.relative_to(DA.ROOT)
    payload = {
        "manifest": "EDGE_HARD_LINE_ABLATION",
        "branch": "line",
        "sampling_weight": 0,
        "status": "PRESERVED, NOT USED IN FINAL TRAINING",
        "why_zero": [
            "V_vis < 4: outside point-valid support (G1 gate 0% by design)",
            "the final F3 route needs a Point-PnP initialisation first",
            "EDGE -> line gain -> final pose benefit is NOT_TESTED",
        ],
        "reopen_condition": "only if REAL evaluation shows a line-hard failure"
                            " in deployment",
        "never": "corner stream",
        "n_unique": int(len(edge)),
        "items": [{"dataset_id": "EDGE_HARD_TRUNC_TRAIN",
                   "frame_id": stem,
                   "frame_path": f"{rel}::labels/{stem}_label.json",
                   "branch": "line", "sampling_weight": 0,
                   "stratum": "EDGE_LINE_HARD"}
                  for stem in sorted(edge["frame_id"])],
    }
    text = json.dumps(payload, indent=1)
    (DA.RELEASE_OUT / "EDGE_HARD_LINE_ABLATION.json").write_text(text)
    return len(edge), hashlib.sha256(text.encode()).hexdigest()


def main():
    EDA.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(
        (DA.RELEASE_OUT / "FINAL_SYNTH_TRAIN_V1.json").read_text())
    items = pd.DataFrame(manifest["items"])
    frame = pd.read_parquet(DA.AUDIT / "positive_frame_features_binned.parquet")
    broad = frame[frame["dataset_id"] == "BROAD_40K"].copy()

    checks = {"n_unique": int(items["frame_id"].nunique()),
              "n_rows": int(len(items)),
              "matches_parquet": bool(
                  set(items["frame_id"]) == set(broad["frame_id"]))}
    if not (checks["n_unique"] == checks["n_rows"] == 40000
            and checks["matches_parquet"]):
        raise SystemExit(f"manifest integrity failed: {checks}")

    merged = broad.merge(items[["frame_id", "src_shard"]], on="frame_id")
    merged["trunc_label"] = np.where(merged["truncation"].fillna(False),
                                     "truncated", "full")

    summary = {"n": 40000, "numeric": {}, "categorical": {}, "checks": checks}
    for axis in NUMERIC:
        values = pd.to_numeric(merged[axis], errors="coerce").dropna()
        q = values.quantile([.05, .25, .5, .75, .95])
        summary["numeric"][axis] = {
            "n": int(len(values)), "mean": round(float(values.mean()), 4),
            "std": round(float(values.std()), 4),
            "p05": round(float(q.loc[.05]), 4), "p25": round(float(q.loc[.25]), 4),
            "median": round(float(q.loc[.50]), 4),
            "p75": round(float(q.loc[.75]), 4), "p95": round(float(q.loc[.95]), 4)}
    for axis in CATEGORICAL:
        counts = merged[axis].astype(str).value_counts()
        summary["categorical"][axis] = {
            "n_unique": int(counts.size),
            "top": str(counts.index[0]),
            "top_share": round(float(counts.iloc[0] / counts.sum()), 4),
            "counts": {str(k): int(v) for k, v in counts.head(25).items()}}

    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    for ax, axis in zip(axes.ravel(), NUMERIC):
        values = pd.to_numeric(merged[axis], errors="coerce").dropna()
        ax.hist(values, bins=40, color="#22405f")
        ax.set_title(axis, fontsize=8)
        ax.grid(alpha=.25)
    fig.suptitle("FINAL_SYNTH_TRAIN_V1 (BROAD 40,000) -- 1D sanity")
    fig.tight_layout()
    fig.savefig(EDA / "F01_1d_sanity.png", dpi=130)
    plt.close(fig)

    pairs = [("elev_bin", "yaw_bin", CV.ELEV_NAMES, CV.YAW_NAMES),
             ("elev_bin", "vvis_bin", CV.ELEV_NAMES, CV.VVIS_NAMES),
             ("yaw_bin", "vvis_bin", CV.YAW_NAMES, CV.VVIS_NAMES),
             ("size_bin", "dist_bin", CV.SIZE_NAMES, CV.DIST_NAMES)]
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.2))
    for ax, (rows, cols, rn, cn) in zip(axes, pairs):
        table = (pd.crosstab(merged[rows], merged[cols])
                 .reindex(index=rn, columns=cn, fill_value=0))
        data = table.to_numpy(float)
        ax.imshow(data / max(data.sum(), 1) * 100, cmap="magma", aspect="auto")
        ax.set_xticks(range(len(cn)), cn, fontsize=7, rotation=30)
        ax.set_yticks(range(len(rn)), rn, fontsize=7)
        ax.set_title(f"{rows} x {cols}", fontsize=9)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                if data[i, j]:
                    ax.text(j, i, int(data[i, j]), ha="center", va="center",
                            fontsize=6, color="white")
        summary.setdefault("cells", {})[f"{rows} x {cols}"] = {
            f"{r}|{c}": int(table.loc[r, c]) for r in rn for c in cn}
    fig.suptitle("FINAL_SYNTH_TRAIN_V1 -- 2D coverage")
    fig.tight_layout()
    fig.savefig(EDA / "F02_2d_coverage.png", dpi=130)
    plt.close(fig)

    n_edge, edge_sha = edge_ablation_manifest(frame)
    summary["EDGE_HARD_LINE_ABLATION"] = {"n": n_edge, "sampling_weight": 0,
                                          "sha256": edge_sha}
    (EDA / "FINAL_TRAIN_SANITY.json").write_text(
        json.dumps(summary, indent=1, default=str))

    print(f"  n_unique {checks['n_unique']}  parquet 일치 {checks['matches_parquet']}")
    for axis in ("elevation_deg_actual", "canonical_frontal_yaw_deg",
                 "V_vis_actual", "bbox_diag_norm"):
        e = summary["numeric"][axis]
        print(f"  {axis:26} med {e['median']:8.3f}  p05 {e['p05']:8.3f}"
              f"  p95 {e['p95']:8.3f}")
    for axis in ("resolution", "pallet_type", "src_shard"):
        e = summary["categorical"][axis]
        print(f"  {axis:26} unique {e['n_unique']:3d}  top {e['top']}"
              f" ({e['top_share']:.3f})")
    print(f"  EDGE ablation manifest n={n_edge} weight=0")
    print(f"-> {EDA}")


if __name__ == "__main__":
    main()
