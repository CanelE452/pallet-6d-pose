"""PHASE 5-6 figures.  Histogram and ECDF for every 1D axis, heatmaps for the
interactions, and a before/after panel for the deployment mixture.

BROAD is always drawn as the prior; add-ons are drawn against it rather than on
their own, because the question the brief asks is never "what does this add-on
look like" but "what does it change about the pool it joins".
"""
from __future__ import annotations

import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import da_common as DA      # noqa: E402
import da_coverage as CV    # noqa: E402

DPI = 130
POOL_COLOURS = {
    "BROAD_MH_TRAIN": "#22405f", "BROAD_MH_DEV": "#7fa8cc",
    "CORNER_LA_Y15_30": "#d18a2a", "CORNER_LA_Y30_PLUS": "#b3452c",
    "EDGE_HARD_TRUNC_TRAIN": "#3f7d4e", "EDGE_HARD_TRUNC_DEV": "#79b48a",
    "EDGE_HARD_TRUNC_UNTOUCHED": "#a8cfb4",
    "EDGE_HARD_CLEAN_UNTOUCHED": "#6a4c93",
}


def pools(frame):
    label = frame["dataset_id"].astype(str).copy()
    broad = frame["dataset_id"] == "BROAD_40K"
    label[broad] = "BROAD_" + frame.loc[broad, "mh_split"].fillna("UNSPLIT")
    return label


def _axis_figure(frame, column, path, title, bins=40):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for name, block in frame.groupby("pool", observed=True):
        values = pd.to_numeric(block[column], errors="coerce").dropna()
        if values.empty:
            continue
        colour = POOL_COLOURS.get(name)
        axes[0].hist(values, bins=bins, density=True, histtype="step",
                     lw=1.6, label=f"{name} (n={len(values)})", color=colour)
        ordered = np.sort(values.to_numpy())
        axes[1].plot(ordered, np.linspace(0, 1, len(ordered)),
                     lw=1.6, label=name, color=colour)
    axes[0].set_title(f"{title} -- density")
    axes[1].set_title(f"{title} -- ECDF")
    for ax in axes:
        ax.grid(alpha=.25)
        ax.set_xlabel(column)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def _categorical_figure(frame, column, path, title):
    table = (pd.crosstab(frame["pool"], frame[column].astype(str),
                         normalize="index") * 100)
    fig, ax = plt.subplots(figsize=(max(8, 1.1 * table.shape[1] + 4), 4.2))
    table.plot(kind="bar", stacked=True, ax=ax, width=.8, colormap="tab20")
    ax.set_ylabel("% of pool")
    ax.set_title(title)
    ax.legend(fontsize=7, ncol=2, bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def _heatmap(ax, block, rows, cols, row_names, col_names, title):
    table = (pd.crosstab(block[rows], block[cols])
             .reindex(index=row_names, columns=col_names, fill_value=0))
    data = table.to_numpy(float)
    share = data / max(data.sum(), 1) * 100
    im = ax.imshow(share, cmap="magma", aspect="auto")
    ax.set_xticks(range(len(col_names)), col_names, fontsize=7, rotation=30)
    ax.set_yticks(range(len(row_names)), row_names, fontsize=7)
    ax.set_title(f"{title}\n(n={int(data.sum())})", fontsize=8)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if data[i, j]:
                ax.text(j, i, int(data[i, j]), ha="center", va="center",
                        fontsize=6,
                        color="white" if share[i, j] < share.max() * .6 else "black")
    return im


def heat_panel(frame, groups, rows, cols, row_names, col_names, path, title):
    n = len(groups)
    fig, axes = plt.subplots(1, n, figsize=(4.1 * n, 4.0), squeeze=False)
    for ax, name in zip(axes[0], groups):
        block = frame[frame["pool"] == name] if name in set(frame["pool"]) \
            else frame[frame["_final"]] if name == "FINAL" else frame.iloc[:0]
        _heatmap(ax, block, rows, cols, row_names, col_names, f"{title}\n{name}")
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def main():
    DA.EDA.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(DA.AUDIT / "positive_frame_features_binned.parquet")
    frame["pool"] = pools(frame)

    # 01 counts
    counts = frame["pool"].value_counts()
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.bar(counts.index, counts.to_numpy(),
           color=[POOL_COLOURS.get(k, "#888") for k in counts.index])
    for i, v in enumerate(counts.to_numpy()):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("frames")
    ax.set_title("positive pools -- unique frames")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right", fontsize=8)
    fig.tight_layout()
    fig.savefig(DA.EDA / "01_dataset_counts.png", dpi=DPI)
    plt.close(fig)

    for name, column, title in [
            ("02_elevation_ecdf", "elevation_deg_actual", "elevation (deg)"),
            ("03_yaw_ecdf", "canonical_frontal_yaw_deg", "canonical frontal yaw (deg)"),
            ("04_distance_ecdf", "camera_distance_actual_m", "camera distance (m)"),
            ("05_bbox_diag_ecdf", "bbox_diag_norm", "bbox diagonal (normalised)"),
            ("06_vvis_distribution", "V_vis_actual", "V_vis_actual"),
            ("08_luma_distribution", "luma_frame", "frame luma"),
            ("21_line_valid_roles", "line_valid_roles", "line-valid roles (of 12)"),
            ("22_n_inframe", "n_inframe", "corners in frame"),
            ("23_bbox_area_fraction", "bbox_area_fraction", "bbox area fraction")]:
        _axis_figure(frame, column, DA.EDA / f"{name}.png", title)

    for name, column, title in [
            ("07_truncation_distribution", "truncation", "truncation flag"),
            ("09_resolution_distribution", "resolution", "resolution"),
            ("10_pallet_type_distribution", "pallet_type", "pallet type"),
            ("24_background_distribution", "background_asset", "background asset"),
            ("25_occlusion_distribution", "self_occlusion", "self occlusion")]:
        _categorical_figure(frame, column, DA.EDA / f"{name}.png", title)

    addons = [p for p in POOL_COLOURS if p.startswith(("CORNER_LA", "EDGE"))
              and p in set(frame["pool"])]
    heat_panel(frame, ["BROAD_MH_TRAIN"], "elev_bin", "yaw_bin",
               CV.ELEV_NAMES, CV.YAW_NAMES,
               DA.EDA / "11_elevation_yaw_heatmap_BROAD.png", "elev x yaw")
    heat_panel(frame, addons, "elev_bin", "yaw_bin",
               CV.ELEV_NAMES, CV.YAW_NAMES,
               DA.EDA / "12_elevation_yaw_heatmap_ADDONS.png", "elev x yaw")
    heat_panel(frame, ["BROAD_MH_TRAIN", *addons], "elev_bin", "vvis_bin",
               CV.ELEV_NAMES, CV.VVIS_NAMES,
               DA.EDA / "14_elevation_vvis_heatmap.png", "elev x V_vis")
    heat_panel(frame, ["BROAD_MH_TRAIN", *addons], "yaw_bin", "vvis_bin",
               CV.YAW_NAMES, CV.VVIS_NAMES,
               DA.EDA / "15_yaw_vvis_heatmap.png", "yaw x V_vis")
    heat_panel(frame, ["BROAD_MH_TRAIN", *addons], "size_bin", "dist_bin",
               CV.SIZE_NAMES, CV.DIST_NAMES,
               DA.EDA / "16_size_distance_heatmap.png", "size x distance")
    frame["trunc_label"] = np.where(frame["truncation"].fillna(False),
                                    "truncated", "full")
    heat_panel(frame, ["BROAD_MH_TRAIN", *addons], "trunc_label", "vvis_bin",
               ["full", "truncated"], CV.VVIS_NAMES,
               DA.EDA / "17_truncation_vvis_heatmap.png", "truncation x V_vis")
    heat_panel(frame, ["BROAD_MH_TRAIN", *addons], "elev_bin", "size_bin",
               CV.ELEV_NAMES, CV.SIZE_NAMES,
               DA.EDA / "26_elevation_size_heatmap.png", "elev x size")
    heat_panel(frame, ["BROAD_MH_TRAIN", *addons], "pallet_type", "elev_bin",
               sorted(frame["pallet_type"].dropna().unique()), CV.ELEV_NAMES,
               DA.EDA / "27_pallettype_elevation_heatmap.png", "type x elev")

    # 18/19 diversity
    div = pd.read_csv(DA.AUDIT / "CATEGORICAL_DIVERSITY.csv")
    for name, axis, path in [
            ("source diversity", "scene_preset", "18_source_diversity.png"),
            ("background diversity", "background_asset",
             "19_background_diversity.png")]:
        block = div[div["axis"] == axis]
        if block.empty:
            continue
        fig, ax = plt.subplots(figsize=(9, 4.0))
        ax.bar(block["dataset_id"], block["effective_count"], color="#3f7d4e")
        ax.plot(block["dataset_id"], block["n_unique"], "o--", color="#b3452c",
                label="unique count")
        ax.set_ylabel("effective count = exp(entropy)")
        ax.set_title(f"{name} -- effective vs unique")
        ax.legend(fontsize=8)
        plt.setp(ax.get_xticklabels(), rotation=20, ha="right", fontsize=7)
        fig.tight_layout()
        fig.savefig(DA.EDA / path, dpi=DPI)
        plt.close(fig)

    print(f"-> {len(list(DA.EDA.glob('*.png')))} figures in {DA.EDA}")


if __name__ == "__main__":
    main()
