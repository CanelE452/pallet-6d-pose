"""PHASE 9-10 / 13-14 -- manifests, exposure policy, mixture validation.

No images are copied.  A manifest is a list of (dataset_id, frame_path, branch,
sampling_weight, stratum), so the original pools stay untouched and a ratio can
be reproduced exactly.

The EDGE exposure ratio is chosen by a rule written down here, before any
coverage number is read, and the rule is lexicographic exactly as PHASE 7A
requires -- fill the line-hard deficit first, then protect the broad modes.  The
candidate grid is fixed; the data picks from it, not the other way round.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import da_common as DA      # noqa: E402
import da_coverage as CV    # noqa: E402

CANDIDATE_RATIOS = {"CONSERVATIVE": 0.05, "BALANCED": 0.12, "AGGRESSIVE": 0.20}

# --- selection rule, fixed before any number is looked at -------------------
LINE_HARD_BROAD_MAX_SHARE = 0.01   # a cell BROAD barely covers
LINE_HARD_EDGE_MIN_RATIO = 2.0     # ...and EDGE covers at least twice as densely
LINE_HARD_TARGET_GAIN = 2.0        # required exposure multiple after mixing
BROAD_MODE_MIN_SHARE = 0.05        # a cell that carries the broad distribution
BROAD_MODE_RETENTION = 0.85        # ...must keep this much of its exposure
MAX_EDGE_REPEAT = 3.0              # unique EDGE frames may not be recycled more
                                   # than this many times per broad pass

LINE_CELL_KEYS = ("vvis_bin", "trunc_label", "size_bin")


def cell_shares(block, keys):
    if block.empty:
        return pd.Series(dtype=float)
    return block.groupby(list(keys), observed=True).size() / len(block)


def mixture_share(broad_share, edge_share, ratio):
    index = broad_share.index.union(edge_share.index)
    b = broad_share.reindex(index, fill_value=0.0)
    e = edge_share.reindex(index, fill_value=0.0)
    return (1 - ratio) * b + ratio * e


def evaluate_ratio(broad_share, edge_share, ratio, hard_cells, mode_cells,
                   n_broad, n_edge):
    """Clause 1 is reported as VACUOUS wherever BROAD covers a cell with zero
    frames: a "reach 2x the BROAD exposure" test divides by zero there and is
    satisfied by any ratio above 0, so it cannot rank the candidates.  That is
    the case here -- the G1 gate (V_vis >= 4) means BROAD has no V_vis <= 3
    frames at all -- and it is recorded rather than quietly passed."""
    mixed = mixture_share(broad_share, edge_share, ratio)
    base = broad_share.reindex(mixed.index, fill_value=0.0)
    gains, retention = {}, {}
    for cell in hard_cells:
        before = float(base.get(cell, 0.0))
        after = float(mixed.get(cell, 0.0))
        gains[str(cell)] = (after / before) if before > 0 else float("inf")
    for cell in mode_cells:
        before = float(base.get(cell, 0.0))
        after = float(mixed.get(cell, 0.0))
        retention[str(cell)] = (after / before) if before > 0 else 1.0
    repeat = (ratio / max(n_edge, 1)) / max((1 - ratio) / max(n_broad, 1), 1e-12)
    vacuous = [str(c) for c in hard_cells if float(base.get(c, 0.0)) == 0.0]
    absolute = float(sum(float(mixed.get(c, 0.0)) for c in hard_cells))
    return {
        "clause_1_vacuous_cells": vacuous,
        "clause_1_is_vacuous": len(vacuous) == len(hard_cells) and bool(hard_cells),
        "hard_region_share_after": round(absolute, 5),
        "ratio": ratio,
        "hard_cell_gain_min": min(gains.values()) if gains else None,
        "hard_cells_meeting_target":
            int(sum(1 for v in gains.values() if v >= LINE_HARD_TARGET_GAIN)),
        "hard_cells_total": len(gains),
        "broad_mode_retention_min": min(retention.values()) if retention else 1.0,
        "edge_repeat_vs_broad_pass": round(float(repeat), 3),
        "gains": {k: (None if np.isinf(v) else round(v, 3))
                  for k, v in gains.items()},
        "retention": {k: round(v, 4) for k, v in retention.items()},
    }


def choose_ratio(report):
    """Smallest candidate satisfying every clause that can actually discriminate.

    When clause 1 is vacuous the choice is made by the protective clauses alone
    plus a conservatism default, and the caller records that -- claiming the
    coverage analysis selected the ratio would overstate what was measured.
    """
    reasons = []
    for name in ("CONSERVATIVE", "BALANCED", "AGGRESSIVE"):
        entry = report[name]
        clause1 = (entry["clause_1_is_vacuous"]
                   or entry["hard_cells_meeting_target"]
                   == entry["hard_cells_total"])
        ok = (entry["hard_cells_total"] > 0 and clause1
              and entry["broad_mode_retention_min"] >= BROAD_MODE_RETENTION
              and entry["edge_repeat_vs_broad_pass"] <= MAX_EDGE_REPEAT)
        reasons.append((name, ok, entry))
        if ok:
            vac = entry["clause_1_is_vacuous"]
            basis = ("protective clauses only (clause 1 vacuous: BROAD covers"
                     " every line-hard cell with 0 frames, so no ratio can be"
                     " shown to be required by coverage) + conservatism default"
                     if vac else "all three clauses")
            return name, basis
    return None, "no candidate satisfied the protective clauses"


def frame_path(dataset_id, frame_id):
    source = next(s for s in DA.POSITIVE_SOURCES if s.dataset_id == dataset_id)
    rel = source.path.relative_to(DA.ROOT)
    if source.kind == "dir":
        return f"{rel}/labels/{frame_id}_label.json"
    return f"{rel}::labels/{frame_id}_label.json"


def write_manifest(path, name, branch, blocks, policy):
    items = []
    for dataset_id, frame, weight, stratum in blocks:
        items.append({"dataset_id": dataset_id,
                      "frame_path": frame_path(dataset_id, frame),
                      "frame_id": frame,
                      "branch": branch,
                      "sampling_weight": round(float(weight), 8),
                      "stratum": stratum})
    payload = {"manifest": name, "branch": branch, "n_unique": len(items),
               "policy": policy, "items": items}
    path.write_text(json.dumps(payload, indent=1))
    return len(items)


def main():
    DA.RELEASE_OUT.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(DA.AUDIT / "positive_frame_features_binned.parquet")
    frame["trunc_label"] = np.where(frame["truncation"].fillna(False),
                                    "truncated", "full")

    broad_train = frame[(frame["dataset_id"] == "BROAD_40K")
                        & (frame["mh_split"] == "MH_TRAIN")]
    broad_dev = frame[(frame["dataset_id"] == "BROAD_40K")
                      & (frame["mh_split"] == "MH_DEV")]
    edge_train = frame[frame["dataset_id"] == "EDGE_HARD_TRUNC_TRAIN"]

    broad_share = cell_shares(broad_train, LINE_CELL_KEYS)
    edge_share = cell_shares(edge_train, LINE_CELL_KEYS)

    index = broad_share.index.union(edge_share.index)
    b = broad_share.reindex(index, fill_value=0.0)
    e = edge_share.reindex(index, fill_value=0.0)
    hard_cells = [c for c in index
                  if b[c] <= LINE_HARD_BROAD_MAX_SHARE
                  and (e[c] >= LINE_HARD_EDGE_MIN_RATIO * b[c]) and e[c] > 0]
    mode_cells = [c for c in index if b[c] >= BROAD_MODE_MIN_SHARE]

    report = {name: evaluate_ratio(broad_share, edge_share, ratio, hard_cells,
                                   mode_cells, len(broad_train), len(edge_train))
              for name, ratio in CANDIDATE_RATIOS.items()}
    chosen, chosen_basis = choose_ratio(report)
    policy = {
        "cell_keys": list(LINE_CELL_KEYS),
        "rule": {
            "line_hard_cell": f"BROAD share <= {LINE_HARD_BROAD_MAX_SHARE} and"
                              f" EDGE share >= {LINE_HARD_EDGE_MIN_RATIO}x BROAD",
            "clause_1": f"every line-hard cell reaches >= "
                        f"{LINE_HARD_TARGET_GAIN}x its BROAD-only exposure",
            "clause_2": f"every BROAD mode (share >= {BROAD_MODE_MIN_SHARE})"
                        f" keeps >= {BROAD_MODE_RETENTION} of its exposure",
            "clause_3": f"unique EDGE frames repeat <= {MAX_EDGE_REPEAT}x per"
                        f" broad pass",
            "selection": "smallest candidate satisfying all clauses",
        },
        "candidates": CANDIDATE_RATIOS,
        "n_line_hard_cells": len(hard_cells),
        "n_broad_mode_cells": len(mode_cells),
        "line_hard_cells": [str(c) for c in hard_cells],
        "evaluation": report,
        "CHOSEN": chosen,
        "CHOSEN_RATIO": CANDIDATE_RATIOS.get(chosen) if chosen else None,
        "CHOSEN_BASIS": chosen_basis,
        "clause_1_status": "VACUOUS -- BROAD has zero V_vis<=3 frames because"
                           " the G1 gate requires V_vis>=4, so 'reach 2x the"
                           " BROAD exposure' divides by zero and passes at any"
                           " ratio. Reported, not silently passed.",
        "note_if_none": None if chosen else
            "no candidate satisfied every clause; EDGE stays a candidate and is"
            " not promoted. The manifest below records exposure 0.",
    }

    # ---- PAPER_CORE_V1 ----------------------------------------------------
    core_weight = 1.0 / max(len(broad_train), 1)   # every manifest normalises
    core_items = [("BROAD_40K", f, core_weight, s) for f, s in
                  zip(broad_train["frame_id"], broad_train["stratum"]
                      if "stratum" in broad_train else broad_train["elev_bin"])]
    n_corner = write_manifest(
        DA.RELEASE_OUT / "PAPER_CORE_V1_corner_manifest.json",
        "PAPER_CORE_V1", "corner", core_items,
        {"composition": "BROAD_40K MH_TRAIN only",
         "reason": "the only pool with established positive training evidence"})
    n_line = write_manifest(
        DA.RELEASE_OUT / "PAPER_CORE_V1_line_manifest.json",
        "PAPER_CORE_V1", "line", core_items,
        {"composition": "BROAD_40K MH_TRAIN only",
         "reason": "line branch shares the core; no add-on is qualified"})

    # ---- DEPLOYMENT_CANDIDATE_V1 -----------------------------------------
    ratio = CANDIDATE_RATIOS.get(chosen, 0.0) if chosen else 0.0
    dep_corner = core_items
    write_manifest(
        DA.RELEASE_OUT / "DEPLOYMENT_CANDIDATE_V1_corner_manifest.json",
        "DEPLOYMENT_CANDIDATE_V1", "corner", dep_corner,
        {"composition": "BROAD_40K MH_TRAIN only",
         "reason": "CORNER_LA enrichment is NOT_ESTABLISHED, so it is not"
                   " promoted even for deployment"})

    broad_weight = (1 - ratio) / max(len(broad_train), 1)
    edge_weight = ratio / max(len(edge_train), 1)
    dep_line = [("BROAD_40K", f, broad_weight, "BROAD")
                for f in broad_train["frame_id"]]
    dep_line += [("EDGE_HARD_TRUNC_TRAIN", f, edge_weight, "EDGE_LINE_HARD")
                 for f in edge_train["frame_id"]]
    n_dep_line = write_manifest(
        DA.RELEASE_OUT / "DEPLOYMENT_CANDIDATE_V1_line_manifest.json",
        "DEPLOYMENT_CANDIDATE_V1", "line", dep_line,
        {"composition": f"BROAD_40K MH_TRAIN + EDGE_HARD_TRUNC_TRAIN at"
                        f" exposure {ratio}",
         "edge_exposure_ratio": ratio,
         "reason": "EDGE is point-invalid by design (G1 false) and never enters"
                   " the corner stream"})

    # ---- PHASE 13 before/after -------------------------------------------
    before = cell_shares(broad_train, LINE_CELL_KEYS)
    after = mixture_share(before, edge_share, ratio)
    validation = {
        "edge_exposure_ratio": ratio,
        "cells": {str(c): {"before": round(float(before.get(c, 0.0)), 6),
                           "after": round(float(after.get(c, 0.0)), 6)}
                  for c in after.index},
        "diversity_preserved": {},
    }
    for axis in ("pallet_type", "resolution", "background_asset", "noise_tier"):
        if axis not in broad_train:
            continue
        b_counts = broad_train[axis].astype(str).value_counts(normalize=True)
        mix = pd.concat([broad_train[axis], edge_train[axis]]).astype(str)
        m_counts = mix.value_counts(normalize=True)
        validation["diversity_preserved"][axis] = {
            "broad_unique": int(b_counts.size),
            "mixture_unique": int(m_counts.size),
            "broad_top_share": round(float(b_counts.iloc[0]), 4),
            "mixture_top_share": round(float(m_counts.iloc[0]), 4)}

    (DA.RELEASE_OUT / "SAMPLING_POLICY.json").write_text(
        json.dumps({"policy": policy, "validation": validation,
                    "PAPER_CORE_V1": {"corner_n": n_corner, "line_n": n_line},
                    "DEPLOYMENT_CANDIDATE_V1": {
                        "corner_n": len(dep_corner), "line_n": n_dep_line},
                    "excluded_from_training": {
                        "BROAD_40K MH_DEV": int(len(broad_dev)),
                        "EDGE_HARD_TRUNC_DEV": int((frame["dataset_id"] ==
                                                    "EDGE_HARD_TRUNC_DEV").sum()),
                        "EDGE_HARD_TRUNC_UNTOUCHED": int(
                            (frame["dataset_id"] ==
                             "EDGE_HARD_TRUNC_UNTOUCHED").sum()),
                        "EDGE_HARD_CLEAN_UNTOUCHED": int(
                            (frame["dataset_id"] ==
                             "EDGE_HARD_CLEAN_UNTOUCHED").sum()),
                        "CORNER_LA_Y15_30": int((frame["dataset_id"] ==
                                                 "CORNER_LA_Y15_30").sum()),
                        "CORNER_LA_Y30_PLUS": int((frame["dataset_id"] ==
                                                   "CORNER_LA_Y30_PLUS").sum()),
                        "NEGATIVE_SYNTH_V1": 10000},
                    }, indent=1, default=str))

    digest = []
    for path in sorted(DA.RELEASE_OUT.glob("*.json")):
        digest.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  "
                      f"{path.name}")
    (DA.RELEASE_OUT / "checksums.sha256").write_text("\n".join(digest) + "\n")

    print(f"  line-hard cells {len(hard_cells)}   broad modes {len(mode_cells)}")
    for name in ("CONSERVATIVE", "BALANCED", "AGGRESSIVE"):
        e = report[name]
        print(f"  {name:13} ratio {e['ratio']:.2f}  hard met "
              f"{e['hard_cells_meeting_target']}/{e['hard_cells_total']}  "
              f"mode retention min {e['broad_mode_retention_min']:.3f}  "
              f"edge repeat {e['edge_repeat_vs_broad_pass']:.2f}x")
    print(f"  CHOSEN = {chosen}  (ratio {ratio})")
    print("-> dataset_release/")


if __name__ == "__main__":
    main()
