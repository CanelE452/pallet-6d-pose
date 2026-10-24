"""Read frozen source VAL outputs; stratify pixel states without new inference.

These are mask-derived pixel states, never semantic self-visibility labels.
Both a learned head and its stored frozen SAM logit prior are summarized.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def boolean(value):
    return str(value).lower() in {"true", "1"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--synthetic-labels", required=True)
    parser.add_argument("--ycb-queries", required=True)
    args = parser.parse_args()
    run = Path(args.run)
    out_json, out_csv = run / "SOURCE_PIXEL_STATE_METRICS.json", run / "SOURCE_PIXEL_STATE_METRICS.csv"
    if out_json.exists() or out_csv.exists():
        raise FileExistsError("refusing to overwrite source state metrics")
    with Path(args.synthetic_labels).open() as stream:
        synthetic = {(row["id"], int(row["native_corner"])): row for row in csv.DictReader(stream)}
    with Path(args.ycb_queries).open() as stream:
        public = {row["sample_id"]: row for row in csv.DictReader(stream)}
    with np.load(args.cache, allow_pickle=False) as cache:
        arrays = {key: cache[key] for key in ["id", "native_corner", "sample_id", "source", "y", "boundary", "sam"]}
    with np.load(run / "VAL_SOURCE_OUTPUTS.npz", allow_pickle=False) as outputs:
        index, probability, y = outputs["index"], outputs["probability"], outputs["y"]
    if not np.array_equal(y, arrays["y"][index]):
        raise AssertionError("saved outputs no longer align with source cache")
    sources = arrays["source"][index].astype(str)
    states = []
    for query in index:
        source = str(arrays["source"][query]).upper()
        if source == "SYNTH":
            row = synthetic[(str(arrays["id"][query]), int(arrays["native_corner"][query]))]
            visible, full = boolean(row["M4_member"]), boolean(row["M0_member"])
        elif source == "YCB":
            row = public[str(arrays["sample_id"][query])]
            visible, full = boolean(row["presence_target"]), boolean(row["full_member"])
        else:
            raise ValueError(f"unexpected source {source}")
        state = "visible_target" if visible else "external_proxy" if full else "outside_full_support"
        if int(visible) != int(arrays["y"][query]):
            raise AssertionError("pixel state conflicts with exact center target")
        states.append(state)
    states = np.asarray(states)
    sam_logit = 8 * arrays["sam"][index, 0, 4, 4].astype(np.float64)
    sam_probability = 1 / (1 + np.exp(-np.clip(sam_logit, -80, 80)))
    boundary = arrays["boundary"][index]
    rows = []
    for source in np.unique(sources):
        for state in ("visible_target", "external_proxy", "outside_full_support"):
            for spatial in ("ALL", "boundary", "nonboundary"):
                mask = (sources == source) & (states == state)
                if spatial != "ALL":
                    mask &= boundary if spatial == "boundary" else ~boundary
                if not mask.any():
                    continue
                target = y[mask].astype(bool)
                for arm, p in (("TRAINED_HEAD", probability[mask]), ("FROZEN_SAM_PRIOR", sam_probability[mask])):
                    pred = p >= 0.5
                    rows.append({"source": str(source), "pixel_state": state, "spatial": spatial, "arm": arm,
                                 "N": int(mask.sum()), "predicted_present": int(pred.sum()),
                                 "predicted_absent": int((~pred).sum()),
                                 "correct_count": int((pred == target).sum()),
                                 "correct_rate": float((pred == target).mean()),
                                 "mean_presence_probability": float(p.mean()),
                                 "TP": int((pred & target).sum()), "FP": int((pred & ~target).sum()),
                                 "FN": int((~pred & target).sum()), "TN": int((~pred & ~target).sum()),
                                 "threshold": 0.5})
    checkpoint = torch.load(run / "BEST_QUERY_HEAD.pt", map_location="cpu", weights_only=False)
    report = {"target": "query-pixel visible-instance membership", "semantic_self_GT": "UNKNOWN, not converted to outside or external",
              "source_cache": str(Path(args.cache).resolve()), "saved_source_outputs_sha256": sha256(run / "VAL_SOURCE_OUTPUTS.npz"),
              "checkpoint_sha256": sha256(run / "BEST_QUERY_HEAD.pt"),
              "selected_sam_prior_weight": float(checkpoint["state_dict"]["sam_prior_weight"]),
              "selected_update": checkpoint["updates"], "rows": rows,
              "new_inference": False, "current_evaluation_used": False,
              "source_label_sha256": {"SYNTH": sha256(args.synthetic_labels), "YCB": sha256(args.ycb_queries)}}
    out_json.write_text(json.dumps(report, indent=2) + "\n")
    with out_csv.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"selected_update": report["selected_update"],
                      "selected_sam_prior_weight": report["selected_sam_prior_weight"],
                      "ALL_state_rows": [row for row in rows if row["spatial"] == "ALL"]}))


if __name__ == "__main__":
    main()
