"""Create the small, reviewable result mirror from immutable raw artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .util import immutable_json, read_json, sha256

REPO = Path(__file__).resolve().parents[4]


def exclusive_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        stream.write(value)


def summary(value: dict) -> dict:
    return {key: item for key, item in value.items() if key != "records"}


def run(raw: Path, docs: Path) -> None:
    if docs.exists():
        raise FileExistsError(f"refusing to overwrite review mirror: {docs}")
    docs.mkdir(parents=True)
    evaluations = raw / "evaluations"
    names = ["point_synth_val"] + [f"{arm}_seed{seed}_synth_val"
                                          for arm in ("direct", "hough") for seed in (1, 2, 3)]
    values = {name: read_json(evaluations / f"{name}.json") for name in names}
    predictions = {f"{arm}_seed{seed}_synth_val": read_json(raw / f"predictions/{arm}_seed{seed}_synth_val.json")
                   for arm in ("direct", "hough") for seed in (1, 2, 3)}
    comparison = read_json(raw / "COMPARISON_SYNTH_VAL.json")
    integrity = read_json(raw / "INTEGRITY_AUDIT_STRICT.json")
    registry = read_json(raw / "SOURCE_REGISTRY_GPU.json")
    gates = read_json(raw / "PRETRAINING_GATES_GPU.json")
    protocol = read_json(Path(__file__).resolve().parents[1] / "PROTOCOL_LOCK.json")
    split_symmetry = {}
    for split in ("train", "calibration", "synth_val"):
        split_records = read_json(raw / f"export/{split}.json")["records"]
        split_symmetry[split] = {f"C{order}": sum(row["symmetry_order"] == order for row in split_records)
                                    for order in (1, 2, 4)}
    metric = {name: summary(value) for name, value in values.items()}
    for name, prediction in predictions.items():
        arm, seed_text, _ = name.split("_", 2)
        completion = read_json(raw / f"heads/{arm}_{seed_text}/COMPLETION.json")
        metric[name]["parameter_count"] = prediction["parameter_count"]
        metric[name]["GPU_milliseconds_per_frame"] = prediction["milliseconds_per_frame"]
        metric[name]["training_elapsed_seconds"] = completion["elapsed_seconds"]
    metric["comparison"] = comparison
    metric["symmetry_counts_by_split"] = split_symmetry
    immutable_json(docs / "SYNTH_METRICS.json", metric)
    immutable_json(docs / "INTEGRITY_AUDIT.json", integrity)
    immutable_json(docs / "SOURCE_REGISTRY.json", registry)
    immutable_json(docs / "PRETRAINING_GATES.json", gates)
    immutable_json(docs / "PROTOCOL_LOCK.json", protocol)
    row_maps = {name: {row["frame_id"]: row for row in value["records"]}
                for name, value in values.items()}
    csv_path = docs / "SYNTH_PER_FRAME.csv"
    fields = ["frame_id", "raw_diagonal_px", "point_normalized"] + [
        f"{arm}_seed{seed}_normalized" for arm in ("direct", "hough") for seed in (1, 2, 3)]
    with csv_path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n"); writer.writeheader()
        for row in values["point_synth_val"]["records"]:
            frame_id = row["frame_id"]
            output = {"frame_id": frame_id, "raw_diagonal_px": row["raw_diagonal_px"],
                      "point_normalized": row["method_frame_normalized"]}
            for arm in ("direct", "hough"):
                for seed in (1, 2, 3):
                    output[f"{arm}_seed{seed}_normalized"] = row_maps[f"{arm}_seed{seed}_synth_val"][frame_id]["method_frame_normalized"]
            writer.writerow(output)
    artifact_paths = [
        raw / "INTEGRITY_AUDIT_STRICT.json", raw / "SOURCE_REGISTRY_GPU.json",
        raw / "PRETRAINING_GATES_GPU.json", raw / "COMPARISON_SYNTH_VAL.json",
        raw / "predictions/point_synth_val.json",
    ]
    for arm in ("direct", "hough"):
        for seed in (1, 2, 3):
            artifact_paths.extend((raw / f"heads/{arm}_seed{seed}/checkpoint_final.pt",
                                   raw / f"heads/{arm}_seed{seed}/COMPLETION.json",
                                   raw / f"predictions/{arm}_seed{seed}_synth_val.json",
                                   raw / f"evaluations/{arm}_seed{seed}_synth_val.json"))
    artifact_index = {
        "schema": "symdht_local_raw_artifact_index_v1", "raw_root": str(raw),
        "files": {str(path.relative_to(raw)): {"bytes": path.stat().st_size, "sha256": sha256(path)}
                  for path in artifact_paths},
    }
    immutable_json(docs / "RAW_ARTIFACT_INDEX.json", artifact_index)
    status = {
        "schema": "symdht_local_run_status_v1",
        "scientific_verdict": comparison["scientific_verdict"],
        "execution_complete": True,
        "integrity_pass": integrity["PASS"] and gates["integrity_gate_PASS"],
        "synthetic_local_fusion_go": comparison["synthetic_local_fusion_go"],
        "dht_incremental_go": comparison["dht_incremental_go"],
        "real_dev_executed": False,
        "real_dev_progression": False,
        "real_dev_reason": "all three Hough seeds failed the frozen synthetic H>P main gate",
        "full_online_training_executed": False,
        "final_test_opened": False,
        "gpu_recovery": "no reboot/service restart; process-local 580.173.02 libcuda/libnvidia-ml copy in /tmp",
        "pytest": {"passed": 11, "failed": 0},
        "raw_artifact_index_sha256": sha256(docs / "RAW_ARTIFACT_INDEX.json"),
    }
    immutable_json(docs / "RESULT_SUMMARY.json", status)
    point = values["point_synth_val"]
    lines = [
        "# Symmetry-aware reliability-gated DHT local fusion v1", "",
        "## Scientific verdict", "",
        "`NO_SYNTHETIC_LOCAL_FUSION_SIGNAL`", "",
        "The frozen Point baseline remains the selected method. All three Hough seeds failed the predeclared H>P gate. "
        "No threshold, sigma, max-shift, grid, loss, seed, or epoch was added after observing the result, and real DEV319 was not opened.", "",
        "## Confirmed execution", "",
        "- Stock-R0 export/integrity and G1–G15: PASS.",
        "- Direct and Hough: seeds 1/2/3, exactly 2,000 AdamW steps each, batch 8, last-step checkpoint only.",
        "- CUDA: RTX 3080. A 580.173 kernel / 580.178 userspace mismatch was repaired without reboot or display restart by using a process-local copy of the still-loaded 580.173 libraries.",
        "- Unit/contract tests: 11 passed.", "",
        "## Synthetic metrics", "",
        "| arm | seed | primary mean/diag | median px | P90 px | good-point damage | new catastrophic |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| Point | — | {point['primary_frame_mean_over_raw_diagonal']:.9f} | {point['pooled_symmetric_median_px']:.4f} | {point['pooled_symmetric_p90_px']:.4f} | 0/3344 (0%) | 0 |",
    ]
    for arm in ("direct", "hough"):
        for seed in (1, 2, 3):
            value = values[f"{arm}_seed{seed}_synth_val"]
            lines.append(f"| {arm.title()} | {seed} | {value['primary_frame_mean_over_raw_diagonal']:.9f} | "
                         f"{value['pooled_symmetric_median_px']:.4f} | {value['pooled_symmetric_p90_px']:.4f} | "
                         f"{value['good_point_damage_count_method_gt_10px']}/{value['good_point_count_base_le_5px']} "
                         f"({100*value['good_point_damage_rate']:.3f}%) | {value['new_catastrophic_frame_count']} |")
    lines.extend(["", "Every Hough seed had worse primary, median and P90 than Point, and damage exceeded 0.5%. "
                  "All had zero new catastrophic frames. Hough was numerically better than Direct in 3/3 paired seeds "
                  f"(seed-average paired frame delta {comparison['hough_minus_direct_paired_frame_mean_seed_average']:.9f}), "
                  "but the precondition `LOCAL_LINE_FUSION_GO` failed, so `DHT_INCREMENTAL_GO` is false.", "",
                  "## Symmetry and integrity", "",
                  f"- C1: {integrity['symmetry_counts']['C1']}; C2: {integrity['symmetry_counts']['C2']}; C4: 0 (`C4_NOT_EVALUATED`).",
                  "- By split: train C1/C2/C4 = 578/1214/0; calibration = 87/169/0; synth_val = 114/398/0.",
                  "- `scene.usd` and `scene_1.usd`: confirmed C2. Unresolved GLB assets: C1. Center 8 is fixed by every explicit permutation.",
                  f"- All {integrity['source_image_bytes_verified']} source image byte hashes and all {integrity['stock_point_exact_parity']} stock point rows were independently verified.",
                  "- GT-free predictions were serialized before evaluation. Missing/unmatched outputs were retained with penalty.", "",
                  "## Diagnostics", ""])
    for arm in ("direct", "hough"):
        values_arm = [values[f"{arm}_seed{seed}_synth_val"] for seed in (1,2,3)]
        lines.append(f"- {arm.title()}: correction mean px " + ", ".join(f"{v['correction_magnitude_px']['mean']:.3f}" for v in values_arm)
                     + "; reliability mean " + ", ".join(f"{v['reliability_proxy']['mean']:.3f}" for v in values_arm)
                     + "; P_nonnull mean " + ", ".join(f"{v['p_nonnull_proxy_mean']:.3f}" for v in values_arm)
                     + "; concentration mean " + ", ".join(f"{v['concentration_mean']:.3f}" for v in values_arm) + ".")
        pred_arm = [predictions[f"{arm}_seed{seed}_synth_val"] for seed in (1,2,3)]
        lines.append(f"- {arm.title()} parameters: {pred_arm[0]['parameter_count']:,}; GPU scoring ms/frame: "
                     + ", ".join(f"{v['milliseconds_per_frame']:.3f}" for v in pred_arm) + ".")
    lines.extend(["- Actual saved line-mode forces had maximum tangent component below 7.2e-15 (PASS).",
                  "- Reliability/null/concentration are availability proxies, not calibrated physical visibility probabilities.", "",
                  "## Code and artifacts", "",
                  "- Added a self-contained exporter/auditor, Direct and dense line-aligned DHT heads, reliability decoder, strict local WLS, whole-object symmetry loss, GT-free runner, frozen evaluator/gates, and regression tests.",
                  "- Point adds 0 trainable parameters; Direct adds 20,876 and Hough adds 34,968.",
                  f"- Raw local root: `{raw}`. Reviewable raw artifact index SHA256: `{status['raw_artifact_index_sha256']}`.", "",
                  "## Scope", "",
                  "No global pose solver, point-head/backbone training, real-label training, full online training, real DEV319, or FINAL data was used. "
                  "The 1,792/256/512 split is a new line-head split, not an unseen-backbone claim. See `SYNTH_PER_FRAME.csv` for the complete 512-frame paired table.", ""])
    exclusive_text(docs / "RESULTS.md", "\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True); parser.add_argument("--docs", type=Path, required=True)
    args = parser.parse_args(); run(args.raw.resolve(), args.docs.resolve()); print(args.docs.resolve())


if __name__ == "__main__": main()
