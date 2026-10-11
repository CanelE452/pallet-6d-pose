"""Read-only B/C input and exact symmetry-supervision audit; never fit or solve PnP."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch


def read(path):
    return json.loads(Path(path).read_text())


def binding(source, relative):
    p = source / relative
    return {"path": relative, "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            "bytes": p.stat().st_size}


def dump(path, value):
    path = Path(path)
    if path.exists():
        raise RuntimeError("Preserve completed artifact: " + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--public-root", type=Path, required=True)
    p.add_argument("--private-output", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    source, public = args.source_root.resolve(), args.public_root.resolve()
    torch.set_num_threads(2)
    sys.path.insert(0, str(source))
    from scripts.research.pallet_n3_completion_v3.model import select_symmetric_target

    start = time.perf_counter()
    src_rel = "data/pallet/results/pallet_line_pose_v1/SOURCE_MANIFEST.json"
    src = read(source / src_rel)["records"]
    partitions = np.array([r["partition"] for r in src])
    by_id = {r["id"]: i for i, r in enumerate(src)}
    assert len(by_id) == len(src) == 60000
    side_rel = "data/pallet/results/pallet_dim_conditioned_p_v1/DIMENSION_SIDECAR.npz"
    side = np.load(source / side_rel, allow_pickle=False)
    assert np.array_equal(side["record_index"], np.arange(len(src)))
    permutations, group_valid, order = (side[k] for k in ("permutations", "group_valid", "order"))
    result = {"source_manifest": binding(source, src_rel), "sidecar": binding(source, side_rel),
              "exact_algorithm": "model.select_symmetric_target on frozen initial points; identity-first exact tie",
              "original_train_groups": dict(Counter("C" + str(int(x)) for x in order[partitions == "train"])),
              "backbones": {}}
    specs = [
        ("YOLO", "data/pallet/results/pallet_line_pose_v1/cache",
         "data/pallet/results/pallet_final_ml_contribution_test_v1/B_line_vs_point/order_seed"),
        ("DOPE", "data/pallet/results/pallet_dope_refiner_20261001_v1/cache",
         "data/pallet/results/pallet_n3_completion_v3/orders/dope_seed"),
        ("ResNet18", "data/pallet/results/pallet_n3_completion_v3/resnet18_constant_cache",
         "data/pallet/results/pallet_n3_completion_v3/orders/resnet18_seed"),
    ]
    for backbone, directory, prefix in specs:
        keys = ("points", "point_valid", "boxes", "gt_points", "gt_valid")
        arrays = {k: np.load(source / directory / (k + ".npy"), mmap_mode="r") for k in keys}
        orders = [np.load(source / (prefix + str(seed) + ".npy")).ravel() for seed in (1, 2, 3)]
        unique = np.unique(orders[0])
        assert all(np.array_equal(unique, np.unique(x)) for x in orders)
        assert (partitions[unique] == "train").all()
        branch_by_row = np.full(len(src), -1, np.int64)
        ties = np.zeros(len(unique), bool)
        changed_target = np.zeros(len(unique), bool)
        valid_count = np.zeros(len(unique), np.int64)
        for begin in range(0, len(unique), 4096):
            rows = unique[begin:begin + 4096]
            batch = {k: torch.from_numpy(np.array(v[rows], copy=True)) for k, v in arrays.items()}
            diagonal = torch.linalg.vector_norm(batch["boxes"][:, 2:] - batch["boxes"][:, :2], dim=-1).clamp_min(1)
            gt, valid, branch, cost = select_symmetric_target(
                batch["points"], batch["point_valid"], batch["gt_points"], batch["gt_valid"],
                torch.from_numpy(permutations[rows].copy()), torch.from_numpy(group_valid[rows].copy()), diagonal)
            branch_by_row[rows] = branch.numpy()
            costs = cost.numpy()
            ties[begin:begin + len(rows)] = ((costs == costs[:, :1]) & group_valid[rows]).sum(-1) > 1
            original = batch["gt_points"].numpy()
            same = ((gt.numpy() == original) | (np.isnan(gt.numpy()) & np.isnan(original))).all((-2, -1))
            same &= (valid.numpy() == batch["gt_valid"].numpy()).all(-1)
            changed_target[begin:begin + len(rows)] = ~same
            valid_count[begin:begin + len(rows)] = valid[:, :8].sum(-1).numpy()
        groups = {}
        for g in (1, 2, 4):
            mask = order[unique] == g
            groups["C" + str(g)] = {"total": int(mask.sum()),
                "nonidentity": int((branch_by_row[unique[mask]] != 0).sum()),
                "identity_exact_cost_tie": int(ties[mask].sum()),
                "changed_target_tuple": int(changed_target[mask].sum())}
        result["backbones"][backbone] = {
            "unique_usable_train": len(unique), "nonidentity": int((branch_by_row[unique] != 0).sum()),
            "nonidentity_fraction": float((branch_by_row[unique] != 0).mean()), "groups": groups,
            "optimizer_exposures": {str(seed): {"total": len(o), "nonidentity": int((branch_by_row[o] != 0).sum()),
                "order": binding(source, prefix + str(seed) + ".npy")} for seed, o in enumerate(orders, 1)},
            "selected_valid_corner_counts": {str(k): int(v) for k, v in Counter(map(int, valid_count)).items()},
            "arrays": [binding(source, directory + "/" + k + ".npy") for k in keys],
        }
    result["elapsed_seconds"] = time.perf_counter() - start
    result["new_PnP_calls"] = result["optimizer_updates"] = 0
    result["verdict"] = "LOW_SUPERVISION_INTERVENTION_RATE"
    dump(args.output / "B2_SYMMETRY.json", result)

    split_rel = "_docs/experiments/pallet_selector_recovery_v1/stage2_synth_scorer/SYNTHETIC_SPLIT_LOCK.json"
    split = read(source / split_rel)
    old_rows = read(source / split["records"]["path"])
    selected = []
    old_train_dropped = [r["id"] for r in old_rows if r["split"] == "TRAIN" and src[by_id[r["id"]]]["partition"] != "train"]
    def renderer_group(r):
        return "G38_merged_archive" if r["source"] == "G38" else "P0_TEX_" + r["scenario_id"].split(":")[1].rsplit("_f", 1)[0]
    eligible = [r for r in src if r["partition"] == "train" and renderer_group(r) in split["groups"]["TRAIN"]]
    eligible.sort(key=lambda r: hashlib.sha256(("20261011|" + r["id"]).encode()).hexdigest())
    assert len(eligible) >= 4096
    for r in eligible[:4096]:
        selected.append({"id": r["id"], "record_index": by_id[r["id"]], "split": "TRAIN", "renderer_group": renderer_group(r),
            "original_partition": r["partition"], "image_sha256": r["image_sha256"]})
    for r in old_rows:
        if r["split"] == "TRAIN":
            continue
        i = by_id[r["id"]]
        selected.append({"id": r["id"], "record_index": i, "split": r["split"], "renderer_group": r["group"],
            "original_partition": src[i]["partition"], "image_sha256": src[i]["image_sha256"]})
    group_parts = {}
    scenario_parts = {}
    sha_parts = {}
    for r in selected:
        for mapping, key in ((group_parts, r["renderer_group"]),
                             (scenario_parts, src[r["record_index"]]["scenario_id"]),
                             (sha_parts, r["image_sha256"])):
            assert key not in mapping or mapping[key] == r["split"]
            mapping[key] = r["split"]
    manifest = read(source / "data/pallet/results/pallet_line_pose_v1/cache/CACHE_MANIFEST.json")
    normalization = read(public / "_docs/experiments/pallet_dim_conditioned_p_v1/DIM_NORMALIZATION_LOCK.json")
    norm_features = (np.log(side["dimensions"][partitions == "train"]))
    mean_dim = side["dimensions"][partitions == "train"].mean(0)
    mean_log = norm_features.mean(0)
    mean_feature = np.r_[np.log(mean_dim), np.log(mean_dim[0] / mean_dim[1]),
                         np.log(mean_dim[2] / np.sqrt(mean_dim[0] * mean_dim[1]))]
    contract = {
        "status": "SOURCE_CONTRACT_VERIFIED_CURRENT_N3_FEATURES_PENDING",
        "frozen_detector_sha256": "970a0913b38ed4c9e3662837abccbf9d91b8b0858deafae854c1055e477644f7",
        "counts": dict(Counter(r["split"] for r in selected)), "source_split": binding(source, split_rel),
        "TRAIN_rule": "Original paper TRAIN and historical classifier TRAIN-assigned renderer groups only; SHA256('20261011|'+frame ID) ascending first4096, budget fixed before fitting",
        "TRAIN_original_nontrain_removed": len(old_train_dropped), "TRAIN_original_partition_only": True,
        "VAL_TEST_original_group_assignment_preserved": True,
        "original_partition_exposure_counts": {"|".join(k): int(v) for k, v in Counter((r["split"], r["original_partition"]) for r in selected).items()},
        "no_renderer_group_or_scenario_or_image_sha_crossing": True,
        "old_feature_cache_reusable": False,
        "old_feature_reason": "Old S0/S1 checkpoint hashes 1dae620f../1f018068.. differ from current frozen R0; old448dim global average is not predicted-ROI2x2 spatial pooling",
        "current_cache": {"path": "data/pallet/results/pallet_line_pose_v1/cache", "arrays": manifest["arrays"],
                          "feature_recipe": manifest["feature_recipe"], "input_recipe": manifest["input_recipe"]},
        "V": {"P3": {"channels": 64, "stride": 8}, "P4": {"channels": 128, "stride": 16},
              "pool": "2x2 equal-area spatial means inside predicted box and actual input bounds", "dimensions": 768},
        "D": {"input_order": "canonical W,D,H = fixed XYZ[0,2,1]", "features": normalization["input"],
              "normalization": "existing original TRAIN-only DIM_NORMALIZATION_LOCK", "mean_dimensions_WDH_m": mean_dim.tolist(),
              "mean_dimension_features_minus_mean_log_features": (mean_feature - np.asarray(normalization["mean"])).tolist(),
              "mean_dimensions_is_normalized_zero": bool(np.allclose(mean_feature, normalization["mean"], rtol=0, atol=1e-10))},
        "label": "y_long = norm(Xcf[1]-Xcf[0]) > norm(Xcf[4]-Xcf[0]); exact renderer only; squares/unclear remain excluded from fitting",
        "candidate_order": ["long-face-front", "short-face-front"],
        "existing_label_conversion": "old SYNTH_LABELS.parity==0 means y_long==1; never infer label from binary field name",
        "camera_facing_dimensions_correction": "19345/40000 G38 rows changed; corrected before any main dimension-conditioned optimizer update",
        "new_high_cost_raw_RGB_features_required": 0,
        "GPU_head_forward_needed": len(selected),
        "G_new_solver_call_upper_bound": 4 * len(selected),
        "G_note": "Use current Two-stage seed1 points for geometry; old S0/S1/PoseFix geometry cannot be renamed. Official two-candidate selector only; each hypothesis one SQPnP+RefineLM",
        "fitting_started": False, "real_evaluation_started": False,
    }
    dump(args.private_output / "C_SYNTH_RECORDS.json", selected)
    dump(args.output / "BC_INPUT_CONTRACT.json", contract)
    print(json.dumps({"B2": {b: {k: v[k] for k in ("unique_usable_train", "nonidentity", "nonidentity_fraction")} for b, v in result["backbones"].items()},
                      "C_counts": contract["counts"], "C_new_PnP_upper_bound": contract["G_new_solver_call_upper_bound"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
