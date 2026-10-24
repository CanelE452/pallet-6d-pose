"""Saved scores only. Export no candidate cut; never load pixels or models."""
from pathlib import Path
import hashlib
import json
import numpy as np

H = Path(__file__).resolve().parent
P = H.parents[1]
V132 = P / "query_visibility_actual_initializer_source_20261024_v132"
V133 = P / "query_visibility_native_geometry_control_20261024_v133"


def read(path):
    return json.loads(Path(path).read_text())


def bind(path):
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def metric(raw, available, target, mask):
    original_d = int(np.sum(mask & (target == 0)))
    original_e = int(np.sum(mask & (target == 1)))
    selected = mask & available & (target >= 0)
    z, y = raw[selected], target[selected]
    assert np.isfinite(z).all() and original_d > 0 and original_e > 0
    ds, es = z[y == 0], z[y == 1]
    auc = float(np.mean((es[:, None] > ds).astype(float) + .5 * (es[:, None] == ds))) if len(ds) and len(es) else None
    d_correct, e_correct = 0, len(es)
    best = (min(d_correct / original_d, e_correct / original_e), d_correct, e_correct)
    max_e = {"0.98": None, "0.99": None}

    def observe():
        nonlocal best
        value = min(d_correct / original_d, e_correct / original_e)
        if value > best[0]:
            best = (value, d_correct, e_correct)
        for requirement in max_e:
            if d_correct / original_d >= float(requirement):
                recall = e_correct / original_e
                if max_e[requirement] is None or recall > max_e[requirement]["E_recall"]:
                    max_e[requirement] = {"E_recall": recall, "D_correct": d_correct, "D_original": original_d,
                                          "E_correct": e_correct, "E_original": original_e}

    observe()
    order = np.argsort(z, kind="stable")
    zs, ys = z[order], y[order]
    starts = np.r_[0, np.flatnonzero(zs[1:] != zs[:-1]) + 1, len(zs)]
    for lo, hi in zip(starts[:-1], starts[1:]):
        d_correct += int(np.sum(ys[lo:hi] == 0))
        e_correct -= int(np.sum(ys[lo:hi] == 1))
        observe()
    pred = np.where(available, (raw >= 0).astype(np.int8), -1)
    cm = [[int(np.sum(mask & (target == t) & (pred == p))) for p in (0, 1, -1)] for t in (0, 1)]
    return {"D_original": original_d, "E_original": original_e, "D_available": len(ds), "E_available": len(es),
            "available_E_positive_AUC": auc, "fixed_zero_CM_D_E_rows_D_E_U_columns": cm,
            "posthoc_best_min_full_class_recall": best[0], "posthoc_best_D_correct": best[1], "posthoc_best_E_correct": best[2],
            "maximum_E_at_D_requirement_diagnostic_only": max_e,
            "threshold_values_exported_or_applied": False}


def main():
    # A tied positive/negative pair cannot be separated by any scalar cut.
    fixture = metric(np.array([0., 0., 1., np.nan]), np.array([True, True, True, False]),
                     np.array([0, 1, 1, 0]), np.ones(4, bool))
    assert fixture["posthoc_best_min_full_class_recall"] == .5
    assert fixture["maximum_E_at_D_requirement_diagnostic_only"]["0.98"] is None
    policy = read(H / "POLICY_BEFORE_SAVED_SCORES.json")
    assert policy["threshold_values_exported_or_applied"] is False
    seal = read(V132 / "saved_frozen_visibility_evaluation_v1/EVALUATION_SEAL.json")
    ledger = Path(seal["scores"]["path"])
    assert bind(ledger) == seal["scores"]
    rows = [json.loads(line) for line in ledger.read_text().splitlines()]
    assert len(rows) == 6144 and all(r["global_index"] == i for i, r in enumerate(rows))
    target = np.array([{"D": 0, "E": 1}.get(r["raw_state"], -1) for r in rows])
    split = np.array([r["split"] for r in rows])
    boundary = np.array([r["convention_boundary"] for r in rows])
    error = np.array([r["error_px"] if r["error_px"] is not None else np.nan for r in rows])
    same_material = np.array([r["factors"]["foreign_family"] == "SAME_TARGET_MATERIAL" for r in rows])
    groups = {"TRAIN": split == "TRAIN", "VALIDATION": split == "VALIDATION",
              "VALIDATION_NONBOUNDARY": (split == "VALIDATION") & ~boundary,
              "VALIDATION_LE3": (split == "VALIDATION") & (error <= 3),
              "VALIDATION_NONBOUNDARY_SAME_MATERIAL": (split == "VALIDATION") & ~boundary & same_material}
    models = {"C16": (np.array([r["C16"]["raw_logit"] if r["C16"]["raw_logit"] is not None else np.nan for r in rows]),
                       np.array([r["C16"]["available"] for r in rows]))}
    inputs = [bind(ledger)]
    for short, arm in (("C24", "C24_ZERO_GEOMETRY_CONTROL"), ("C25", "C25_NATIVE_GEOMETRY42")):
        directory = V133 / "runs/C24_C25_MATCHED_G42_FIXED8000" / arm
        path = directory / "FINAL_NATIVE6144_SCORES_INPUT_AND_OUTPUT.npz"
        with np.load(path, allow_pickle=False) as packet:
            assert np.array_equal(packet["global_index"], np.arange(6144))
            assert np.array_equal(packet["external_target_CF"], target)
            assert np.array_equal(packet["split"], split)
            models[short] = (packet["raw_head_logit_before_U"].copy(), packet["query_available"].copy())
        inputs.append(bind(path))
    result = {name: {model: metric(raw, available, target, mask) for model, (raw, available) in models.items()}
              for name, mask in groups.items()}
    output = {"completed": True, "PASS": True, "previous_goal_turn_classification": "PROGRESS",
              "previous_turn_evidence": "V133 executed two matched8000 source heads, both failed admission; geometry alone insufficient.",
              "code": bind(Path(__file__)), "policy": bind(H / "POLICY_BEFORE_SAVED_SCORES.json"), "inputs": inputs,
              "groups": result, "teacher_S_U_outside_external_task": True, "all6144_rows_retained": True,
              "new_private_RGB_models_checkpoints_fit_PnP": 0, "threshold_values_exported_or_applied": False,
              "new_model_or_threshold_selection": False}
    with (H / "RANKING_DIAGNOSIS_NO_THRESHOLD_EXPORT.json").open("x") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
    for group, metrics in result.items():
        print(group, {m: {k: x[k] for k in ("available_E_positive_AUC", "posthoc_best_min_full_class_recall", "maximum_E_at_D_requirement_diagnostic_only")}
                      for m, x in metrics.items()})


if __name__ == "__main__":
    main()
