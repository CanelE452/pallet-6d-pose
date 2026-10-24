"""Post-forward source GT join for the frozen BEFORE-query intervention."""
from pathlib import Path
import hashlib
import json
import numpy as np

H = Path(__file__).resolve().parent
V = H.parent
P = V.parent
SOURCE = P / "query_visibility_actual_initializer_source_20261024_v132"
CONTROL = V / "same_before_query_control_v1"
OUT = V / "saved_fixed_query_evaluation_v1"


def read(p):
    return json.loads(Path(p).read_text())


def bind(p):
    return {"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}


def check(b):
    assert bind(Path(b["path"])) == b
    return Path(b["path"])


def metric(y, pred, mask):
    cm = np.array([[np.sum(mask & (y == t) & (pred == p)) for p in (0, 1, -1)] for t in (0, 1)], dtype=np.int64)
    ratio = lambda n, d: float(n / d) if d else None
    return {"CM_D_E_rows_D_E_U_columns": cm.tolist(), "D_recall": ratio(cm[0, 0], cm[0].sum()),
            "E_recall": ratio(cm[1, 1], cm[1].sum()), "D_precision": ratio(cm[0, 0], cm[:, 0].sum()),
            "E_precision": ratio(cm[1, 1], cm[:, 1].sum()), "prediction_U_known": int(cm[:, 2].sum()),
            "prediction_U_rate_known": ratio(cm[:, 2].sum(), cm.sum()), "original_rows": int(mask.sum()),
            "prediction_U_all": int(np.sum(mask & (pred == -1))), "known_DE": int(cm.sum())}


def main():
    assert not OUT.exists()
    # No GT is opened until the actual frozen forward is complete.
    seal = read(CONTROL / "PREDICTION_SEAL.json")
    assert seal["completed"] and seal["PASS"] and seal["GT_read"] is False
    assert seal["source_input_intervention"] and seal["actual_query_for_changed_image"] is False
    assert read(check(seal["BEFORE_parity"]))["PASS"]
    es = read(SOURCE / "saved_frozen_visibility_evaluation_v1/EVALUATION_SEAL.json")
    ledger = check(es["scores"])
    rows = [json.loads(line) for line in ledger.read_text().splitlines()]
    assert len(rows) == 6144 and all(r["global_index"] == i for i, r in enumerate(rows))
    ids = np.array([r["query_id"] for r in rows])
    y = np.array([{"D": 0, "E": 1}.get(r["raw_state"], -1) for r in rows])
    state = np.array([r["raw_state"] for r in rows])
    fold = np.array([r["split"] for r in rows])
    variant = np.array([r["variant"] for r in rows])
    boundary = np.array([r["convention_boundary"] for r in rows])
    family = np.array([r["factors"]["foreign_family"] for r in rows])
    original_error = np.array([r["error_px"] if r["error_px"] is not None else np.nan for r in rows])
    refs = np.array([r["reference_xy"] for r in rows], dtype=np.float64)
    original = {"C16": np.array([{"D": 0, "E": 1, "U": -1}[r["C16"]["prediction"]] for r in rows])}
    for model, arm in (("C24", "C24_ZERO_GEOMETRY_CONTROL"), ("C25", "C25_NATIVE_GEOMETRY42")):
        p = P / "query_visibility_native_geometry_control_20261024_v133/runs/C24_C25_MATCHED_G42_FIXED8000" / arm / "FINAL_NATIVE6144_SCORES_INPUT_AND_OUTPUT.npz"
        with np.load(p, allow_pickle=False) as f:
            assert np.array_equal(f["global_index"], np.arange(6144))
            original[model] = f["prediction"].copy()
    fixed = {}
    shared_q = None
    for model in ("C16", "C24", "C25"):
        with np.load(check(seal["models"][model]["predictions"]), allow_pickle=False) as f:
            assert np.array_equal(f["global_index"], np.arange(6144)) and np.array_equal(f["occurrence_id"], ids)
            assert np.array_equal(f["prediction"], np.where(f["query_available"], (f["raw_head_logit_before_U"] >= 0).astype(np.int8), -1))
            fixed[model] = f["prediction"].copy()
            q = f["query_xy"].copy()
        if shared_q is None:
            shared_q = q.copy()
        else:
            assert np.array_equal(shared_q, q, equal_nan=True)
        assert np.array_equal(fixed[model][variant == "BEFORE"], original[model][variant == "BEFORE"])
    frozen_error = np.linalg.norm(q - refs, axis=1)
    groups = {}
    for name in ("ALL", "TRAIN", "VALIDATION"):
        m = np.ones(6144, bool) if name == "ALL" else fold == name
        groups[name] = m
        groups[name + "::historical_area_agreement"] = m & ~boundary
        groups[name + "::actual_error_LE3"] = m & (original_error <= 3)
        groups[name + "::frozen_error_LE3"] = m & (frozen_error <= 3)
        for v in np.unique(variant):
            groups[name + "::" + v] = m & (variant == v)
        for f in np.unique(family):
            groups[name + "::" + f] = m & (family == f)
    results = {}
    for name, mask in groups.items():
        results[name] = {"teacher_states": {s: int(np.sum(mask & (state == s))) for s in "DESU"}, "models": {}}
        for model in fixed:
            common = mask & (original[model] != -1) & (fixed[model] != -1)
            changed = mask & (original[model] != fixed[model])
            results[name]["models"][model] = {
                "actual_query": metric(y, original[model], mask), "fixed_BEFORE_query": metric(y, fixed[model], mask),
                "common_available_actual": metric(y, original[model], common),
                "common_available_fixed": metric(y, fixed[model], common),
                "prediction_changed": int(changed.sum()),
                "D_correct_to_E": int(np.sum(changed & (y == 0) & (original[model] == 0) & (fixed[model] == 1))),
                "D_E_to_correct": int(np.sum(changed & (y == 0) & (original[model] == 1) & (fixed[model] == 0))),
                "E_miss_to_correct": int(np.sum(changed & (y == 1) & (original[model] == 0) & (fixed[model] == 1))),
                "E_correct_to_miss": int(np.sum(changed & (y == 1) & (original[model] == 1) & (fixed[model] == 0)))}
    cases = []
    for base, slot in ((128, 0), (132, 0), (136, 3), (137, 3), (134, 0), (135, 0), (157, 0), (173, 0)):
        for vi in range(4):
            i = base * 32 + vi * 8 + slot
            cases.append({"index": i, "base": base, "slot": slot, "variant": str(variant[i]), "teacher_state": str(state[i]),
                          "actual_error_px": float(original_error[i]) if np.isfinite(original_error[i]) else None,
                          "frozen_error_px": float(frozen_error[i]) if np.isfinite(frozen_error[i]) else None,
                          "actual_prediction": {m: int(original[m][i]) for m in fixed}, "fixed_prediction": {m: int(fixed[m][i]) for m in fixed}})
    OUT.mkdir()
    output = {"completed": True, "PASS": True, "code": bind(Path(__file__)), "frozen_forward": bind(CONTROL / "PREDICTION_SEAL.json"),
              "GT_ledger": bind(ledger), "all6144_preserved": True, "primary_convention": "CF verified against original source training labels",
              "boundary_subgroup_meaning": "Disagreement with obsolete projected-area rule only; no evidence of native-ontology uncertainty for these192 configs.",
              "results": results, "cases": cases, "source_only": True, "threshold_changed": False, "new_model_selected": False,
              "new_fit_private_V112_PnP_or_deployed_query_correction": 0,
              "limitations": ["Source input intervention using actual BEFORE q8/L/context; it is not deployment-time correction or oracle GT recrop.",
                              "Actual versus fixed-input differences change q8/L/context jointly. Fixed-input four-image contrasts isolate RGB changes, not the cause of the initializer's motion.",
                              "Original slots, changed availability, S/U, and both original-error/fixed-error strata are retained."]}
    path = OUT / "SOURCE_FIXED_QUERY_DIAGNOSIS.json"
    with path.open("x") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps({"completed": True, "result": bind(path), "VAL": results["VALIDATION"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
