"""Saved-only TRAIN causal diagnostic; no model calls or development selection.

Targets are read only after the actual frozen prediction receipt is verified.
The anchor is the original COVER initializer query, not a production replay.
"""
from pathlib import Path
import argparse, hashlib, json
import numpy as np

HERE = Path(__file__).resolve().parent
P = HERE.parent
VARIANTS = ("BEFORE", "COVER", "NEAR_SHAM", "FAR_SHAM")
SUPPORT = ("visible_end", "foreign_front")


def require(ok, msg):
    if not bool(ok):
        raise AssertionError(msg)


def bind(path):
    path = Path(path).resolve()
    require(path.is_relative_to(P), "Project source only")
    require(not any(x in str(path).lower() for x in ("_v112", "_v177", "_v178", "_v169", "current223", "private_evaluation", "synthetic_val")), "Forbidden source")
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return {"path": str(path), "sha256": h.hexdigest()}


def checked(b):
    require(bind(b["path"]) == b, "Binding drift: " + b["path"])
    return Path(b["path"])


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def write(path, value):
    with Path(path).open("x") as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")


def load_npz(b):
    with np.load(checked(b), allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def same(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def fraction(a, b):
    return None if b == 0 else int(a) / int(b)


def metrics(truth, pred, mask):
    require(truth.shape == pred.shape == mask.shape, "Metric shape")
    require(np.isin(truth, [-1, 0, 1]).all() and np.isin(pred, [-1, 0, 1]).all(), "Binary + UNKNOWN only")
    known = mask & (truth >= 0)
    cm = np.zeros((2, 3), np.int64)
    for t, p in zip(truth[known], pred[known]):
        cm[t, 2 if p == -1 else p] += 1
    tn, fp, u0 = map(int, cm[0])
    fn, tp, u1 = map(int, cm[1])
    unknown = mask & (truth < 0)
    return dict(rows=int(mask.sum()), known=int(known.sum()), unknown=int(unknown.sum()),
                CM_NO_YES_by_NO_YES_U=cm.tolist(), positive_precision=fraction(tp, tp + fp),
                positive_recall_including_U=fraction(tp, tp + fn + u1),
                known_prediction_U=int(u0 + u1), known_U_rate=fraction(u0 + u1, known.sum()),
                unknown_truth_prediction_counts_NO_YES_U=[int((pred[unknown] == k).sum()) for k in (0, 1, -1)])


def pair_summary(truth, pred, logits, left, right, available):
    """All1024 matched base-native pairs, including UNKNOWN/U in separate cells."""
    result = {}
    for j, name in enumerate(SUPPORT):
        ta, tb = truth[left, j], truth[right, j]
        pa, pb = pred[left, j], pred[right, j]
        av = available[left] & available[right]
        cells = {}
        for aa in (-1, 0, 1):
            for bb in (-1, 0, 1):
                select = (ta == aa) & (tb == bb)
                known_available = select & av
                response = np.zeros((3, 3), np.int64)
                for xa, xb in zip(pa[select], pb[select]):
                    response[2 if xa == -1 else xa, 2 if xb == -1 else xb] += 1
                finite = select & np.isfinite(logits[left, j]) & np.isfinite(logits[right, j])
                delta = logits[right[finite], j] - logits[left[finite], j]
                cells[f"{aa}_to_{bb}"] = dict(pairs=int(select.sum()), both_input_available=int(known_available.sum()),
                    prediction_transition_NO_YES_U_by_NO_YES_U=response.tolist(),
                    both_predictions_correct=(int((known_available & (pa == aa) & (pb == bb)).sum()) if aa >= 0 and bb >= 0 else None),
                    right_minus_left_logit=dict(n=int(len(delta)), mean=None if not len(delta) else float(delta.mean()),
                        median=None if not len(delta) else float(np.median(delta)),
                        positive_count=int((delta > 0).sum()), zero_count=int((delta == 0).sum()), negative_count=int((delta < 0).sum())))
        result[name] = cells
    all_known = (truth[left] >= 0).all(1) & (truth[right] >= 0).all(1)
    canonical = all_known & (truth[left] == [1, 0]).all(1) & (truth[right] == [0, 1]).all(1)
    bothav = available[left] & available[right]
    correct = (pred[left] == truth[left]).all(1) & (pred[right] == truth[right]).all(1)
    result["joint_visible_only_to_foreign_only"] = dict(pairs=int(canonical.sum()),
        both_available=int((canonical & bothav).sum()), both_supports_correct_at_both_images=int((canonical & bothav & correct).sum()),
        all_other_or_unknown_pairs=int((~canonical).sum()), not_a_success_gate=True)
    return result


def self_test():
    t = np.array([0, 0, 1, 1, 1, -1], np.int8)
    p = np.array([0, 1, 1, 0, -1, 1], np.int8)
    m = metrics(t, p, np.ones(6, bool))
    require(m["CM_NO_YES_by_NO_YES_U"] == [[1, 1, 0], [1, 1, 1]], "CM includes prediction U")
    require(m["positive_precision"] == .5 and m["positive_recall_including_U"] == 1 / 3, "P/R denominators")
    require(m["unknown_truth_prediction_counts_NO_YES_U"] == [0, 1, 0], "Unknown not a negative")
    tt = np.array([[1, 0], [0, 1], [-1, -1], [-1, -1]], np.int8)
    pp = np.array([[1, 0], [0, 1], [-1, -1], [-1, -1]], np.int8)
    zz = np.array([[2., -2.], [-3., 3.], [np.nan, np.nan], [np.nan, np.nan]])
    pair = pair_summary(tt, pp, zz, np.array([0, 2]), np.array([1, 3]), np.array([1, 1, 0, 0], bool))
    require(pair["joint_visible_only_to_foreign_only"]["both_supports_correct_at_both_images"] == 1, "Joint paired correctness")
    require(pair["foreign_front"]["0_to_1"]["right_minus_left_logit"]["mean"] == 5., "Paired delta sign")
    require(pair["foreign_front"]["-1_to_-1"]["both_predictions_correct"] is None, "Unknown transitions unscored")
    return dict(PASS=True, meaningful_checks=6, source_pixels_or_models_read=0)


def run(entry_path):
    e = read(entry_path)
    require(e.get("approved_saved_only_evaluation") is True, "Root saved-only entry")
    require(e["code"] == bind(__file__), "Frozen evaluator")
    for k in ("scope", "raw_complete", "prediction_complete", "self_checks"):
        checked(e[k])
    require(read(e["self_checks"]["path"])["PASS"], "Invented meaningful checks")
    frozen = read(e["prediction_complete"]["path"])
    require(frozen["PASS"] and frozen["completed"], "Closed predictions before any target access")
    require(frozen.get("new_training_updates", 0) == 0, "Frozen inference only")
    require(len(e["arm_outputs"]) == 2 and set(e["arm_outputs"]) == set(frozen["arms"]), "Both closed frozen arms")
    for arm, output in e["arm_outputs"].items():
        require(output == frozen["arms"][arm]["outputs"], "Output belongs to closed prediction receipt")
    source = read(e["raw_complete"]["path"])
    require(source["PASS"] and source["original_COVERanchors1024_byteparity"], "Raw parity admission")
    metadata = rows(checked(source["metadata"]))
    require(len(metadata) == 4096, "All rows retained")
    # Numeric inputs are needed solely to certify q/L and row identity; no model is imported.
    with np.load(checked(source["inputs"]), allow_pickle=False) as z:
        identity = {k: z[k].copy() for k in ("row_id", "original_global_index", "native_slot", "query_xy", "L", "original_query_available")}
    av = identity["original_query_available"]
    require(int((~av).sum()) == 96, "Cloned original anchor U96")
    require(identity["original_global_index"].tolist() == list(range(4096)), "Diagnostic IDs")
    for i, row in enumerate(metadata):
        require((row["base_index"], row["variant"], row["native_slot"]) ==
                (i // 32, VARIANTS[(i % 32) // 8], i % 8), "Frozen base/variant/native row order")
    cube = np.arange(4096).reshape(128, 4, 8)
    for base in range(128):
        for vi in range(4):
            ids, anchor = cube[base, vi], cube[base, 1]
            for key in ("query_xy", "L", "native_slot", "original_query_available"):
                require(same(identity[key][ids], identity[key][anchor]), "Same anchor " + key)
    outputs = {}
    for arm, b in e["arm_outputs"].items():
        a = load_npz(b)
        for key in identity:
            require(same(a[key], identity[key]), "Prediction identity " + arm + ":" + key)
        require(a["raw_support_logits"].shape == (4096, 2) and a["raw_owner_logits"].shape == (4096, 3), "Output shapes")
        require(np.isnan(a["raw_support_logits"][~av]).all() and np.isnan(a["raw_owner_logits"][~av]).all(), "Original U remains NaN")
        require(np.isfinite(a["raw_support_logits"][av]).all() and np.isfinite(a["raw_owner_logits"][av]).all(), "Available finite")
        expected = np.full((4096, 2), -1, np.int8)
        expected[av] = (a["raw_support_logits"][av] > 0).astype(np.int8)
        require(np.array_equal(expected, a["support_prediction"]), "Unchanged support decoder")
        outputs[arm] = a
    # Both arms have been loaded, identity-checked and closed. Only now read diagnostic GT.
    targets = load_npz(source["targets"])
    truth = targets["support_target"]
    require(truth.shape == (4096, 2) and np.isin(truth, [-1, 0, 1]).all(), "Separate fixed targets")
    require((truth[~av] == -1).all(), "Missing anchor truth UNKNOWN")
    groups = {"ALL": np.ones(4096, bool)}
    for variant in VARIANTS:
        groups["variant:" + variant] = np.array([r["variant"] == variant for r in metadata])
    for key in ("geometry_family", "material", "aspect", "foreign_family"):
        vals = [r["source_image_original_metadata"]["factors"][key] for r in metadata]
        for value in sorted(set(vals)):
            groups[key + ":" + value] = np.array([x == value for x in vals])
    for native in range(8):
        groups[f"native:{native}"] = identity["native_slot"] == native
    result = {}
    left = cube[:, 0, :].reshape(-1)
    for arm, a in outputs.items():
        pred, logits = a["support_prediction"], a["raw_support_logits"]
        strata = {name: {sup: metrics(truth[:, j], pred[:, j], mask) for j, sup in enumerate(SUPPORT)} for name, mask in groups.items()}
        paired = {"BEFORE_to_" + variant: pair_summary(truth, pred, logits, left, cube[:, vi, :].reshape(-1), av) for vi, variant in enumerate(VARIANTS) if vi}
        result[arm] = dict(strata=strata, paired=paired)
    out = Path(e["output_dir"]).resolve()
    require(out.parent == HERE and not out.exists(), "Fresh phase output")
    out.mkdir()
    rp = out / "FIXED_SAME_QUERY_METRICS_AND_PAIRED_RESPONSES.json"
    write(rp, dict(completed=True, PASS=True, schema="V182_SAVED_TRAIN_COMMON_COVER_QUERY_DIAGNOSTIC_v1", arms=result,
                   support_order=list(SUPPORT), query_authority="Original COVER inferred q/L/flags repeated on four saved TRAIN images. Separate diagnostic, not original production replay.",
                   training_data_only=True, heldout_performance=False, physical_native_or_real_transfer_gold=False, no_model_or_threshold_choice=True))
    ledger = out / "ALL4096_PAIRED_DIAGNOSTIC_ROWS.jsonl"
    with ledger.open("x") as f:
        for i, r in enumerate(metadata):
            line = {k: r[k] for k in ("row_id", "diagnostic_row_index", "base_index", "variant", "native_slot", "query_xy", "L", "source_image_original_global_index", "anchor_original_global_index", "original_query_available")}
            line["support_truth"] = truth[i].tolist()
            line["responses"] = {arm: dict(support_prediction=a["support_prediction"][i].tolist(), raw_support_logits=[float(x) if np.isfinite(x) else None for x in a["raw_support_logits"][i]], owner_prediction=int(a["owner_prediction"][i])) for arm, a in outputs.items()}
            f.write(json.dumps(line, ensure_ascii=False, allow_nan=False) + "\n")
    for k in ("raw_complete", "prediction_complete", "code", "scope", "self_checks"):
        checked(e[k])
    for b in e["arm_outputs"].values():
        checked(b)
    write(out / "COMPLETE.json", dict(completed=True, PASS=True, entry=bind(entry_path), code=bind(__file__), raw_complete=e["raw_complete"], prediction_complete=e["prediction_complete"],
        metrics=bind(rp), rows=bind(ledger), all4096_retained=True, known_unknown_and_U_separate=True, new_model_forward_training_or_threshold_changes=0,
        source_TRAIN_diagnostic_only=True, fit_admission=False, selected_model=None, fully_solved=False))
    print(json.dumps(dict(completed=True, PASS=True, complete=bind(out / "COMPLETE.json"))))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--self-test", action="store_true")
    g.add_argument("--entry", type=Path)
    args = ap.parse_args()
    if args.self_test:
        print(json.dumps(self_test()))
    else:
        run(args.entry)
