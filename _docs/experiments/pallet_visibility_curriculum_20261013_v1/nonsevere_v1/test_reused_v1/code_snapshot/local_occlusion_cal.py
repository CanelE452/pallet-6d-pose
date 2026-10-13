"""CAL-only diagnosis of a frozen source local-support head.

The source head has a local rendered-body proxy target. Human real CAL labels
are opened only after a new prediction-only feature cache is sealed. This
experiment does not modify observations, initial self-occlusion or live API.
"""
from __future__ import annotations

import argparse
import builtins
from collections import Counter
from contextlib import contextmanager
import io
from pathlib import Path
import time

import cv2
import numpy as np
import torch

from . import appearance as A
from . import common as C
from .calibration import features, SCALARS, VECTORS
from .local_occlusion import infer_local_support


BASE_NAMES = (["q_rgb_" + name for name in SCALARS]
              + ["q_rgb_" + name + "_" + channel for name in VECTORS for channel in ("L", "a", "b")]
              + ["q_rgb_edge_distance_over_patch_radius"])
LOCAL_NAMES = ["q_source_support_probability", "q_source_support_entropy"]
PROBE_NAMES = ["projected_source_support_probability", "projected_source_support_entropy",
               "q_minus_projected_source_support", "normalized_initial_projection_residual"]
FEATURE_SETS = {
    "RGB26": BASE_NAMES,
    "RGB26_SOURCE_LOCAL2": BASE_NAMES + LOCAL_NAMES,
    "RGB26_SOURCE_LOCAL_PROBE6": BASE_NAMES + LOCAL_NAMES + PROBE_NAMES,
}
STRENGTHS = (.01, .1, 1., 10.)
THRESHOLDS = tuple(np.round(np.arange(.2, .81, .05), 8))


def binding(path):
    path = Path(path)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=C.sha(path))


@contextmanager
def no_scoring_reads():
    originals = (builtins.open, io.open, Path.open)
    blocked = ("SCORING_METADATA", "ORACLE_CAL_INPUTS", "TARGETS.json", "STATIC_VISIBILITY", "DATA_CONTRACT", "test_v1", "FINAL_")
    def wrap(function):
        def guarded(path, *args, **kwargs):
            if any(token in str(path) for token in blocked):
                raise AssertionError("Prediction-only phase attempted a scoring read: " + str(path))
            return function(path, *args, **kwargs)
        return guarded
    builtins.open, io.open, Path.open = map(wrap, originals)
    try:
        yield
    finally:
        builtins.open, io.open, Path.open = originals


def infer(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    frames = C.read(args.inputs)["frames"]
    if len(frames) != 76 or any(r["partition"] != "CAL" or r["population_role"] != "DEV" for r in frames):
        raise AssertionError("Exact isolated CAL76 required")
    source_completion = C.read(Path(args.weights).parent / "COMPLETION.json")
    if source_completion["steps"] != 1024 or not source_completion["complete"]:
        raise AssertionError("Source head must finish its fixed protocol before real diagnosis")
    selected = next(row for row in source_completion["weights"] if row["path"] == str(Path(args.weights)))
    if binding(args.weights) != selected:
        raise AssertionError("Frozen source weights binding differs")
    code = Path(__file__).parent
    dependencies = [Path(__file__), args.inputs, args.weights] + [code / name for name in
        ("local_occlusion.py", "appearance.py", "calibration.py", "common.py")]
    protocol = dict(schema="frozen_source_local_support_real_CAL_v1", frames=76,
        source_head=binding(args.weights), source_steps=1024, source_real_label_training_reads=0,
        source_label_scope="local rendered-body support proxy with synthetic opacity, not physical3D visibility",
        feature_sets=FEATURE_SETS, strengths=list(STRENGTHS), thresholds=list(THRESHOLDS),
        folds="LeaveOneGroupOut over five CAL sessions", calibration_head="StandardScaler plus balanced logistic",
        selection="Best pooled LOGO AUC within each declared family; fixed threshold grid by pooled crossfit balanced accuracy",
        clear_gate="AUC>=baseline+.02, external recall>=baseline, direct recall>=baseline-.02, paired session-bootstrap AUC gain lower95CI>0",
        target="Human DIRECT_VISIBLE versus EXTERNAL_OCCLUDED; coordinate accuracy is a separate event",
        prior_context="Earlier CAL and root TEST summaries motivated this source-head pilot; later reused TEST is not independent",
        inference_before_CAL_label_read=True, new_TEST_reads=0, FINAL_reads=0,
        frozen_H_unchanged=True, fit_points_unchanged=True, initial_projection_extra_RGB_probes_only=True,
        absolute_coordinates_as_features=False, IDs_sessions_material_as_features=False,
        baseline_historical_soft_fields="Exact existing RGB26; absent soft_*_color_support remain two zero columns",
        inference_device=args.device, bindings=[binding(path) for path in dependencies])
    C.write(out / "PROTOCOL.json", protocol)
    cv2.setNumThreads(1); torch.set_num_threads(1)
    start = time.monotonic()
    records = []
    with no_scoring_reads():
        for index, frame in enumerate(frames):
            image_path = C.ROOT / frame["image"]
            if C.sha(image_path) != frame["image_sha256"]:
                raise AssertionError("Frozen original RGB changed")
            image = cv2.imread(str(image_path))
            if image is None:
                raise FileNotFoundError(str(image_path))
            q = np.asarray(frame["points"]["N3_SUBPIX"], float)[:8]
            initial = frame["initial_poses"]["N3_SUBPIX"]
            xyz = C.cuboid(*initial["cf_extents"])
            R, t = np.asarray(initial["R_cf"]), np.asarray(initial["centroid"])
            H = C.self_hidden(xyz, R, t)
            projected = C.project(xyz, R, t, np.asarray(frame["K"]))
            raw = A.infer_appearance(image, q, H)
            query = infer_local_support(image, q, H, args.weights, args.device)
            probe = infer_local_support(image, projected, H, args.weights, args.device)
            base = features(raw)
            radius = max(1., raw["diagnostics"]["patch_radius_px"])
            values = {name: [] for name in FEATURE_SETS}
            for k in range(8):
                local = [query["local_support_score"][k], query["entropy"][k]]
                residual = np.linalg.norm(q[k] - projected[k]) / radius
                residual = min(8., float(residual)) if np.isfinite(residual) else 8.
                extra = [probe["local_support_score"][k], probe["entropy"][k],
                         local[0] - probe["local_support_score"][k], residual]
                values["RGB26"].append(base[k].tolist())
                values["RGB26_SOURCE_LOCAL2"].append(base[k].tolist() + local)
                values["RGB26_SOURCE_LOCAL_PROBE6"].append(base[k].tolist() + local + extra)
            for name, value in values.items():
                if np.asarray(value).shape != (8, len(FEATURE_SETS[name])) or not np.isfinite(value).all():
                    raise AssertionError("Feature schema/finite values failed: " + name)
            valid = np.isfinite(q).all(1) & ~(q == -1).all(1)
            inside = valid & (q[:, 0] >= 0) & (q[:, 0] < image.shape[1]) & (q[:, 1] >= 0) & (q[:, 1] < image.shape[0])
            records.append(dict(id=frame["id"], session=frame["session"], partition="CAL",
                arm="N3_SUBPIX", frozen_q=q.tolist(), frozen_self_hidden=H.tolist(),
                native_coordinate_valid=valid.tolist(), native_coordinate_in_frame=inside.tolist(),
                projected_extra_probe_xy=projected.tolist(), features=values,
                frozen_source_query=query, frozen_source_projected_probe=probe,
                source_image_sha256=frame["image_sha256"], fit_points_changed=False))
            if (index + 1) % 16 == 0:
                print("LOCAL_HEAD_CAL_FEATURES", index + 1, round(time.monotonic() - start, 2), flush=True)
    if [binding(path) for path in dependencies] != protocol["bindings"]:
        raise AssertionError("Frozen inference dependency changed")
    C.save_rows(out / "PREDICTION_FEATURES.jsonl.gz", records)
    C.write(out / "FEATURES_SEAL.json", dict(complete=True, frames=76,
        feature_cache=binding(out / "PREDICTION_FEATURES.jsonl.gz"), protocol=binding(out / "PROTOCOL.json"),
        human_scoring_reads_before_seal=0, new_TEST_reads=0, FINAL_reads=0,
        source_head=binding(args.weights), new_PnP_calls=0, initial_projection_calls=76,
        source_head_patch_queries=76 * 8 * 2, elapsed_seconds=time.monotonic() - start))


def confusion(y, p, threshold):
    positive = p >= threshold
    visible, external = y == 1, y == 0
    vp = int((visible & positive).sum()); vn = int((visible & ~positive).sum())
    ep = int((external & positive).sum()); en = int((external & ~positive).sum())
    vr = vp / int(visible.sum()) if visible.any() else None
    er = en / int(external.sum()) if external.any() else None
    return dict(direct_visible=vp + vn, external_occluded=ep + en,
        direct_pred_visible=vp, direct_pred_nonvisible=vn,
        external_pred_visible=ep, external_pred_nonvisible=en,
        direct_recall=vr, external_recall=er,
        balanced_accuracy=float(np.mean([rate for rate in (vr, er) if rate is not None])))


def evaluate(args):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import LeaveOneGroupOut
    from sklearn.preprocessing import StandardScaler
    out = Path(args.output)
    if (out / "RESULTS.json").exists():
        raise FileExistsError("Preserve a completed CAL diagnosis")
    seal = C.read(out / "FEATURES_SEAL.json")
    if not seal["complete"] or binding(out / "PREDICTION_FEATURES.jsonl.gz") != seal["feature_cache"]:
        raise AssertionError("Prediction-only cache seal differs")
    labels = C.read(args.scoring)["frames"]
    if len(labels) != 76 or any(row["partition"] != "CAL" for row in labels):
        raise AssertionError("Isolated CAL76 labels only")
    labels = {row["id"]: row for row in labels}
    records = list(C.rows(out / "PREDICTION_FEATURES.jsonl.gz"))
    matrices = {name: [] for name in FEATURE_SETS}
    y, groups, keys, source_probability, entropy, skipped = [], [], [], [], [], Counter()
    for row in records:
        for k, state in enumerate(labels[row["id"]]["human_states_native"]):
            if state not in ("DIRECT_VISIBLE", "EXTERNAL_OCCLUDED"):
                skipped["non_target_human_state"] += 1; continue
            if row["frozen_self_hidden"][k] or not row["native_coordinate_in_frame"][k]:
                skipped["frozen_self_or_native_out_of_frame"] += 1; continue
            for name in matrices:
                matrices[name].append(row["features"][name][k])
            y.append(int(state == "DIRECT_VISIBLE")); groups.append(row["session"])
            keys.append([row["id"], k])
            source_probability.append(row["frozen_source_query"]["local_support_score"][k])
            entropy.append(row["frozen_source_query"]["entropy"][k])
    y, groups = np.asarray(y), np.asarray(groups)
    source_probability, entropy = np.asarray(source_probability), np.asarray(entropy)
    matrices = {name: np.asarray(values) for name, values in matrices.items()}
    if len(set(groups)) != 5:
        raise AssertionError("Five CAL sessions required")
    unique = sorted(set(groups)); selected = {}; candidates = {}; predictions = {}
    raw = dict(AUC=float(roc_auc_score(y, source_probability)),
        confusion_at0p5=confusion(y, source_probability, .5),
        fixed_threshold_curve=[dict(threshold=t, **confusion(y, source_probability, t)) for t in THRESHOLDS],
        sessions=[dict(session=s, AUC=float(roc_auc_score(y[groups == s], source_probability[groups == s]))
                       if len(np.unique(y[groups == s])) == 2 else None,
                       **confusion(y[groups == s], source_probability[groups == s], .5)) for s in unique],
        unknown_high_entropy_fraction=float(np.mean(entropy >= .8)),
        unknown_fraction_by_label={name: float(np.mean(entropy[y == value] >= .8))
                                   for name, value in (("DIRECT_VISIBLE", 1), ("EXTERNAL_OCCLUDED", 0))},
        caveat="Raw score was trained on a source local support proxy, not real semantic keypoint visibility")
    for name, X in matrices.items():
        settings = []
        for strength in STRENGTHS:
            probability = np.zeros(len(y)); folds = []
            for train_idx, test_idx in LeaveOneGroupOut().split(X, y, groups):
                scaler = StandardScaler().fit(X[train_idx])
                head = LogisticRegression(C=strength, class_weight="balanced", max_iter=2000, random_state=20261013)
                head.fit(scaler.transform(X[train_idx]), y[train_idx])
                probability[test_idx] = head.predict_proba(scaler.transform(X[test_idx]))[:, 1]
                folds.append(dict(session=str(groups[test_idx][0]), points=len(test_idx), train_points=len(train_idx),
                    train_visible=int(y[train_idx].sum()), train_external=int((1 - y[train_idx]).sum()),
                    AUC=float(roc_auc_score(y[test_idx], probability[test_idx]))
                    if len(np.unique(y[test_idx])) == 2 else None))
            threshold = max(THRESHOLDS, key=lambda t: (confusion(y, probability, t)["balanced_accuracy"], -abs(t - .5)))
            settings.append(dict(C=strength, AUC=float(roc_auc_score(y, probability)), threshold=float(threshold),
                confusion=confusion(y, probability, threshold), probabilities=probability.tolist(),
                folds=[dict(fold, **confusion(y[groups == fold["session"]], probability[groups == fold["session"]], threshold))
                       for fold in folds]))
        best = max(settings, key=lambda item: item["AUC"])
        candidates[name] = settings; predictions[name] = np.asarray(best["probabilities"])
        scaler = StandardScaler().fit(X)
        head = LogisticRegression(C=best["C"], class_weight="balanced", max_iter=2000, random_state=20261013)
        head.fit(scaler.transform(X), y)
        model = dict(schema="source_local_support_CAL_logistic_v1", feature_family=name, feature_names=FEATURE_SETS[name],
            mean=scaler.mean_.tolist(), scale=scaler.scale_.tolist(), coefficients=head.coef_[0].tolist(),
            intercept=float(head.intercept_[0]), C=best["C"], visible_threshold=best["threshold"],
            crossfit_AUC=best["AUC"], crossfit_confusion=best["confusion"], source_head=seal["source_head"],
            training_points=len(y), direct_points=int(y.sum()), external_points=int((1 - y).sum()),
            TEST_reads=0, FINAL_reads=0, feature_seal_sha256=C.sha(out / "FEATURES_SEAL.json"), integrated=False)
        C.write(out / (name + "_MODEL.json"), model)
        selected[name] = dict(best, model=binding(out / (name + "_MODEL.json")))
    rng = np.random.default_rng(20261013); invalid = 0
    differences = {name: [] for name in FEATURE_SETS if name != "RGB26"}
    for _ in range(2000):
        chosen = rng.choice(unique, len(unique), replace=True)
        ids = np.concatenate([np.flatnonzero(groups == session) for session in chosen])
        if len(np.unique(y[ids])) < 2:
            invalid += 1; continue
        base_auc = roc_auc_score(y[ids], predictions["RGB26"][ids])
        for name in differences:
            differences[name].append(float(roc_auc_score(y[ids], predictions[name][ids]) - base_auc))
    baseline = selected["RGB26"]; gates = {}
    for name, delta in differences.items():
        best = selected[name]; interval = np.quantile(delta, [.025, .975]).tolist()
        passed = (best["AUC"] >= baseline["AUC"] + .02
            and best["confusion"]["external_recall"] >= baseline["confusion"]["external_recall"]
            and best["confusion"]["direct_recall"] >= baseline["confusion"]["direct_recall"] - .02
            and interval[0] > 0)
        gates[name] = dict(pooled_AUC_gain=best["AUC"] - baseline["AUC"], session_bootstrap_95CI=interval,
            bootstrap_valid_draws=len(delta), bootstrap_invalid_one_class_draws=invalid,
            clear_crossfit_gate=bool(passed), integration_performed=False,
            CI_is_development_diagnostic_not_confirmatory=True)
    C.write(out / "RESULTS.json", dict(complete=True, schema="frozen_source_local_support_CAL_results_v1",
        points=len(y), frames=76, sessions=unique, direct=int(y.sum()), external=int((1 - y).sum()),
        skipped=dict(skipped), raw_source_frozen_head=raw, feature_sets=FEATURE_SETS,
        selected=selected, candidates=candidates, augmented_gates=gates,
        labels_source=binding(args.scoring), feature_seal=binding(out / "FEATURES_SEAL.json"),
        keys=keys, labels=y.tolist(), LOGO_group_per_point=groups.tolist(), new_TEST_reads=0, FINAL_reads=0,
        no_new_PnP_calls=True, original_observations_changed=False, source_head_frozen=True,
        limitations=["Source targets are local rendered-body proxy and synthetic opacity; real labels are human semantic corner visibility.",
                     "Only five CAL sessions; C and threshold selected from the same summarized crossfit predictions.",
                     "Entropy is not an out-of-domain or physical ownership certificate.",
                     "Prior CAL and root TEST summaries motivated this development pilot; TEST reuse remains non-independent."]))
    print("RAW_SOURCE_HEAD", "AUC", round(raw["AUC"], 4), "confusion", raw["confusion_at0p5"], flush=True)
    for name, best in selected.items():
        print(name, "C", best["C"], "AUC", round(best["AUC"], 4), "threshold", best["threshold"],
            "directRecall", round(best["confusion"]["direct_recall"], 4),
            "externalRecall", round(best["confusion"]["external_recall"], 4), flush=True)
    print("SOURCE_LOCAL_CAL_GATES", gates, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("infer", "evaluate"))
    parser.add_argument("--inputs", type=Path, default=C.SCRATCH / "CAL_INPUTS.json")
    parser.add_argument("--weights", type=Path, default=C.SCRATCH / "local_occlusion_v1/best_source_CAL.pt")
    parser.add_argument("--scoring", type=Path, default=C.SCRATCH / "CAL_SCORING_METADATA.json")
    parser.add_argument("--output", type=Path, default=C.SCRATCH / "local_occlusion_real_cal_v1")
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    args = parser.parse_args()
    (infer if args.stage == "infer" else evaluate)(args)


if __name__ == "__main__":
    main()
