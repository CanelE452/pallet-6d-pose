"""CAL-only initial-projection RGB probes and grouped visibility diagnostics.

Initial projected corners are extra image probes, never observations for a
pose fit. Frozen self-occlusion H is unchanged. All prediction features are
sealed before opening the isolated CAL human-state metadata. TEST/FINAL are
not consumed; this proposal was motivated by a prior TEST summary from root.
"""
from __future__ import annotations

import argparse
import builtins
from collections import Counter
from contextlib import contextmanager
import io
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np


ROOT = Path("/home/minjae/Documents/github/pallet-pose")
sys.path.insert(0, str(ROOT))
from scripts.research.pallet_visibility_curriculum_20261013_v1 import common as C
from scripts.research.pallet_visibility_curriculum_20261013_v1 import appearance as A
from scripts.research.pallet_visibility_curriculum_20261013_v1.calibration import features, SCALARS, VECTORS
from scripts.research.pallet_visibility_curriculum_20261013_v1.segmentation import segment
from scripts.research.pallet_visibility_curriculum_20261013_v1.segmented_appearance import infer_segmented_appearance


SCRATCH = C.SCRATCH
BASE_NAMES = (["q_rgb_" + name for name in SCALARS]
              + ["q_rgb_" + name + "_" + channel for name in VECTORS for channel in ("L", "a", "b")]
              + ["q_rgb_edge_distance_over_patch_radius"])
CNN_KEYS = ("segmentation_probability_patch_mean", "segmentation_effective_probability_nearby_inward_mean",
            "segmentation_effective_probability_outward_mean")
CNN_NAMES = ["q_" + key for key in CNN_KEYS]
PROJ_NAMES = (["projected_rgb_patch_lab_mean_" + v for v in ("L", "a", "b")]
              + ["projected_rgb_patch_lab_std_" + v for v in ("L", "a", "b")]
              + ["projected_rgb_visibility_score", "projected_rgb_boundary_score", "projected_rgb_local_accuracy_score"]
              + ["projected_" + k for k in CNN_KEYS])
DIFF_NAMES = ["q_minus_projected_rgb_visibility", "q_minus_projected_inward_material",
              "q_minus_projected_nearby_inward_material", "q_minus_projected_normalized_edge_distance",
              "q_minus_projected_CNN_patch_probability", "q_minus_projected_soft_inward_Lab_support",
              "q_minus_projected_Lmean_over255", "q_minus_projected_Lstd_over255"]
SET_NAMES = {
    "RGB26": BASE_NAMES,
    "RGB26_CNN3": BASE_NAMES + CNN_NAMES,
    "RGB26_CNN3_PROJECTED": BASE_NAMES + CNN_NAMES + PROJ_NAMES + DIFF_NAMES + ["normalized_initial_projection_residual"],
}
STRENGTHS = (.01, .1, 1., 10.)
THRESHOLDS = tuple(np.round(np.arange(.2, .81, .05), 8))


def binding(path):
    path = Path(path)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=C.sha(path))


@contextmanager
def no_scoring_reads():
    originals = (builtins.open, io.open, Path.open)
    blocked = ("SCORING_METADATA", "TARGETS.json", "STATIC_VISIBILITY", "DATA_CONTRACT", "test_v1", "FINAL_")
    def wrap(fn):
        def guarded(path, *args, **kwargs):
            if any(token in str(path) for token in blocked):
                raise AssertionError("Prediction feature phase attempted scoring read: " + str(path))
            return fn(path, *args, **kwargs)
        return guarded
    builtins.open, io.open, Path.open = map(wrap, originals)
    try:
        yield
    finally:
        builtins.open, io.open, Path.open = originals


def scalar(row, key):
    value = row.get(key)
    return float(value) if value is not None else 0.


def vector(row, key):
    value = row.get(key)
    return [0., 0., 0.] if value is None else [float(x) for x in value]


def cnn3(app, index):
    row = app["diagnostics"]["corners"][index]
    return [scalar(row, key) for key in CNN_KEYS]


def feature_sets(q, projected, raw, learned, projected_raw, projected_learned):
    base = features(raw)
    output = {name: [] for name in SET_NAMES}
    radius = max(1, raw["diagnostics"]["patch_radius_px"])
    projected_radius = max(1, projected_raw["diagnostics"]["patch_radius_px"])
    for k in range(8):
        r = raw["diagnostics"]["corners"][k]
        p = projected_raw["diagnostics"]["corners"][k]
        extra = cnn3(learned, k)
        projected_values = (vector(p, "patch_lab_mean") + vector(p, "patch_lab_std")
                            + [float(projected_raw[name][k]) for name in ("visibility_score", "boundary_score", "local_accuracy_score")]
                            + cnn3(projected_learned, k))
        # Each residual is relative image geometry or colour/support, not an
        # absolute pixel coordinate, object dimension, keypoint/session ID.
        differences = [float(raw["visibility_score"][k] - projected_raw["visibility_score"][k]),
            scalar(r, "inward_material_fraction") - scalar(p, "inward_material_fraction"),
            scalar(r, "nearby_inward_material_fraction") - scalar(p, "nearby_inward_material_fraction"),
            min(4., scalar(r, "edge_distance_px") / radius) - min(4., scalar(p, "edge_distance_px") / projected_radius),
            extra[0] - cnn3(projected_learned, k)[0],
            scalar(r, "soft_inward_material_support") - scalar(p, "soft_inward_material_support"),
            (vector(r, "patch_lab_mean")[0] - vector(p, "patch_lab_mean")[0]) / 255.,
            (vector(r, "patch_lab_std")[0] - vector(p, "patch_lab_std")[0]) / 255.]
        residual = float(np.linalg.norm(q[k] - projected[k]) / radius)
        residual = min(8., residual) if np.isfinite(residual) else 8.
        output["RGB26"].append(base[k].tolist())
        output["RGB26_CNN3"].append(base[k].tolist() + extra)
        output["RGB26_CNN3_PROJECTED"].append(base[k].tolist() + extra + projected_values + differences + [residual])
    for name, values in output.items():
        value = np.asarray(values)
        if value.shape != (8, len(SET_NAMES[name])) or not np.isfinite(value).all():
            raise AssertionError("Feature schema/finite values failed: " + name)
    return output


def infer(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    config = C.read(args.config)
    weights = Path(config["segmentation_weights"])
    frames = C.read(args.inputs)["frames"]
    if len(frames) != 76 or any(r["partition"] != "CAL" or r["population_role"] != "DEV" for r in frames):
        raise AssertionError("Exact CAL76 only, original DEV roles required")
    code = ROOT / "scripts/research/pallet_visibility_curriculum_20261013_v1"
    dependencies = [Path(__file__), args.inputs, args.config, weights] + [code / name for name in
        ("appearance.py", "calibration.py", "segmentation.py", "segmented_appearance.py", "common.py")]
    protocol = dict(schema="CAL_initial_projection_RGB_probe_v1", feature_sets=SET_NAMES,
        strengths=list(STRENGTHS), thresholds=list(THRESHOLDS), folds="LeaveOneGroupOut over5CAL sessions",
        frames=76, sessions=sorted({r["session"] for r in frames}), arm="N3_SUBPIX",
        projected_points_are_extra_image_probes_only=True, projected_points_used_as_fit_observations=False,
        no_new_PnP_calls=True, frozen_initial_self_hidden_unchanged=True,
        feature_inputs="current RGB, frozen q and initial pose, K/dimensions, frozen source CNN scores",
        label_target="human DIRECT_VISIBLE versus EXTERNAL_OCCLUDED; coordinate reliability is not the label",
        features_excluded=["absolute_coordinates", "keypoint_ID", "session_ID", "patch_size", "object_dimensions", "material_ID"],
        baseline_exact_existing_RGB26=True,
        baseline_historical_soft_color_fields="Historical calibration SCALARS names soft_*_color_support are absent from appearance; reproduce both zero columns rather than silently changing baseline.",
        selection="Best LOGO pooled AUC per declared family; threshold by fixed pooled crossfit balanced accuracy grid",
        clear_gate="Augmented crossfit AUC>=baseline+0.02, external recall>=baseline, direct recall>=baseline-0.02, paired session-bootstrap AUC difference lower95CI>0",
        inference_before_CAL_scoring_read=True, new_TEST_reads=0, FINAL_reads=0,
        design_informed_by_prior_root_TEST_summary=True,
        interpretation="CAL development diagnostic; prior root TEST summary motivated this followup, so later TEST reuse cannot be called independent confirmation",
        bindings=[binding(p) for p in dependencies])
    C.write(out / "PROTOCOL.json", protocol)
    begin = time.monotonic()
    cv2.setNumThreads(1)
    records = []
    with no_scoring_reads():
        for index, frame in enumerate(frames):
            image_path = C.ROOT / frame["image"]
            if C.sha(image_path) != frame["image_sha256"]:
                raise ValueError("Image SHA differs: " + frame["id"])
            image = cv2.imread(str(image_path))
            if image is None:
                raise FileNotFoundError(str(image_path))
            q = np.asarray(frame["points"]["N3_SUBPIX"], float)[:8]
            initial = frame["initial_poses"]["N3_SUBPIX"]
            xyz = C.cuboid(*initial["cf_extents"])
            R, t = np.asarray(initial["R_cf"]), np.asarray(initial["centroid"])
            H = C.self_hidden(xyz, R, t)
            projected = C.project(xyz, R, t, np.asarray(frame["K"]))
            mask = segment(image, weights)
            raw = A.infer_appearance(image, q, H)
            learned = infer_segmented_appearance(image, q, H, mask)
            projected_raw = A.infer_appearance(image, projected, H)
            projected_learned = infer_segmented_appearance(image, projected, H, mask)
            values = feature_sets(q, projected, raw, learned, projected_raw, projected_learned)
            valid = np.isfinite(q).all(1) & ~(q == -1).all(1)
            inside = valid & (q[:, 0] >= 0) & (q[:, 0] < image.shape[1]) & (q[:, 1] >= 0) & (q[:, 1] < image.shape[0])
            records.append(dict(id=frame["id"], session=frame["session"], partition="CAL", arm="N3_SUBPIX",
                frozen_q=q.tolist(), initial_projected_probe_xy=projected.tolist(), frozen_self_hidden=H.tolist(),
                native_coordinate_valid=valid.tolist(), native_coordinate_in_frame=inside.tolist(),
                original_raw_states=raw["states"], original_segmented_states=learned["states"],
                projected_probe_states=projected_raw["states"], features=values,
                source_image_sha256=frame["image_sha256"], projected_points_used_as_observations=False))
            if (index + 1) % 16 == 0:
                print("PROJECTED_CAL_FEATURES", index + 1, round(time.monotonic() - begin, 2), flush=True)
    if [binding(p) for p in dependencies] != protocol["bindings"]:
        raise AssertionError("Frozen dependency changed during feature inference")
    C.save_rows(out / "PREDICTION_FEATURES.jsonl.gz", records)
    C.write(out / "FEATURES_SEAL.json", dict(complete=True, frames=76,
        feature_cache=binding(out / "PREDICTION_FEATURES.jsonl.gz"), protocol=binding(out / "PROTOCOL.json"),
        scoring_reads_before_seal=0, human_state_reads=0, TEST_reads=0, FINAL_reads=0,
        new_PnP_calls=0, initial_known_pose_projection_calls=76,
        source_CNN_inference_images=76, appearance_calls=304, elapsed_seconds=time.monotonic() - begin))


def confusion(y, p, threshold):
    positive = p >= threshold
    visible, external = y == 1, y == 0
    vp = int((visible & positive).sum())
    vn = int((visible & ~positive).sum())
    ep = int((external & positive).sum())
    en = int((external & ~positive).sum())
    vr = vp / int(visible.sum()) if visible.any() else None
    er = en / int(external.sum()) if external.any() else None
    rates = [v for v in [vr, er] if v is not None]
    return dict(direct_visible=vp + vn, external_occluded=ep + en,
        direct_pred_visible=vp, direct_pred_nonvisible=vn,
        external_pred_visible=ep, external_pred_nonvisible=en,
        direct_recall=vr, external_recall=er, balanced_accuracy=float(np.mean(rates)))


def evaluate(args):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import LeaveOneGroupOut
    from sklearn.preprocessing import StandardScaler
    out = Path(args.output)
    if (out / "RESULTS.json").exists():
        raise FileExistsError("Preserve completed CAL experiment")
    seal = C.read(out / "FEATURES_SEAL.json")
    if not seal["complete"] or binding(out / "PREDICTION_FEATURES.jsonl.gz") != seal["feature_cache"]:
        raise AssertionError("Prediction feature seal differs")
    # The only annotation opened is root's separately isolated CAL76 file.
    labels = C.read(args.scoring)["frames"]
    if len(labels) != 76 or any(r["partition"] != "CAL" for r in labels):
        raise AssertionError("No TEST labels permitted")
    labels = {r["id"]: r for r in labels}
    records = list(C.rows(out / "PREDICTION_FEATURES.jsonl.gz"))
    matrices = {name: [] for name in SET_NAMES}
    y, groups, keys, skipped = [], [], [], Counter()
    for row in records:
        states = labels[row["id"]]["human_states_native"]
        for k, state in enumerate(states):
            if state not in ("DIRECT_VISIBLE", "EXTERNAL_OCCLUDED"):
                skipped["non_target_human_state"] += 1
                continue
            if row["frozen_self_hidden"][k] or not row["native_coordinate_in_frame"][k]:
                skipped["frozen_self_or_native_out_of_frame"] += 1
                continue
            for name in matrices:
                matrices[name].append(row["features"][name][k])
            y.append(int(state == "DIRECT_VISIBLE"))
            groups.append(row["session"])
            keys.append([row["id"], k])
    y, groups = np.asarray(y), np.asarray(groups)
    matrices = {name: np.asarray(values) for name, values in matrices.items()}
    if len(set(groups)) != 5:
        raise AssertionError("Five CAL groups required")
    predictions, candidates, selected_models = {}, {}, {}
    for name, X in matrices.items():
        settings = []
        for strength in STRENGTHS:
            probability = np.zeros(len(y))
            folds = []
            for train_idx, test_idx in LeaveOneGroupOut().split(X, y, groups):
                scaler = StandardScaler().fit(X[train_idx])
                head = LogisticRegression(C=strength, class_weight="balanced", max_iter=2000, random_state=20261013)
                head.fit(scaler.transform(X[train_idx]), y[train_idx])
                probability[test_idx] = head.predict_proba(scaler.transform(X[test_idx]))[:, 1]
                folds.append(dict(session=str(groups[test_idx][0]), points=len(test_idx),
                    train_points=len(train_idx), train_visible=int(y[train_idx].sum()),
                    train_external=int((1 - y[train_idx]).sum()),
                    AUC=float(roc_auc_score(y[test_idx], probability[test_idx])) if len(np.unique(y[test_idx])) == 2 else None))
            threshold = max(THRESHOLDS, key=lambda t: (confusion(y, probability, t)["balanced_accuracy"], -abs(t - .5)))
            settings.append(dict(C=strength, AUC=float(roc_auc_score(y, probability)),
                threshold=float(threshold), confusion=confusion(y, probability, threshold),
                folds=[dict(fold, **confusion(y[groups == fold["session"]], probability[groups == fold["session"]], threshold)) for fold in folds],
                probabilities=probability.tolist()))
        best = max(settings, key=lambda item: item["AUC"])
        candidates[name] = settings
        predictions[name] = np.asarray(best["probabilities"])
        scaler = StandardScaler().fit(X)
        head = LogisticRegression(C=best["C"], class_weight="balanced", max_iter=2000, random_state=20261013).fit(scaler.transform(X), y)
        model = dict(schema="CAL_RGB_projection_probe_logistic_v1", feature_family=name, feature_names=SET_NAMES[name],
            mean=scaler.mean_.tolist(), scale=scaler.scale_.tolist(), coefficients=head.coef_[0].tolist(),
            intercept=float(head.intercept_[0]), C=best["C"], visible_threshold=best["threshold"],
            crossfit_AUC=best["AUC"], crossfit_confusion=best["confusion"], training_points=len(y),
            direct_points=int(y.sum()), external_points=int((1 - y).sum()), TEST_reads=0, FINAL_reads=0,
            projected_points_are_extra_image_probes=True, projected_points_are_fit_observations=False,
            feature_seal_sha256=C.sha(out / "FEATURES_SEAL.json"))
        C.write(out / (name + "_MODEL.json"), model)
        selected_models[name] = dict(best, model=binding(out / (name + "_MODEL.json")))
    baseline = selected_models["RGB26"]
    unique = sorted(set(groups))
    rng = np.random.default_rng(20261013)
    differences = {name: [] for name in SET_NAMES if name != "RGB26"}
    invalid = 0
    for _ in range(2000):
        chosen = rng.choice(unique, len(unique), replace=True)
        ids = np.concatenate([np.flatnonzero(groups == session) for session in chosen])
        if len(np.unique(y[ids])) < 2:
            invalid += 1
            continue
        base_auc = roc_auc_score(y[ids], predictions["RGB26"][ids])
        for name in differences:
            differences[name].append(float(roc_auc_score(y[ids], predictions[name][ids]) - base_auc))
    gates = {}
    for name, delta in differences.items():
        best = selected_models[name]
        interval = np.quantile(delta, [.025, .975]).tolist()
        passed = (best["AUC"] >= baseline["AUC"] + .02
            and best["confusion"]["external_recall"] >= baseline["confusion"]["external_recall"]
            and best["confusion"]["direct_recall"] >= baseline["confusion"]["direct_recall"] - .02
            and interval[0] > 0)
        gates[name] = dict(pooled_AUC_gain=best["AUC"] - baseline["AUC"], session_bootstrap_95CI=interval,
            bootstrap_valid_draws=len(delta), bootstrap_invalid_one_class_draws=invalid,
            clear_crossfit_gate=bool(passed), integration_performed=False,
            CI_is_development_diagnostic_not_confirmatory=True)
    result = dict(schema="CAL_initial_projection_RGB_probe_results_v1", complete=True,
        points=len(y), frames=76, sessions=unique, direct=int(y.sum()), external=int((1 - y).sum()),
        skipped=dict(skipped), feature_sets=SET_NAMES, candidates=candidates, selected=selected_models,
        augmented_gates=gates, labels_source=binding(args.scoring), feature_seal=binding(out / "FEATURES_SEAL.json"),
        keys=keys, labels=y.tolist(), LOGO_group_per_point=groups.tolist(),
        TEST_reads=0, FINAL_reads=0, new_PnP_calls=0, original_observations_changed=False,
        meaning="Human point visibility discrimination, not ground-truth coordinate-error detection or certified3D occlusion ownership",
        limitations=["Threshold/C selection uses the same crossfit predictions subsequently summarized; performance is CAL model selection evidence.",
                     "Only five known CAL sessions; clustered intervals are exploratory.",
                     "Initial reprojection is an extra RGB probe, never an independent measured point.",
                     "Proposal was motivated by an earlier root TEST summary; it does not restore independence of that TEST."])
    C.write(out / "RESULTS.json", result)
    for name, model in selected_models.items():
        print(name, "C", model["C"], "AUC", round(model["AUC"], 4), "threshold", model["threshold"],
              "directRecall", round(model["confusion"]["direct_recall"], 4), "externalRecall", round(model["confusion"]["external_recall"], 4), flush=True)
    print("CAL_CLEAR_GATES", json.dumps(gates), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("infer", "evaluate"))
    parser.add_argument("--inputs", type=Path, default=SCRATCH / "CAL_INPUTS.json")
    parser.add_argument("--config", type=Path, default=SCRATCH / "config_segmentation_v1.json")
    parser.add_argument("--scoring", type=Path, default=SCRATCH / "CAL_SCORING_METADATA.json")
    parser.add_argument("--output", type=Path, default=SCRATCH / "project_visibility_cal_v1")
    args = parser.parse_args()
    (infer if args.stage == "infer" else evaluate)(args)


if __name__ == "__main__":
    main()
