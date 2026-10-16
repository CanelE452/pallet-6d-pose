"""CAL-only learning of localization reliability from GT-free local evidence.

This head predicts coordinate accuracy conditional on a corner being directly
visible. It does not predict depth ownership, physical pose correctness, or
visibility. The reference XY labels are accessed only by the training function;
predict_accuracy uses the original RGB diagnostics and supplied predicted pose.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path

import numpy as np

from scripts.research.pallet_visibility_curriculum_20261013_v1 import common as C
from scripts.research.pallet_visibility_curriculum_20261013_v1.calibration import features
from scripts.research.pallet_visibility_curriculum_20261013_v1.curriculum import solve_curriculum
from scripts.research.pallet_visibility_curriculum_20261013_v1.scoring import pose_metric


STRENGTHS = (.01, .1, 1., 10.)
FAMILIES = ("RGB26", "RGB26_REPROJECTION3")
LABEL_RADIUS_PX = 8.0


def accuracy_features(appearance, points, xyz, K, initial_R, initial_t, family):
    rgb = features(appearance)
    if rgb.shape != (8, 26):
        raise ValueError("Expected the frozen 26-field local RGB feature contract")
    if family == "RGB26":
        return rgb
    if family != "RGB26_REPROJECTION3":
        raise ValueError("Unknown predeclared localization feature family")
    points = np.asarray(points, float)
    if points.shape != (8, 2):
        raise ValueError("Expected eight independent predicted coordinates")
    projected = C.project(xyz, initial_R, initial_t, K)
    residual = projected - points
    radius = max(1., float(appearance["diagnostics"]["patch_radius_px"]))
    geometry = np.column_stack((np.linalg.norm(residual, axis=1), np.abs(residual))) / radius
    return np.column_stack((rgb, np.nan_to_num(geometry, nan=0., posinf=0., neginf=0.)))


def _probabilities(x, model):
    z = (x - np.asarray(model["mean"])) / np.asarray(model["scale"])
    logit = z @ np.asarray(model["coefficients"]) + model["intercept"]
    return 1. / (1. + np.exp(-np.clip(logit, -60, 60)))


def predict_accuracy(appearance, points, xyz, K, initial_R, initial_t, model):
    """Clone appearance and replace local_accuracy_score with learned evidence."""
    result = deepcopy(appearance)
    x = accuracy_features(appearance, points, xyz, K, initial_R, initial_t, model["feature_family"])
    if not np.isfinite(x).all():
        raise ValueError("Localization reliability features must be finite")
    probability = _probabilities(x, model)
    for index, state in enumerate(result["states"]):
        if state in ("OUT_OF_FRAME", "SELF_OCCLUDED"):
            continue
        reason = result["diagnostics"]["corners"][index].get("reason")
        if reason not in ("rgb_evidence_only", "frozen_source_CNN_mask_and_raw_RGB_features"):
            continue
        result["local_accuracy_score"][index] = float(np.clip(probability[index], .01, .99))
    result["diagnostics"]["localization_accuracy"] = dict(
        schema=model["schema"], feature_family=model["feature_family"], probabilities=probability.tolist(),
        target="reference XY error <= 8 px conditional on DIRECT_VISIBLE", ground_truth_input=False,
        physical_pose_accuracy_probability=False, visibility_probability=False,
        learned_from="CAL_ONLY", new_detector_forward=False)
    return result


def _summary(rows):
    out = dict(frames=len(rows), accepted=sum(r["accepted"] for r in rows))
    for key in ("translation_cm", "rotation_deg", "ADDsym_cm"):
        values = np.asarray([r["metric"][key] for r in rows])
        out[key] = dict(mean=float(values.mean()), median=float(np.median(values)), P90=float(np.quantile(values, .9)))
    out["success_5cm_5deg"] = float(np.mean([r["metric"]["success_5cm_5deg"] for r in rows]))
    return out


def train(bundle_path, prediction_directory, output):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, brier_score_loss
    from sklearn.preprocessing import StandardScaler

    bundle = C.read(bundle_path)
    frames = {f["id"]:f for f in bundle["frames"]}
    if len(frames) != 76 or any(f["partition"] != "CAL" for f in frames.values()):
        raise ValueError("This experiment requires the frozen CAL-only 76-frame bundle")
    protocol = C.read(prediction_directory / "PROTOCOL.json")
    if protocol["partition"] != "calibration" or set(protocol["ids"]) != set(frames):
        raise ValueError("Predictions must be exactly the CAL bundle population")
    raw = [r for r in C.rows(prediction_directory / "PREDICTIONS.jsonl.gz") if r["method"] == "RGB_CURRICULUM"]
    if len(raw) != 76 or {r["id"] for r in raw} != set(frames):
        raise ValueError("CAL RGB_CURRICULUM predictions must have exactly one row per frame")
    output.mkdir(parents=True, exist_ok=False)
    # Write the grid and label definition before extracting any labels or fitting.
    C.write(output / "PROTOCOL.json", dict(schema="CAL_LOCALIZATION_RELIABILITY_V1", diagnostic=True,
        fixed_C_grid=STRENGTHS, fixed_feature_families=FAMILIES, fixed_label_radius_px=LABEL_RADIUS_PX,
        eligible_label_population="human DIRECT_VISIBLE, not initial SELF_OCCLUDED or OUT_OF_FRAME, finite independent predicted coordinates",
        negative_examples="DIRECT_VISIBLE with reference XY error > 8 px; external occluders excluded",
        validation="Leave-one-session-out CAL; pooled and individual-session AUC",
        selection="Maximum pooled held-session AUC, tie by feature family then smaller C",
        success_rule="CV AUC strictly exceeds both original local_accuracy AUC and 0.59",
        physical_pose_truth_used_for_accuracy_labels=False, TEST_accessed=False, FINAL_accessed=False,
        module_sha256=C.sha(__file__), RGB_feature_module_sha256=C.sha(Path(features.__code__.co_filename)),
        bundle_sha256=C.sha(bundle_path), predictions_sha256=C.sha(prediction_directory / "PREDICTIONS.jsonl.gz")))
    feature_rows = {family:[] for family in FAMILIES}
    all_features = {}
    labels, sessions, keys, errors, local_scores, visibility_scores = [], [], [], [], [], []
    for row in raw:
        frame = frames[row["id"]]
        initial = frame["initial_poses"][row["arm"]]
        xyz = C.cuboid(*initial["cf_extents"])
        q = np.asarray(row["original_points"])
        appearance = row["appearance"]
        rows_by_family = {family:accuracy_features(appearance, q, xyz, frame["K"],
            initial["R_cf"], initial["centroid"], family) for family in FAMILIES}
        all_features[row["id"]] = rows_by_family
        metadata = frame["scoring_metadata"]
        reference = np.asarray(frame["target"]["gt"], float)[:8][metadata["native_to_reference_corner_ids"]]
        distance = np.linalg.norm(q-reference, axis=1)
        for index, actual in enumerate(metadata["human_states_native"]):
            if actual != "DIRECT_VISIBLE" or appearance["states"][index] in ("SELF_OCCLUDED", "OUT_OF_FRAME"):
                continue
            if not np.isfinite(q[index]).all() or np.all(q[index] == -1):
                continue
            for family in FAMILIES:
                feature_rows[family].append(rows_by_family[family][index])
            labels.append(distance[index] <= LABEL_RADIUS_PX)
            sessions.append(row["session"])
            keys.append([row["id"], index])
            errors.append(distance[index])
            local_scores.append(appearance["local_accuracy_score"][index])
            visibility_scores.append(appearance["visibility_score"][index])
    y, groups = np.asarray(labels), np.asarray(sessions)
    X = {family:np.asarray(rows) for family, rows in feature_rows.items()}
    if not all(np.isfinite(values).all() for values in X.values()):
        raise ValueError("Nonfinite training evidence")
    candidates = []
    for family in FAMILIES:
        for strength in STRENGTHS:
            probabilities = np.zeros(len(y))
            frame_probabilities = {}
            folds = []
            for held_session in sorted(set(sessions)):
                train_mask, valid_mask = groups != held_session, groups == held_session
                scaler = StandardScaler().fit(X[family][train_mask])
                head = LogisticRegression(C=strength, class_weight="balanced", max_iter=2000, random_state=20261013)
                head.fit(scaler.transform(X[family][train_mask]), y[train_mask])
                probabilities[valid_mask] = head.predict_proba(scaler.transform(X[family][valid_mask]))[:, 1]
                for row in raw:
                    if row["session"] == held_session:
                        frame_probabilities[row["id"]] = head.predict_proba(scaler.transform(all_features[row["id"]][family]))[:, 1].tolist()
                folds.append(dict(session=held_session, points=int(valid_mask.sum()), accurate=int(y[valid_mask].sum()),
                    AUC=float(roc_auc_score(y[valid_mask], probabilities[valid_mask])),
                    Brier=float(brier_score_loss(y[valid_mask], probabilities[valid_mask]))))
            candidates.append(dict(feature_family=family, C=strength,
                CV_AUC=float(roc_auc_score(y, probabilities)), CV_Brier=float(brier_score_loss(y, probabilities)),
                folds=folds, probabilities=probabilities.tolist(), frame_probabilities=frame_probabilities))
    baseline_local = float(roc_auc_score(y, local_scores))
    baseline_visibility = float(roc_auc_score(y, visibility_scores))
    best = max(candidates, key=lambda c:(c["CV_AUC"], -FAMILIES.index(c["feature_family"]), -c["C"]))
    useful = bool(best["CV_AUC"] > max(.59, baseline_local))
    scaler = StandardScaler().fit(X[best["feature_family"]])
    head = LogisticRegression(C=best["C"], class_weight="balanced", max_iter=2000, random_state=20261013)
    head.fit(scaler.transform(X[best["feature_family"]]), y)
    model = dict(schema="localization_accuracy_logistic_v1", feature_family=best["feature_family"],
        mean=scaler.mean_.tolist(), scale=scaler.scale_.tolist(), coefficients=head.coef_[0].tolist(),
        intercept=float(head.intercept_[0]), C=best["C"], fixed_label_radius_px=LABEL_RADIUS_PX,
        training_partition="CAL_ONLY", training_frames=76, training_points=len(y), accurate_points=int(y.sum()),
        training_sessions=sorted(set(sessions)), CV_AUC=best["CV_AUC"], CV_Brier=best["CV_Brier"],
        useful_evidence=useful, target_conditional_on_visible=True, predicts_physical_pose_accuracy=False,
        predicts_visibility=False, probabilities_calibrated=False,
        TEST_accessed=False, FINAL_accessed=False, excluded_features=["corner_ID", "absolute_coordinate", "image_size", "session_ID"],
        training_predictions_sha256=C.sha(prediction_directory / "PREDICTIONS.jsonl.gz"))
    # Fit model parameters only after the held-session selection is complete.
    C.write(output / "MODEL.json", model)
    C.write(output / "TRAINING.json", dict(points=len(y), positives=int(y.sum()), negatives=int((~y).sum()),
        by_session={s:dict(Counter("accurate" if y[i] else "inaccurate" for i in range(len(y)) if groups[i]==s)) for s in sorted(set(sessions))},
        baseline_local_accuracy_AUC=baseline_local, baseline_visibility_AUC=baseline_visibility,
        candidates=candidates, keys=keys, labels=y.tolist(), reference_error_px=errors,
        selected={k:best[k] for k in ("feature_family", "C", "CV_AUC", "CV_Brier", "folds")},
        useful_evidence=useful, model_sha256=C.sha(output / "MODEL.json")))
    pose_rows = []
    for row in raw:
        frame = frames[row["id"]]
        initial = frame["initial_poses"][row["arm"]]
        xyz = C.cuboid(*initial["cf_extents"])
        q, K = np.asarray(row["original_points"]), np.asarray(frame["K"])
        R, t = np.asarray(initial["R_cf"]), np.asarray(initial["centroid"])
        Q = R.T @ np.asarray(initial["R_physical"])
        for variant in ("ORIGINAL_ACCURACY", "FULL_CAL_ACCURACY", "HELD_SESSION_ACCURACY"):
            appearance = deepcopy(row["appearance"])
            if variant == "FULL_CAL_ACCURACY":
                appearance = predict_accuracy(appearance, q, xyz, K, R, t, model)
            elif variant == "HELD_SESSION_ACCURACY":
                probabilities = best["frame_probabilities"][row["id"]]
                for index, state in enumerate(appearance["states"]):
                    if state not in ("OUT_OF_FRAME", "SELF_OCCLUDED"):
                        appearance["local_accuracy_score"][index] = float(np.clip(probabilities[index], .01, .99))
            cfg = dict(protocol["settings"].get("curriculum") or {}, minimum_accuracy=.01)
            solution = solve_curriculum(q, xyz, K, R, t, row["self_hidden"], appearance, cfg)
            pose = dict(initial, R_cf=solution["R"], R_physical=(np.asarray(solution["R"]) @ Q).tolist(), centroid=solution["t"])
            pose_rows.append(dict(id=row["id"], session=row["session"], variant=variant,
                accepted=solution["accepted"], metric=pose_metric(pose, frame["target"]["truth"]),
                model_phase_source="fixed original initial pose; no oracle pose/model supplied to solver"))
    C.save_rows(output / "POSE_ROWS.jsonl.gz", pose_rows)
    grouped = defaultdict(list)
    for row in pose_rows:
        grouped[row["variant"]].append(row)
    C.write(output / "RESULTS.json", dict(points=len(y), fixed_candidates=len(candidates), selected_family=best["feature_family"],
        selected_C=best["C"], held_session_AUC=best["CV_AUC"], original_local_accuracy_AUC=baseline_local,
        original_visibility_AUC=baseline_visibility, useful_evidence=useful,
        pose_summary={name:_summary(rows) for name, rows in grouped.items()},
        limits=["CAL tuning; held-session selection is not an untouched external test",
               "Class-balanced logistic outputs are reliability evidence, not calibrated physical correctness probabilities",
               "Accuracy target uses reference 2D labels only on directly visible corners",
               "All pose diagnostics preserve initial dimensions, physical basis and hard self mask",
               "Visibility source was already trained on all CAL; held-session accuracy pose arm is not fully held-session visibility"],
        TEST_accessed=False, FINAL_accessed=False, MODEL_sha256=C.sha(output / "MODEL.json")))
    print("BASELINES", "local_AUC", baseline_local, "visibility_AUC", baseline_visibility, flush=True)
    for candidate in candidates:
        print("CANDIDATE", candidate["feature_family"], candidate["C"], "CV_AUC", round(candidate["CV_AUC"], 5),
              "fold_AUC", [round(f["AUC"], 5) for f in candidate["folds"]], flush=True)
    print("SELECTED", best["feature_family"], best["C"], "CV_AUC", best["CV_AUC"], "USEFUL", useful, flush=True)
    for name, rows in grouped.items():
        print("POSE", name, _summary(rows), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    train(args.bundle, args.predictions, args.output)
