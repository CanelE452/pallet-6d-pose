"""CAL46-only fit of actual visibility, with all geometric S removed first."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import time

import cv2
import numpy as np

from . import common as C
from .joint_observation import JointObservationModel
from .segmentation import segment
from .state_visibility_v2 import (SCHEMA, LOCAL_FEATURE_NAMES, StateVisibilityModel,
    extract_state_features, geometry_state_gates, patch_radius, predict_actual_visibility, sha256)


BASE = C.DOC / "nonsevere_v1"
DEFAULT_OUTPUT = BASE / "statewise_joint_ot_20261014_v2/state_visibility"
STRENGTHS = (.01, .1, 1., 10.)
THRESHOLDS = tuple(np.round(np.arange(.1, .901, .025), 8))
SEED = 20261014
PCA_COMPONENTS = 16
ABSTENTION_MARGIN = .05


def binding(path):
    path = Path(path)
    return dict(path=str(path), sha256=sha256(path), bytes=path.stat().st_size)


def initial_geometry(frame):
    pose = frame["initial_poses"]["N3_SUBPIX"]
    if not pose.get("available"):
        raise ValueError("CAL actual visibility requires available frozen initial geometry")
    xyz = C.cuboid(*pose["cf_extents"])
    R, t = np.asarray(pose["R_cf"]), np.asarray(pose["centroid"])
    hidden = C.self_hidden(xyz, R, t)
    projected = C.project(xyz, R, t, frame["K"])
    hull_ids = cv2.convexHull(projected.astype(np.float32), returnPoints=False).reshape(-1)
    roles = ["B" if k in hull_ids else "I" for k in range(8)]
    return hidden, roles, projected


def confusion(y, p, threshold, margin=0.):
    y, p = np.asarray(y), np.asarray(p)
    visible, external = y == 1, y == 0
    b = p >= threshold + margin
    o = p < threshold - margin
    u = ~(b | o)
    vr = float(np.mean(b[visible])) if visible.any() else None
    er = float(np.mean(o[external])) if external.any() else None
    return dict(DIRECT=int(visible.sum()), EXTERNAL=int(external.sum()),
        direct_pred_visible=int((visible & b).sum()), direct_pred_O=int((visible & o).sum()),
        direct_pred_U=int((visible & u).sum()), external_pred_visible=int((external & b).sum()),
        external_pred_O=int((external & o).sum()), external_pred_U=int((external & u).sum()),
        direct_recall=vr, external_recall=er,
        balanced_accuracy=float(np.mean([r for r in (vr, er) if r is not None])))


def fit_parameters(local, spatial, y, strength):
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    ss = StandardScaler().fit(spatial)
    pca = PCA(n_components=PCA_COMPONENTS, svd_solver="full", random_state=SEED)
    compressed = pca.fit_transform(ss.transform(spatial))
    x = np.column_stack([local, compressed])
    cs = StandardScaler().fit(x)
    clf = LogisticRegression(C=strength, class_weight="balanced", solver="lbfgs",
                             max_iter=3000, random_state=SEED).fit(cs.transform(x), y)
    return dict(spatial_mean=ss.mean_, spatial_scale=ss.scale_, pca_mean=pca.mean_,
                pca_components=pca.components_, class_mean=cs.mean_, class_scale=cs.scale_,
                class_coefficients=clf.coef_[0], class_intercept=clf.intercept_)


def fit(args):
    import torch
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import LeaveOneGroupOut

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    inputs_path = BASE / "PREDICTION_INPUTS.json"
    cohort_path = BASE / "COHORT_PROTOCOL.json"
    scoring_path = BASE / "OOF_pose_v1/ISOLATED_CAL_SCORING.json"
    old_descriptor_path = BASE / "joint_model/MODEL.json"
    segmenter_path = BASE / "clean_source_segmentation_v1/best_source_CAL.pt"
    descriptor = C.read(old_descriptor_path)
    backbone = descriptor["backbone"]
    weights = C.ROOT / backbone["weights_repo_relative"]
    if sha256(weights) != backbone["weights_sha256"]:
        raise ValueError("Frozen public RGB weights changed")
    frozen = JointObservationModel({"backbone": backbone}, {}, weights, args.device)
    frames = [f for f in C.read(inputs_path)["frames"] if f["partition"] == "CAL"]
    cohort = C.read(cohort_path)
    if len(frames) != 46 or any(f["id"] in cohort["excluded_severe_ids"] for f in frames):
        raise AssertionError("Exact CLEAN/MODERATE CAL46 required, severe rows=0")
    code = Path(__file__).with_name("state_visibility_v2.py")
    protocol = dict(schema="ACTUAL_VISIBILITY_CAL46_PREREGISTERED_V2", frames=46,
        target="human_states_native DIRECT_VISIBLE versus EXTERNAL_OCCLUDED",
        initial_state_order="S first, conditional/certified T next, invalid/OOB RGB query U, then B/I/O",
        S_pixel_queries=0, initial_S_excluded_before_any_RGB_feature_extraction=True,
        feature_radius_uses_only_nonS_RGB_eligible_queries=True,
        feature_names=LOCAL_FEATURE_NAMES, spatial_features=2560, PCA_components=PCA_COMPONENTS,
        backbone=backbone, backbone_trainable_parameters=0,
        source_segmenter=binding(segmenter_path), mask_is_feature_not_label=True,
        known_CAL_audit=dict(DIRECT=268, EXTERNAL=9, SELF=82, UNANNOTATED=9,
            expected_eligible_DIRECT=263, expected_eligible_EXTERNAL=7,
            expected_human_SELF_initial_geometry_disagreements_loss_ignored=15),
        previous_CAL_label_scope_audit_performed=True,
        feature_cache_precedes_training_label_join=True,
        basis_rule="All scalers/PCA/classifier use only geometric nonS inframe human DIRECT/EXTERNAL training rows, fold training subset only",
        folds="LeaveOneGroupOut over four CAL sessions; no TEST or FINAL label reads",
        C_candidates=list(STRENGTHS), fixed_threshold_grid=list(THRESHOLDS),
        selection="Highest pooled CAL LOGO AUC; tie by LOGO balanced accuracy then smaller C. Threshold maximizes pooled LOGO balanced accuracy; tie nearest 0.5 then smaller threshold",
        abstention_margin=ABSTENTION_MARGIN, abstention_margin_fixed_before_fit=True,
        seed=SEED, model_produces_coordinates=False, coordinate_error_not_a_training_target=True,
        initial_geometry_label_disagreements_not_overwritten=True,
        TEST_rows_for_training_or_selection=0, FINAL_rows_used=0, severe_rows_for_any_fit=0,
        limits=["Only seven RGB-eligible EXTERNAL labels in two CAL sessions",
                "Conditional T is a hypothesis of the frozen initial pose, not independently certified truncation",
                "Human SELF/UNANNOTATED are loss-ignored; U is internal abstention",
                "CAL LOGO selection is development evidence, not an independent test claim"],
        inputs=binding(inputs_path), cohort=binding(cohort_path), isolated_CAL_labels=binding(scoring_path),
        code=[binding(Path(__file__)), binding(code)])
    C.write(out / "PROTOCOL.json", protocol)
    protocol_binding = binding(out / "PROTOCOL.json")
    start = time.monotonic()
    local_rows, spatial_rows, feature_metadata, frame_metadata = [], [], [], []
    for frame in frames:
        image_path = C.ROOT / frame["image"]
        if sha256(image_path) != frame["image_sha256"]:
            raise ValueError("Frozen CAL RGB binding changed")
        image = cv2.imread(str(image_path))
        if image is None or list(image.shape[:2]) != frame["raw_hw"]:
            raise ValueError("Actual CAL image canvas differs")
        q = np.asarray(frame["points"]["N3_SUBPIX"], float)[:8]
        hidden, roles, projected = initial_geometry(frame)
        records, ids = geometry_state_gates(image, q, hidden, roles, projected)
        radius = patch_radius(q, ids)
        if len(ids):
            mask = segment(image, segmenter_path, args.device)
            local, spatial = extract_state_features(image, q, ids, radius, frozen, pallet_mask=mask)
            local_rows.extend(local)
            spatial_rows.extend(spatial)
            feature_metadata.extend(dict(id=frame["id"], native_id=int(k), session=frame["session"],
                initial_role=roles[k], patch_radius_px=radius) for k in ids)
        frame_metadata.append(dict(id=frame["id"], session=frame["session"], partition="CAL",
            image_sha256=frame["image_sha256"], initial_self_hidden=hidden.tolist(),
            initial_roles=roles, RGB_feature_native_ids=ids.tolist(), geometry_records=records))
        print("STATE_FEATURES", len(frame_metadata), "/46", "nonS_queries", len(ids), flush=True)
    local, spatial = np.asarray(local_rows, float), np.asarray(spatial_rows, float)
    np.savez_compressed(out / "NON_S_FEATURES.npz", local_features=local, frozen_spatial=spatial)
    C.write(out / "FEATURE_METADATA.json", dict(rows=feature_metadata, frames=frame_metadata))
    C.write(out / "FEATURE_SEAL.json", dict(complete=True, protocol=protocol_binding,
        feature_arrays=binding(out / "NON_S_FEATURES.npz"),
        feature_metadata=binding(out / "FEATURE_METADATA.json"), frames=46, feature_rows=len(local),
        S_feature_queries=0, TEST_RGB_reads=0, TEST_label_reads=0, FINAL_reads=0,
        training_label_join_performed=False, source_segmenter=binding(segmenter_path)))
    # Only the isolated CAL46 file is opened for this training-label join.
    scoring = C.read(scoring_path)
    if scoring["TEST_rows"] or scoring["FINAL_rows"] or scoring["severe_rows"] or len(scoring["rows"]) != 46:
        raise AssertionError("Only isolated CAL46 label rows are authorized")
    annotations = scoring["rows"]
    selected, y, groups, skipped = [], [], [], Counter()
    for index, row in enumerate(feature_metadata):
        state = annotations[row["id"]]["metadata"]["human_states_native"][row["native_id"]]
        if state not in ("DIRECT_VISIBLE", "EXTERNAL_OCCLUDED"):
            skipped[state] += 1
            continue
        selected.append(index)
        y.append(int(state == "DIRECT_VISIBLE"))
        groups.append(row["session"])
    y, groups = np.asarray(y, int), np.asarray(groups)
    local, spatial = local[selected], spatial[selected]
    keys = [feature_metadata[i] for i in selected]
    if int(y.sum()) != 263 or int((1 - y).sum()) != 7:
        raise AssertionError("Actual visibility CAL eligibility differs from frozen protocol")
    candidates = []
    for strength in STRENGTHS:
        probability = np.zeros(len(y))
        folds = []
        for train, held in LeaveOneGroupOut().split(local, y, groups):
            arrays = fit_parameters(local[train], spatial[train], y[train], strength)
            model = StateVisibilityModel({}, arrays, frozen)
            probability[held] = predict_actual_visibility(local[held], spatial[held], model)
            folds.append(dict(session=str(groups[held][0]), train_rows=len(train), held_rows=len(held),
                train_DIRECT=int(y[train].sum()), train_EXTERNAL=int((1-y[train]).sum()),
                AUC=float(roc_auc_score(y[held], probability[held])) if len(np.unique(y[held])) == 2 else None,
                basis_fit_on_held_rows=False))
        threshold = max(THRESHOLDS, key=lambda t: (confusion(y, probability, t)["balanced_accuracy"], -abs(t-.5), -t))
        candidates.append(dict(C=strength, AUC=float(roc_auc_score(y, probability)),
            visible_threshold=float(threshold), confusion=confusion(y, probability, threshold),
            abstention_confusion=confusion(y, probability, threshold, ABSTENTION_MARGIN),
            folds=folds, probabilities=probability.tolist()))
        print("STATE_CAL_LOGO", strength, "AUC", candidates[-1]["AUC"], "threshold", threshold, flush=True)
    best = max(candidates, key=lambda item: (item["AUC"], item["confusion"]["balanced_accuracy"], -item["C"]))
    arrays = fit_parameters(local, spatial, y, best["C"])
    np.savez_compressed(out / "PARAMETERS.npz", **arrays)
    model_descriptor = dict(schema=SCHEMA, family="frozen_spatial_PCA16_RGB_gradient_TinyMask_actual_visibility",
        arrays_file="PARAMETERS.npz", arrays_sha256=sha256(out / "PARAMETERS.npz"),
        backbone=backbone, pallet_segmenter=dict(weights_repo_relative=str(segmenter_path.relative_to(C.ROOT)),
            weights_sha256=sha256(segmenter_path)), local_feature_names=LOCAL_FEATURE_NAMES,
        spatial_features=2560, PCA_components=PCA_COMPONENTS, C=best["C"],
        visible_threshold=best["visible_threshold"], abstention_margin=ABSTENTION_MARGIN,
        target="human native DIRECT_VISIBLE vs EXTERNAL_OCCLUDED; coordinate quality excluded",
        train_scope="CAL46 CLEAN/MODERATE nonS RGB-eligible DIRECT/EXTERNAL only",
        training_rows=len(y), DIRECT=int(y.sum()), EXTERNAL=int((1-y).sum()),
        S_feature_queries=0, severe_rows_for_any_fit=0, TEST_rows_for_fit=0, FINAL_rows_used=0,
        protocol_sha256=protocol_binding["sha256"], feature_seal_sha256=sha256(out / "FEATURE_SEAL.json"),
        crossfit_AUC=best["AUC"], crossfit_confusion=best["confusion"],
        crossfit_abstention_confusion=best["abstention_confusion"],
        limitations=protocol["limits"])
    C.write(out / "MODEL.json", model_descriptor)
    C.write(out / "CAL_LOGO_SELECTION.json", dict(candidates=candidates, selected_C=best["C"],
        selected_threshold=best["visible_threshold"], keys=keys, labels=y.tolist(),
        ignored_human_states=dict(skipped), coordinate_quality_used=False,
        all_basis_fits_CAL_only=True, TEST_rows_used=0, FINAL_rows_used=0))
    # Ensure JSON/NPZ loading reproduces the in-memory parameters exactly.
    from .state_visibility_v2 import load_state_model
    exported = load_state_model(str(out / "MODEL.json"), args.device)
    in_memory = predict_actual_visibility(local, spatial, StateVisibilityModel(model_descriptor, arrays, frozen))
    reread = predict_actual_visibility(local, spatial, exported)
    delta = float(np.max(np.abs(in_memory-reread)))
    if delta > 1e-12:
        raise AssertionError("Exported state classifier does not reproduce fitted inference")
    C.write(out / "MODEL_FREEZE.json", dict(complete=True, model=binding(out / "MODEL.json"),
        parameters=binding(out / "PARAMETERS.npz"), protocol=protocol_binding,
        CAL_LOGO_selection=binding(out / "CAL_LOGO_SELECTION.json"),
        parameter_export_max_probability_difference=delta, S_feature_queries=0,
        TEST_training_reads=0, FINAL_reads=0, elapsed_seconds=time.monotonic()-start))
    print("STATE_MODEL_FROZEN", str(out / "MODEL.json"), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda")
    fit(parser.parse_args())


if __name__ == "__main__":
    main()
