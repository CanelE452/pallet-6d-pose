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
    RGB26_FEATURE_NAMES, extract_nonS_rgb26, extract_state_features, geometry_state_gates,
    patch_radius, predict_actual_visibility, sha256)


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


def _diagnostic_crop(image, center, gt, title, *, width=192, projection=None):
    """Diagnostic raw-query crop with explicit outside-canvas padding."""
    origin = np.floor(np.asarray(center) - width / 2).astype(int)
    x0, y0 = origin
    h, w = image.shape[:2]
    tile = np.full((width, width, 3), 40, np.uint8)
    sx0, sy0, sx1, sy1 = max(0, x0), max(0, y0), min(w, x0+width), min(h, y0+width)
    if sx1 > sx0 and sy1 > sy0:
        tile[sy0-y0:sy1-y0, sx0-x0:sx1-x0] = image[sy0:sy1, sx0:sx1]
    cv2.drawMarker(tile, tuple(np.rint(np.asarray(center)-origin).astype(int)),
                   (0, 0, 255), cv2.MARKER_CROSS, 15, 2)
    if gt is not None and np.isfinite(gt).all():
        loc = np.rint(np.asarray(gt)-origin).astype(int)
        if np.all((loc >= 0) & (loc < width)):
            cv2.circle(tile, tuple(loc), 6, (0, 255, 255), 2)
    if projection is not None and np.isfinite(projection).all():
        loc = np.rint(np.asarray(projection)-origin).astype(int)
        if np.all((loc >= 0) & (loc < width)):
            cv2.drawMarker(tile, tuple(loc), (255, 255, 0), cv2.MARKER_SQUARE, 9, 1)
    canvas = np.full((width+48, width, 3), 245, np.uint8)
    canvas[48:] = tile
    for j, line in enumerate(title):
        cv2.putText(canvas, line, (4, 18+18*j), cv2.FONT_HERSHEY_SIMPLEX, .36, (20, 20, 20), 1, cv2.LINE_AA)
    return canvas


def make_CAL_O_gallery(frames, annotations, out, probabilities):
    """Nine actual human O labels and same-frame DIRECT controls; CAL only.

    The native-to-reference map is used only to draw a diagnostic annotation
    ring. It is never supplied to the feature extractor or fitted classifier.
    """
    gallery = out / "CAL_O_QUERY_GALLERY"
    gallery.mkdir(exist_ok=False)
    audit = []
    sheets = []
    for frame in frames:
        ann = annotations[frame["id"]]
        states = ann["metadata"]["human_states_native"]
        external_ids = [k for k, s in enumerate(states) if s == "EXTERNAL_OCCLUDED"]
        if not external_ids:
            continue
        image = cv2.imread(str(C.ROOT / frame["image"]))
        q = np.asarray(frame["points"]["N3_SUBPIX"], float)[:8]
        hidden, roles, projected = initial_geometry(frame)
        gates, ids = geometry_state_gates(image, q, hidden, roles, projected)
        direct = [k for k, s in enumerate(states) if s == "DIRECT_VISIBLE" and k in ids]
        reference_gt = np.asarray(ann["target"]["gt"], float)
        native_to_reference = ann["metadata"]["native_to_reference_corner_ids"]
        for k in external_ids:
            control = min(direct, key=lambda j: float(np.linalg.norm(q[j]-q[k]))) if direct else None
            gt = reference_gt[native_to_reference[k]]
            error = float(np.linalg.norm(q[k]-gt))
            p = probabilities.get((frame["id"], k))
            O = _diagnostic_crop(image, q[k], gt, ["EXTERNAL native%d / raw query" % k,
                "GT-ref%d; q-GT %.1fpx" % (native_to_reference[k], error)], projection=projected[k])
            if control is None:
                D = np.full_like(O, 245)
            else:
                dgt = reference_gt[native_to_reference[control]]
                D = _diagnostic_crop(image, q[control], dgt, ["DIRECT native%d / same frame" % control,
                    "GT-ref%d; q-GT %.1fpx" % (native_to_reference[control], np.linalg.norm(q[control]-dgt))], projection=projected[control])
            panel = np.concatenate([O, D], axis=1)
            name = frame["id"].replace(":", "__") + "_native%d.png" % k
            cv2.imwrite(str(gallery / name), panel)
            sheets.append(panel)
            audit.append(dict(id=frame["id"], native_id=k, human_state=states[k],
                native_reference_corner_id=native_to_reference[k], raw_query_xy=q[k].tolist(),
                diagnostic_native_GT_xy=gt.tolist(), raw_query_GT_distance_px=error,
                initial_projection_xy=projected[k].tolist(), initial_projection_GT_distance_px=float(np.linalg.norm(projected[k]-gt)),
                raw_query_initial_projection_distance_px=float(np.linalg.norm(q[k]-projected[k])),
                initial_geometric_state=gates[k]["state"], raw_query_out_of_frame=gates[k]["raw_query_out_of_frame"],
                rgb_feature_eligible=k in ids, CAL_LOGO_p_visible=p,
                same_frame_DIRECT_native_id=control, image=name,
                diagnostic_GT_used_for_features_or_labels=False))
    if len(audit) != 9:
        raise AssertionError("Gallery must include all nine native CAL EXTERNAL labels")
    contact = np.concatenate(sheets, axis=0)
    cv2.imwrite(str(gallery / "ALL_NINE_CAL_O.png"), contact)
    html = ["<!doctype html><meta charset='utf-8'><title>CAL O native query diagnostic</title>",
        "<style>body{font:15px sans-serif;margin:24px}article{margin:24px 0}img{max-width:100%}code{background:#eee}</style>",
        "<h1>Nine human CAL EXTERNAL labels</h1><p>Red cross: original native query. Cyan square: initial model projection. Yellow ring: CAL annotation after the declared native-to-reference mapping. Right: nearest eligible DIRECT query in the same frame. These are CAL diagnostics; annotation coordinates are never image features.</p>"]
    import html as html_module
    for row in audit:
        html.append("<article><p><code>%s native%d</code> initial gate=%s, q-GT=%.1f px, LOGO p-visible=%s</p><img src='%s'></article>" %
            (html_module.escape(row["id"]), row["native_id"], row["initial_geometric_state"],
             row["raw_query_GT_distance_px"], str(row["CAL_LOGO_p_visible"]), row["image"]))
    (gallery / "index.html").write_text("\n".join(html)+"\n")
    C.write(gallery / "NATIVE_PHASE_DIAGNOSTIC.json", dict(schema="CAL_ONLY_NINE_ACTUAL_EXTERNAL_QUERY_DIAGNOSTIC",
        rows=audit, native_to_reference_map_used_only_for_annotation_overlay=True,
        TEST_label_reads=0, FINAL_reads=0, actual_O_labels=9, eligible_RGB_O_labels=7))


def fit_nonS_rgb26(args):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import LeaveOneGroupOut
    from sklearn.preprocessing import StandardScaler
    from . import appearance as A

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(1)
    inputs_path = BASE / "PREDICTION_INPUTS.json"
    scoring_path = BASE / "OOF_pose_v1/ISOLATED_CAL_SCORING.json"
    cohort_path = BASE / "COHORT_PROTOCOL.json"
    frames = [f for f in C.read(inputs_path)["frames"] if f["partition"] == "CAL"]
    cohort = C.read(cohort_path)
    if len(frames) != 46 or any(f["id"] in cohort["excluded_severe_ids"] for f in frames):
        raise AssertionError("Exact nonsevere CAL46 required")
    versions = ("legacy_zero_soft", "soft_support")
    thresholds = tuple(np.round(np.arange(.05, .951, .025), 8))
    margin = .025
    gate = dict(min_LOGO_AUC=.6, min_LOGO_EXTERNAL_recall=.5, max_LOGO_DIRECT_false_O_fraction=.10)
    protocol = dict(schema="NONS_RGB26_ACTUAL_VISIBILITY_CAL46_PREREGISTERED_V1",
        scope="CAL46 CLEAN/MODERATE only; old first model and its diagnostics preserved",
        previous_CAL_findings="First spatial16+local40 model LOGO AUC .5144, EXTERNAL recall0; local40/mask/gradient explorations did not solve O",
        feature_family="NON_S_RGB26", versions=list(versions), feature_names=RGB26_FEATURE_NAMES,
        appearance_config=dict(A.DEFAULT_CONFIG), pure_helpers=["appearance._scene", "appearance._local"],
        old_infer_appearance_called=False, helper_rule="Erase all noneligible coordinates to NaN BEFORE image helpers; S cannot affect global hull/radius/palette or local pixel queries",
        coordinates_sampled="Original eligible nonS native query only; no projected-coordinate RGB patches",
        label_target="native human DIRECT_VISIBLE versus EXTERNAL_OCCLUDED; palette mismatch never a label",
        expected_DIRECT=263, expected_EXTERNAL=7, expected_loss_ignored_human_SELF=15,
        C_candidates=list(STRENGTHS), threshold_grid=list(thresholds), abstention_margin=margin,
        selection="Best pooled CAL four-session LOGO AUC; tie balanced accuracy, smaller C, legacy version. Threshold maximizes binary balanced accuracy; tie nearest0.5 then lower threshold",
        basis_rule="Fit scaler/classifier only on fold-train known DIRECT/EXTERNAL; loss-ignore humanSELF/UNANNOTATED, geometricS never extracted",
        external_O_enabled_gate=gate, failed_gate_behavior="Low visible probability becomes U rather than O; report unresolved",
        gallery="All9 actual CAL O query crops with same-frameDIRECT controls; nativeGT mapping diagnostic overlays only, never training features",
        human_CAL_scope_audit_already_performed=True, training_label_join_after_feature_seal=True,
        TEST_label_reads=0, TEST_feature_choice_reads=0, FINAL_reads=0, severe_rows_for_any_fit=0,
        inputs=binding(inputs_path), isolated_CAL_scoring=binding(scoring_path), cohort=binding(cohort_path),
        code=[binding(Path(__file__)), binding(Path(__file__).with_name("state_visibility_v2.py")), binding(Path(A.__file__))])
    C.write(out / "PROTOCOL.json", protocol)
    C.write(out / "PROTOCOL_SEAL.json", dict(before_feature_extraction=True, before_candidate_training=True,
        protocol=binding(out / "PROTOCOL.json"), only_CAL=True))
    # A source snapshot keeps the old first-model implementation separately.
    import shutil
    snapshot = out / "code_snapshot"
    snapshot.mkdir()
    for name in ("state_visibility_v2.py", "train_state_visibility_v2.py", "test_state_visibility_v2.py"):
        shutil.copy2(Path(__file__).with_name(name), snapshot / name)
    all_features, feature_meta, frame_meta = [], [], []
    for frame in frames:
        path = C.ROOT / frame["image"]
        if sha256(path) != frame["image_sha256"]:
            raise ValueError("Frozen CAL image changed")
        image = cv2.imread(str(path))
        q = np.asarray(frame["points"]["N3_SUBPIX"], float)[:8]
        hidden, roles, projected = initial_geometry(frame)
        records, ids = geometry_state_gates(image, q, hidden, roles, projected)
        features, diag = extract_nonS_rgb26(image, q, ids)
        all_features.extend(features)
        feature_meta.extend(dict(id=frame["id"], native_id=int(k), session=frame["session"],
                                 initial_role=roles[k]) for k in ids)
        frame_meta.append(dict(id=frame["id"], initial_self_hidden=hidden.tolist(),
            RGB_queried_ids=ids.tolist(), geometry_records=records, RGB_diagnostics=diag))
    all_features = np.asarray(all_features, float)
    np.savez_compressed(out / "NON_S_RGB26_FEATURES.npz", features=all_features)
    C.write(out / "FEATURE_METADATA.json", dict(rows=feature_meta, frames=frame_meta))
    C.write(out / "FEATURE_SEAL.json", dict(complete=True, before_training_label_join=True,
        features=binding(out / "NON_S_RGB26_FEATURES.npz"), metadata=binding(out / "FEATURE_METADATA.json"),
        protocol=binding(out / "PROTOCOL.json"), S_pixel_queries=0, feature_rows=len(all_features),
        TEST_RGB_reads=0, TEST_label_reads=0, FINAL_reads=0))
    scoring = C.read(scoring_path)
    if scoring["TEST_rows"] or scoring["FINAL_rows"] or scoring["severe_rows"] or len(scoring["rows"]) != 46:
        raise AssertionError("Isolated CAL46 labels only")
    annotations = scoring["rows"]
    sel, labels, groups, skipped = [], [], [], Counter()
    for i, row in enumerate(feature_meta):
        state = annotations[row["id"]]["metadata"]["human_states_native"][row["native_id"]]
        if state not in ("DIRECT_VISIBLE", "EXTERNAL_OCCLUDED"):
            skipped[state] += 1
            continue
        sel.append(i)
        labels.append(int(state == "DIRECT_VISIBLE"))
        groups.append(row["session"])
    y, groups = np.asarray(labels), np.asarray(groups)
    X = all_features[sel]
    keys = [feature_meta[i] for i in sel]
    if y.sum() != 263 or (1-y).sum() != 7:
        raise AssertionError("CAL state target eligibility changed")
    candidates = []
    for version in versions:
        features = X.copy()
        if version == "legacy_zero_soft":
            features[:, 4:6] = 0.
        for strength in STRENGTHS:
            p = np.zeros(len(y))
            folds = []
            for train, held in LeaveOneGroupOut().split(features, y, groups):
                scaler = StandardScaler().fit(features[train])
                clf = LogisticRegression(C=strength, class_weight="balanced", max_iter=3000,
                    random_state=SEED).fit(scaler.transform(features[train]), y[train])
                p[held] = clf.predict_proba(scaler.transform(features[held]))[:, 1]
                folds.append(dict(session=str(groups[held][0]), train_DIRECT=int(y[train].sum()),
                    train_EXTERNAL=int((1-y[train]).sum()), held_rows=len(held),
                    AUC=float(roc_auc_score(y[held], p[held])) if len(np.unique(y[held])) == 2 else None,
                    scaler_heldout_rows=0))
            threshold = max(thresholds, key=lambda t: (confusion(y, p, t)["balanced_accuracy"], -abs(t-.5), -t))
            candidates.append(dict(version=version, C=strength, AUC=float(roc_auc_score(y, p)),
                visible_threshold=float(threshold), confusion=confusion(y, p, threshold),
                abstention_confusion=confusion(y, p, threshold, margin), folds=folds, probabilities=p.tolist()))
            print("NONS_RGB26_CAL_LOGO", version, strength, candidates[-1]["AUC"], candidates[-1]["abstention_confusion"], flush=True)
    best = max(candidates, key=lambda r: (r["AUC"], r["confusion"]["balanced_accuracy"], -r["C"], r["version"] == "legacy_zero_soft"))
    selected_X = X.copy()
    if best["version"] == "legacy_zero_soft":
        selected_X[:, 4:6] = 0.
    scaler = StandardScaler().fit(selected_X)
    clf = LogisticRegression(C=best["C"], class_weight="balanced", max_iter=3000,
        random_state=SEED).fit(scaler.transform(selected_X), y)
    arrays = dict(class_mean=scaler.mean_, class_scale=scaler.scale_,
        class_coefficients=clf.coef_[0], class_intercept=clf.intercept_)
    np.savez_compressed(out / "PARAMETERS.npz", **arrays)
    cm = best["abstention_confusion"]
    O_enabled = (best["AUC"] >= gate["min_LOGO_AUC"] and cm["external_recall"] >= gate["min_LOGO_EXTERNAL_recall"]
                 and cm["direct_pred_O"] / cm["DIRECT"] <= gate["max_LOGO_DIRECT_false_O_fraction"])
    desc = dict(schema=SCHEMA, feature_family="NON_S_RGB26", rgb26_version=best["version"],
        arrays_file="PARAMETERS.npz", arrays_sha256=sha256(out / "PARAMETERS.npz"),
        feature_names=RGB26_FEATURE_NAMES, C=best["C"], visible_threshold=best["visible_threshold"],
        abstention_margin=margin, external_O_enabled=bool(O_enabled),
        target="human native DIRECT_VISIBLE versus EXTERNAL_OCCLUDED; palette score is only a feature",
        train_scope="CAL46 CLEAN/MODERATE nonS RGB eligible only", DIRECT=263, EXTERNAL=7,
        crossfit_AUC=best["AUC"], crossfit_confusion=best["confusion"],
        crossfit_abstention_confusion=cm, external_O_enabled_gate=gate,
        O_generalization_status="limited_CAL_LOGO_development_evidence" if O_enabled else "unresolved_low_probability_abstains_U",
        protocol_sha256=sha256(out / "PROTOCOL.json"), feature_seal_sha256=sha256(out / "FEATURE_SEAL.json"),
        S_feature_queries=0, TEST_rows_for_fit=0, FINAL_rows_used=0,
        limitations=["Seven RGB eligible O labels across two CAL sessions only",
                     "No independent TEST used for family/threshold selection",
                     "S geometry/human disagreements are loss-ignored; native GT only gallery overlays"])
    C.write(out / "MODEL.json", desc)
    C.write(out / "CAL_LOGO_SELECTION.json", dict(candidates=candidates, selected_version=best["version"],
        selected_C=best["C"], selected_threshold=best["visible_threshold"], keys=keys, labels=y.tolist(),
        ignored_human_states=dict(skipped), external_O_enabled=bool(O_enabled), TEST_rows_used=0, FINAL_rows_used=0))
    probabilities = {(row["id"], row["native_id"]): float(p) for row, p in zip(keys, best["probabilities"])}
    make_CAL_O_gallery(frames, annotations, out, probabilities)
    from .state_visibility_v2 import load_state_model
    model = load_state_model(str(out / "MODEL.json"))
    p1 = clf.predict_proba(scaler.transform(selected_X))[:, 1]
    p2 = predict_actual_visibility(X, None, model)
    delta = float(np.max(np.abs(p1-p2)))
    if delta > 1e-12:
        raise AssertionError("RGB26 export differs from sklearn classifier")
    C.write(out / "MODEL_FREEZE.json", dict(complete=True, model=binding(out / "MODEL.json"),
        parameters=binding(out / "PARAMETERS.npz"), protocol=binding(out / "PROTOCOL.json"),
        CAL_LOGO_selection=binding(out / "CAL_LOGO_SELECTION.json"), export_max_probability_difference=delta,
        S_feature_queries=0, TEST_training_reads=0, FINAL_reads=0, external_O_enabled=bool(O_enabled)))
    print("NONS_RGB26_MODEL_FROZEN", out / "MODEL.json", "O_enabled", O_enabled, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--nonS-rgb26-candidate", action="store_true")
    args = parser.parse_args()
    if args.nonS_rgb26_candidate:
        if args.output == DEFAULT_OUTPUT:
            args.output = DEFAULT_OUTPUT / "nonS_rgb26_candidate_v1"
        fit_nonS_rgb26(args)
    else:
        fit(args)


if __name__ == "__main__":
    main()
