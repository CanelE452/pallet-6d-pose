"""Finite CAL-only ownership/edge-continuation and raw-spatial visibility study."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import shutil

import cv2
import numpy as np

from . import common as C
from .state_visibility_v2 import (SCHEMA, OWNERSHIP_FEATURE_NAMES, RGB26_FEATURE_NAMES,
    StateVisibilityModel, extract_geometry_ownership, geometry_state_gates,
    load_state_model, predict_actual_visibility)
from .train_state_visibility_v2 import (BASE, DEFAULT_OUTPUT, SEED, STRENGTHS, binding,
    confusion, initial_geometry, make_CAL_O_gallery)


FAMILIES = ("NON_S_RGB26_GEOMETRY", "NON_S_RGB26_RAW_SPATIAL", "NON_S_RGB26_GEOMETRY_RAW_SPATIAL")
THRESHOLDS = tuple(np.round(np.arange(.01, .991, .01), 8))
MARGIN = .01
GATE = dict(min_LOGO_AUC=.6, min_LOGO_EXTERNAL_recall=.5, max_LOGO_DIRECT_false_O_fraction=.10)


def _feature_matrix(rgb26, geometry, spatial, family):
    local = np.array(rgb26, float, copy=True)
    local[:, 4:6] = 0.  # Freeze the preceding best RGB26 version.
    if "GEOMETRY" in family:
        local = np.column_stack([local, geometry])
    return local, spatial if "RAW_SPATIAL" in family else None


def _fit(local, spatial, y, strength):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    arrays = {}
    x = local
    if spatial is not None:
        s = StandardScaler().fit(spatial)
        arrays.update(spatial_mean=s.mean_, spatial_scale=s.scale_)
        x = np.column_stack([local, s.transform(spatial)])
    scaler = StandardScaler().fit(x)
    clf = LogisticRegression(C=strength, class_weight="balanced", max_iter=3000,
        random_state=SEED).fit(scaler.transform(x), y)
    arrays.update(class_mean=scaler.mean_, class_scale=scaler.scale_,
                  class_coefficients=clf.coef_[0], class_intercept=clf.intercept_)
    return arrays


def fit(args):
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import LeaveOneGroupOut
    cv2.setNumThreads(1)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    rgb_study = DEFAULT_OUTPUT / "nonS_rgb26_candidate_v1"
    inputs_path = BASE / "PREDICTION_INPUTS.json"
    scoring_path = BASE / "OOF_pose_v1/ISOLATED_CAL_SCORING.json"
    cohort_path = BASE / "COHORT_PROTOCOL.json"
    frames = [f for f in C.read(inputs_path)["frames"] if f["partition"] == "CAL"]
    cohort = C.read(cohort_path)
    if len(frames) != 46 or any(f["id"] in cohort["excluded_severe_ids"] for f in frames):
        raise AssertionError("Only exact nonsevere CAL46 may enter this finite study")
    first_model = C.read(DEFAULT_OUTPUT / "MODEL.json")
    for study, arrays_name in ((DEFAULT_OUTPUT, "NON_S_FEATURES.npz"), (rgb_study, "NON_S_RGB26_FEATURES.npz")):
        seal = C.read(study / "FEATURE_SEAL.json")
        if not seal["complete"]:
            raise AssertionError("Only sealed CAL-only feature caches may be reused")
        key = "feature_arrays" if study == DEFAULT_OUTPUT else "features"
        if binding(study / arrays_name) != seal[key]:
            raise AssertionError("Existing CAL-only feature cache binding changed")
    protocol = dict(schema="CAL46_BOUNDED_OWNERSHIP_RAW_SPATIAL_PREREGISTERED_V3",
        prior_CAL_evidence="RGB26 LOGO AUC .75937 but O output gate failed; cone raw/projection both biased29–41px, stool queries within3–6px",
        finite_families=list(FAMILIES), C_candidates=list(STRENGTHS), RGB26_version="legacy_zero_soft",
        ownership_feature_names=OWNERSHIP_FEATURE_NAMES, RGB26_feature_names=RGB26_FEATURE_NAMES,
        spatial_dim=2560, PCA_used=False,
        coordinate_guidance="Original eligible nonS query plus initial-projected same native point; geometry directions use eligible nonS projected points only",
        S_before_RGB="Erase all excluded q/projected to NaN before scene/ownership features; only eligible IDs contribute hull/radius/palette/directions/edges",
        target="Native human DIRECT_VISIBLE vs EXTERNAL_OCCLUDED; RGB palette/geometry quality never supply negative labels",
        basis_rule="All spatial/class scalers and classifiers fit only on LOGO fold-train known DIRECT/EXTERNAL rows",
        threshold_grid=list(THRESHOLDS), fixed_abstention_margin=MARGIN,
        selection="Within each family/C maximize pooled LOGO balanced accuracy with U counted as missed label; tie DIRECTrecall then nearest0.5 then smaller threshold. Prefer O-gate-passing candidates; highest AUC, tie balanced accuracy, smaller C then simpler family. If no gate passes select highest AUC but lowp becomes U",
        external_O_enabled_gate=GATE, expected_DIRECT=263, expected_EXTERNAL=7,
        initial_geometry_and_native_GT_phase_diagnostic_used=True, diagnostic_GT_never_features=True,
        fresh_features_after_protocol_seal=True, labels_join_after_features_seal=True,
        gallery="All nine CAL external labels with raw/red, initialprojection/cyan, native annotation/yellow and same-frameDIRECT",
        TEST_label_reads=0, TEST_feature_choice_reads=0, FINAL_reads=0, severe_rows_for_any_fit=0,
        previous_frozen_artifacts_unchanged=True,
        source_segmenter=first_model["pallet_segmenter"], backbone=first_model["backbone"],
        bindings=[binding(inputs_path), binding(scoring_path), binding(cohort_path),
            binding(DEFAULT_OUTPUT / "FEATURE_SEAL.json"), binding(rgb_study / "FEATURE_SEAL.json"),
            binding(DEFAULT_OUTPUT / "NON_S_FEATURES.npz"), binding(rgb_study / "NON_S_RGB26_FEATURES.npz"),
            binding(Path(__file__)), binding(Path(__file__).with_name("state_visibility_v2.py")),
            binding(Path(__file__).with_name("train_state_visibility_v2.py"))])
    C.write(out / "PROTOCOL.json", protocol)
    C.write(out / "PROTOCOL_SEAL.json", dict(before_fresh_features=True, before_candidate_training=True,
        protocol=binding(out / "PROTOCOL.json"), only_CAL=True))
    snapshot = out / "code_snapshot"
    snapshot.mkdir()
    for name in ("state_visibility_v2.py", "train_state_visibility_v2.py",
                 "train_state_visibility_bounded_v3.py", "test_state_visibility_v2.py"):
        shutil.copy2(Path(__file__).with_name(name), snapshot / name)
    rgb_meta = C.read(rgb_study / "FEATURE_METADATA.json")["rows"]
    spatial_meta = C.read(DEFAULT_OUTPUT / "FEATURE_METADATA.json")["rows"]
    rgb_index = {(r["id"], r["native_id"]): i for i, r in enumerate(rgb_meta)}
    spatial_index = {(r["id"], r["native_id"]): i for i, r in enumerate(spatial_meta)}
    with np.load(rgb_study / "NON_S_RGB26_FEATURES.npz", allow_pickle=False) as archive:
        rgb_cache = archive["features"].copy()
    with np.load(DEFAULT_OUTPUT / "NON_S_FEATURES.npz", allow_pickle=False) as archive:
        spatial_cache = archive["frozen_spatial"].copy()
    all_rgb, all_geom, all_spatial, metadata = [], [], [], []
    for frame in frames:
        image_path = C.ROOT / frame["image"]
        if C.sha(image_path) != frame["image_sha256"]:
            raise ValueError("Frozen CAL RGB binding changed")
        image = cv2.imread(str(image_path))
        q = np.asarray(frame["points"]["N3_SUBPIX"], float)[:8]
        hidden, roles, projected = initial_geometry(frame)
        _, ids = geometry_state_gates(image, q, hidden, roles, projected)
        geometry = extract_geometry_ownership(image, q, ids, projected)
        for k, geo in zip(ids, geometry):
            key = (frame["id"], int(k))
            if key not in rgb_index or key not in spatial_index:
                raise AssertionError("Previous sealed native phase/eligibility differs")
            all_rgb.append(rgb_cache[rgb_index[key]])
            all_spatial.append(spatial_cache[spatial_index[key]])
            all_geom.append(geo)
            metadata.append(dict(id=frame["id"], native_id=int(k), session=frame["session"]))
    rgb, geo, spatial = np.asarray(all_rgb), np.asarray(all_geom), np.asarray(all_spatial)
    np.savez_compressed(out / "NON_S_FEATURES.npz", rgb26=rgb, ownership24=geo, spatial2560=spatial)
    C.write(out / "FEATURE_METADATA.json", dict(rows=metadata))
    C.write(out / "FEATURE_SEAL.json", dict(complete=True, before_training_label_join=True,
        arrays=binding(out / "NON_S_FEATURES.npz"), metadata=binding(out / "FEATURE_METADATA.json"),
        protocol=binding(out / "PROTOCOL.json"), feature_rows=len(metadata), S_pixel_queries=0,
        TEST_RGB_reads=0, TEST_label_reads=0, FINAL_reads=0))
    scoring = C.read(scoring_path)
    if scoring["TEST_rows"] or scoring["FINAL_rows"] or scoring["severe_rows"] or len(scoring["rows"]) != 46:
        raise AssertionError("Only isolated CAL labels may be joined")
    annotations = scoring["rows"]
    sel, y, groups, ignored = [], [], [], Counter()
    for i, row in enumerate(metadata):
        state = annotations[row["id"]]["metadata"]["human_states_native"][row["native_id"]]
        if state not in ("DIRECT_VISIBLE", "EXTERNAL_OCCLUDED"):
            ignored[state] += 1
            continue
        sel.append(i)
        y.append(int(state == "DIRECT_VISIBLE"))
        groups.append(row["session"])
    y, groups = np.asarray(y), np.asarray(groups)
    rgb, geo, spatial = rgb[sel], geo[sel], spatial[sel]
    keys = [metadata[i] for i in sel]
    if y.sum() != 263 or (1-y).sum() != 7:
        raise AssertionError("Frozen CAL native visibility targets changed")
    candidates = []
    for family in FAMILIES:
        local, features = _feature_matrix(rgb, geo, spatial, family)
        for strength in STRENGTHS:
            p = np.zeros(len(y))
            folds = []
            for train, held in LeaveOneGroupOut().split(local, y, groups):
                arrays = _fit(local[train], None if features is None else features[train], y[train], strength)
                model = StateVisibilityModel(dict(feature_family=family, rgb26_version="legacy_zero_soft"), arrays, None)
                p[held] = predict_actual_visibility(local[held], None if features is None else features[held], model)
                folds.append(dict(session=str(groups[held][0]), held_rows=len(held),
                    train_DIRECT=int(y[train].sum()), train_EXTERNAL=int((1-y[train]).sum()),
                    AUC=float(roc_auc_score(y[held], p[held])) if len(np.unique(y[held])) == 2 else None,
                    held_rows_for_any_basis_fit=0))
            threshold = max(THRESHOLDS, key=lambda t: (confusion(y, p, t, MARGIN)["balanced_accuracy"],
                confusion(y, p, t, MARGIN)["direct_recall"], -abs(t-.5), -t))
            cm = confusion(y, p, threshold, MARGIN)
            auc = float(roc_auc_score(y, p))
            passes = (auc >= GATE["min_LOGO_AUC"] and cm["external_recall"] >= GATE["min_LOGO_EXTERNAL_recall"]
                and cm["direct_pred_O"]/cm["DIRECT"] <= GATE["max_LOGO_DIRECT_false_O_fraction"])
            candidate = dict(family=family, C=strength, AUC=auc, visible_threshold=float(threshold),
                confusion=confusion(y, p, threshold), abstention_confusion=cm,
                O_gate_passes=bool(passes), folds=folds, probabilities=p.tolist())
            candidates.append(candidate)
            print("BOUNDED_CAL_LOGO", family, strength, "AUC", auc, "threshold", threshold,
                  "Ogate", passes, cm, flush=True)
    best = max(candidates, key=lambda r: (r["O_gate_passes"], r["AUC"],
        r["abstention_confusion"]["balanced_accuracy"], -r["C"], -FAMILIES.index(r["family"])))
    local, features = _feature_matrix(rgb, geo, spatial, best["family"])
    arrays = _fit(local, features, y, best["C"])
    np.savez_compressed(out / "PARAMETERS.npz", **arrays)
    desc = dict(schema=SCHEMA, feature_family=best["family"], rgb26_version="legacy_zero_soft",
        arrays_file="PARAMETERS.npz", arrays_sha256=C.sha(out / "PARAMETERS.npz"),
        backbone=first_model["backbone"], pallet_segmenter=first_model["pallet_segmenter"],
        RGB26_feature_names=RGB26_FEATURE_NAMES, ownership_feature_names=OWNERSHIP_FEATURE_NAMES,
        C=best["C"], visible_threshold=best["visible_threshold"], abstention_margin=MARGIN,
        external_O_enabled=best["O_gate_passes"], external_O_enabled_gate=GATE,
        target="Native human DIRECT_VISIBLE versus EXTERNAL_OCCLUDED; no coordinate-error or palette-negative labels",
        train_scope="CAL46 CLEAN/MODERATE nonS RGB eligible only", DIRECT=263, EXTERNAL=7,
        crossfit_AUC=best["AUC"], crossfit_confusion=best["confusion"],
        crossfit_abstention_confusion=best["abstention_confusion"],
        initial_projected_RGB_guidance_used="GEOMETRY" in best["family"],
        projected_coordinates_used_as_observations=False, S_feature_queries=0,
        protocol_sha256=C.sha(out / "PROTOCOL.json"), feature_seal_sha256=C.sha(out / "FEATURE_SEAL.json"),
        TEST_rows_for_fit=0, FINAL_rows_used=0,
        O_generalization_status="limited_CAL_LOGO_development_evidence" if best["O_gate_passes"] else "unresolved_low_probability_abstains_U",
        limitations=["Seven RGB-eligible O labels in only two CAL sessions",
                     "Initial-model RGB probes are conditional guidance, never independent image coordinates",
                     "No TEST feature choice or tuning; CAL LOGO is development evidence"])
    C.write(out / "MODEL.json", desc)
    C.write(out / "CAL_LOGO_SELECTION.json", dict(candidates=candidates, selected_family=best["family"],
        selected_C=best["C"], selected_threshold=best["visible_threshold"],
        external_O_enabled=best["O_gate_passes"], keys=keys, labels=y.tolist(),
        ignored_human_states=dict(ignored), TEST_rows_used=0, FINAL_rows_used=0))
    make_CAL_O_gallery(frames, annotations, out,
        {(k["id"], k["native_id"]): float(p) for k, p in zip(keys, best["probabilities"])})
    exported = load_state_model(str(out / "MODEL.json"), args.device)
    p1 = predict_actual_visibility(local, features, StateVisibilityModel(desc, arrays, None))
    p2 = predict_actual_visibility(local, features, exported)
    delta = float(np.max(np.abs(p1-p2)))
    if delta > 1e-12:
        raise AssertionError("Exported bounded semantic model differs from fitted parameters")
    C.write(out / "MODEL_FREEZE.json", dict(complete=True, model=binding(out / "MODEL.json"),
        parameters=binding(out / "PARAMETERS.npz"), protocol=binding(out / "PROTOCOL.json"),
        selection=binding(out / "CAL_LOGO_SELECTION.json"), export_max_probability_difference=delta,
        S_feature_queries=0, TEST_training_reads=0, FINAL_reads=0,
        external_O_enabled=best["O_gate_passes"]))
    print("BOUNDED_STATE_MODEL_FROZEN", out / "MODEL.json", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT / "bounded_ownership_spatial_v3")
    parser.add_argument("--device", default="cuda")
    fit(parser.parse_args())


if __name__ == "__main__":
    main()
