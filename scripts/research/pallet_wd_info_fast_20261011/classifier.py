"""Eight fixed frozen-feature probes; current N3->SubPix geometry, separate labels.

GPU extraction and CPU fitting are separate CLI phases. No real GT is opened
before classifier states and prediction probabilities have been saved.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
import warnings

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as TF

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.research.pallet_wd_info_fast_20261011 import common as C

NAMES = ("long-face-front", "short-face-front")
ARMS = ("D_LOGISTIC", "G_LOGISTIC", "V_LOGISTIC", "V_MLP", "G_MLP", "G_V_MLP", "G_V_D_MLP", "V_SHUFFLED_LOGISTIC")


def modules():
    sys.path.insert(0, str(C.SOURCE / "scripts/research/pallet_dim_conditioned_p_v1"))
    sys.path.insert(1, str(C.SOURCE / "scripts/research/pallet_final_ml_contribution_test_v1"))
    sys.path.insert(2, str(C.SOURCE))
    import refiner
    import inference
    from scripts.research.pallet_selector_recovery_v1 import features as geo
    from challenge.evaluation_v2 import pnp_selector
    return refiner, inference, geo, pnp_selector


def arrays():
    base = C.SOURCE / "data/pallet/results/pallet_line_pose_v1/cache"
    return {k: np.load(base / (k + ".npy"), mmap_mode="r") for k in
            ("p3", "p4", "points", "point_valid", "boxes", "input_shape", "gain", "score", "selected_candidate_index")}


def roi_vector(p3, p4, box, input_shape):
    box = np.asarray(box, float)
    if not np.isfinite(box).all() or np.any(box[2:] <= box[:2]):
        return np.zeros(768, np.float32), False
    outputs = []
    for feature, stride in ((p3, 8), (p4, 16)):
        if torch.is_tensor(feature):
            feature = feature.detach().cpu().numpy()
        if feature.ndim == 4:
            feature = feature[0]
        h, w = np.asarray(input_shape, int) // stride
        x0, y0 = np.maximum(np.floor(box[:2] / stride).astype(int), 0)
        x1, y1 = np.minimum(np.ceil(box[2:] / stride).astype(int), [w, h])
        if x1 <= x0 or y1 <= y0:
            return np.zeros(768, np.float32), False
        crop = torch.from_numpy(np.array(feature[:, y0:y1, x0:x1], copy=True)).float()
        outputs.append(TF.adaptive_avg_pool2d(crop[None], (2, 2))[0].numpy().ravel())
    out = np.concatenate(outputs).astype(np.float32)
    assert out.shape == (768,) and np.isfinite(out).all()
    return out, True


def subpix(gray, q0, qn, valid):
    q = np.array(qn, float, copy=True)
    h, w = gray.shape
    calls = 0
    for k in range(8):
        if not valid[k] or not np.isfinite(q[k]).all() or (q[k] == -1).all():
            continue
        if not (0 <= q[k, 0] < w and 0 <= q[k, 1] < h) or min(h, w) < 15:
            continue
        calls += 1
        try:
            value = cv2.cornerSubPix(gray, q[k].astype(np.float32).reshape(1, 1, 2).copy(),
                (5, 5), (-1, -1), (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, .001))
            value = np.asarray(value, float).reshape(2)
            if np.isfinite(value).all() and 0 <= value[0] < w and 0 <= value[1] < h:
                q[k] = value
        except (cv2.error, ValueError):
            pass
    usable = valid[:8] & np.isfinite(q0[:8]).all(-1) & np.isfinite(q[:8]).all(-1)
    d = q[:8] - q0[:8]
    length = np.linalg.norm(d, axis=-1)
    factor = np.minimum(1, .01 * math.hypot(h, w) / np.maximum(length, 1e-30))
    q[:8] = np.where(usable[:, None], q0[:8] + d * factor[:, None], q0[:8])
    q[8] = q0[8]
    return q, calls


def candidate_features(q, box, score, confidence, K, xyz, hw, geo, selector):
    c = dict(keypoints_xy=q.tolist(), box_xyxy=np.asarray(box).tolist(), score=float(score), keypoints_conf=np.asarray(confidence).tolist())
    if not np.isfinite(q[:8]).all():
        return np.zeros((2, 94), np.float32), False, -1, []
    try:
        output = selector.select_pnp_hypotheses(q, K,
            dict(x=float(max(xyz[0], xyz[2])), y=float(xyz[1]), z=float(min(xyz[0], xyz[2]))), None).to_dict()
        hh = sorted(output["hypotheses"], key=lambda h: h["name"])
        assert tuple(h["name"] for h in hh) == NAMES
        vv = [geo.vector(h, c, hw) for h in hh]
        valid = all(v is not None for v in vv)
        current = NAMES.index(output["selected_hypothesis"]) if output["selected_hypothesis"] in NAMES else -1
        return (np.stack(vv) if valid else np.zeros((2, 94), np.float32)), valid, current, hh
    except (cv2.error, ValueError, np.linalg.LinAlgError):
        return np.zeros((2, 94), np.float32), False, -1, []


def save_npz(path, **values):
    assert not path.exists()
    np.savez_compressed(path, **values)


def freeze_code():
    path = C.DOC / "C_CODE_LOCK.json"
    lock = dict(status="BEFORE_C_GPU_EXTRACTION_FITS_AND_SOLVERS", code=C.binding(Path(__file__)),
        arms=list(ARMS), geometry_dimensions=188, geometry="94 approved runtime-only features per official candidate, concatenated in physical long/short order",
        visual_dimensions=768, dimension_dimensions=5, common_MLP_input_dimensions=961,
        logistic=dict(C=1, max_iter=1000, fit_intercept=True, solver="lbfgs", class_weight=None,
            standardization="TRAIN only", penalty="L2; sklearn objective sum logloss + ||w||^2/(2C)"),
        MLP=dict(hidden=32, activation="ReLU", loss="BCEWithLogits", optimizer="AdamW", lr=.001, weight_decay=.0001,
            batch=256, max_epochs=30, patience=5, best="earliest minimum VAL BCE", seed=42,
            common_shapes=True, zero_unused_INPUTS_AFTER_STANDARDIZATION=True),
        tie="p==0.5 keeps official physical choice", one_candidate="choose available candidate", no_candidate="production fallback; count failure",
        extraction="synthetic current YOLO once per6144 source RGB to recover missing cached keypoint confidence; ROI and N3 use existing P3/P4",
        probe="first8 fixed already-selected hash records are included in6144 and never repeated",
        parity=dict(selected_index="exact", points_and_box_maxabs=1e-4, P3_P4_FP16_maxabs=0),
        fitting_calls=0, optimizer_updates=0, PnP_calls=0)
    C.write(path, lock)


def extract_synthetic():
    C.lock_ready()
    code = C.read(C.DOC / ("C_CODE_FINAL_LOCK.json" if (C.DOC / "C_CODE_FINAL_LOCK.json").exists() else "C_CODE_LOCK.json"))
    assert C.sha(Path(__file__)) == code["code"]["sha256"]
    R, _, geo, selector = modules()
    source_features = R.E.old("features")
    FrozenYoloFeatures, branch_inputs, canvas_affine = (getattr(source_features, k) for k in ("FrozenYoloFeatures", "branch_inputs", "canvas_affine"))
    records = C.read(C.PRIVATE / "C_SYNTH_RECORDS.json")
    source = C.read(C.SOURCE / "data/pallet/results/pallet_line_pose_v1/SOURCE_MANIFEST.json")["records"]
    a = arrays()
    side = np.load(C.SOURCE / "data/pallet/results/pallet_dim_conditioned_p_v1/DIMENSION_SIDECAR.npz")
    norm = C.read(C.PUBLIC / "_docs/experiments/pallet_dim_conditioned_p_v1/DIM_NORMALIZATION_LOCK.json")
    # Inference-only keys. R,t,Xcf are not opened until label/scoring phases.
    geom = np.load(C.SOURCE / "challenge/yolo_pose_one_model/pallet_translation_loss_v1/GEOMETRY_SIDETABLE.npz")
    gi = {str(x): i for i, x in enumerate(geom["stems"])}
    gp_K, gp_dims, gp_pad = geom["K"], geom["dims"], geom["pad"]
    cp_path = C.SOURCE / "data/pallet/results/pallet_dim_conditioned_p_v1/runs/N3_DIM_SYM_seed1/last.pt"
    cp = torch.load(cp_path, map_location="cpu", weights_only=False)
    assert cp["complete"] and cp["step"] == 6000
    head = R.model("N3_DIM_SYM", cp["config"]).cuda().eval().requires_grad_(False)
    head.load_state_dict(cp["model_state_dict"])
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = False
    detector = FrozenYoloFeatures()
    total = len(records)
    out = dict(ids=np.array([r["id"] for r in records]), split=np.array([r["split"] for r in records]),
        group=np.array([r["renderer_group"] for r in records]), V=np.zeros((total, 768), np.float32),
        D=np.zeros((total, 5), np.float32), q0=np.full((total, 9, 2), np.nan), qN=np.full((total, 9, 2), np.nan),
        qFinal=np.full((total, 9, 2), np.nan), K=np.zeros((total, 3, 3)), xyz=np.zeros((total, 3)),
        support=np.zeros((total, 9), bool), bbox=np.zeros((total, 4)), score=np.zeros(total),
        kpt_conf=np.zeros((total, 9)), hw=np.zeros((total, 2), int), visual_valid=np.zeros(total, bool),
        order=np.zeros(total, int))
    probe = []
    begun = time.monotonic()
    max_point = max_box = max_p3 = max_p4 = 0.
    subpix_calls = forward_count = 0
    with torch.no_grad():
        for j, r in enumerate(records):
            i = r["record_index"]
            src = source[i]
            assert src["id"] == r["id"]
            assert C.sha(src["image"]) == r["image_sha256"]
            image = cv2.imread(src["image"])
            assert list(image.shape[:2]) == src["prepared_shape_hw"]
            fresh = detector.predict(image, already_padded=True, cpu_features=False)
            forward_count += 1
            inp = branch_inputs(fresh)
            old_index = int(a["selected_candidate_index"][i])
            assert fresh["selected_index"] == (None if old_index < 0 else old_index)
            p3_diff = float(np.max(np.abs(fresh["p3"][0].cpu().numpy().astype(float) - np.asarray(a["p3"][i], float))))
            p4_diff = float(np.max(np.abs(fresh["p4"][0].cpu().numpy().astype(float) - np.asarray(a["p4"][i], float))))
            max_p3 = max(max_p3, p3_diff); max_p4 = max(max_p4, p4_diff)
            assert p3_diff == 0 and p4_diff == 0, "Frozen feature/preprocessing parity failed"
            k = gi[r["id"]]
            fx, fy, cx, cy = gp_K[k]; pad = float(gp_pad[k])
            K = np.array([[fx, 0, cx-pad], [0, fy, cy-pad], [0, 0, 1.]])
            xyz = gp_dims[k]
            assert np.allclose(side["dimensions"][i][[0, 2, 1]], xyz, atol=1e-7, rtol=1e-6)
            out["K"][j] = K; out["xyz"][j] = xyz; out["order"][j] = side["order"][i]
            out["hw"][j] = src["raw_shape_hw"]
            out["D"][j] = R.context(side["dimensions"][i:i+1], side["order"][i:i+1], norm, False)[0]
            if inp is not None:
                pd = float(np.max(np.abs(inp["points"] - a["points"][i])))
                bd = float(np.max(np.abs(inp["boxes"] - a["boxes"][i])))
                max_point = max(max_point, pd); max_box = max(max_box, bd)
                assert pd <= 1e-4 and bd <= 1e-4
                assert np.array_equal(inp["input_shape"], a["input_shape"][i]) and inp["gain"] == a["gain"][i]
                batch = {key: torch.from_numpy(np.array(a[key][i:i+1], copy=True)).cuda() for key in
                         ("p3", "p4", "points", "boxes", "point_valid", "input_shape")}
                batch["context"] = torch.from_numpy(out["D"][j:j+1]).cuda()
                o = R.forward(head, batch)
                cap = .01 * math.hypot(*src["raw_shape_hw"]) * inp["gain"]
                decoded = R.decode(o, 1., 1., cap)[0].cpu().numpy()
                candidate = fresh["candidates"][fresh["selected_index"]]
                q0 = np.asarray(candidate["keypoints_xy"], float) - pad
                qn = q0 + (decoded.astype(float) - np.asarray(a["points"][i], float)) / inp["gain"]
                qn[8] = q0[8]
                raw = image[int(pad):image.shape[0]-int(pad), int(pad):image.shape[1]-int(pad)]
                assert list(raw.shape[:2]) == src["raw_shape_hw"]
                qf, n_calls = subpix(cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY), q0, qn, np.asarray(a["point_valid"][i]))
                subpix_calls += n_calls
                gain, offset = canvas_affine(src["prepared_shape_hw"], a["input_shape"][i])
                box = (np.asarray(a["boxes"][i], float).reshape(2, 2) - offset) / gain - pad
                out["bbox"][j] = box.ravel(); out["score"][j] = candidate["score"]
                out["kpt_conf"][j] = candidate["keypoints_conf"]
                out["q0"][j] = q0; out["qN"][j] = qn; out["qFinal"][j] = qf
                out["support"][j] = a["point_valid"][i]
                out["V"][j], out["visual_valid"][j] = roi_vector(a["p3"][i], a["p4"][i], a["boxes"][i], a["input_shape"][i])
            if j < 8:
                probe.append(dict(id=r["id"], points_maxabs=pd if inp is not None else None,
                    bbox_maxabs=bd if inp is not None else None, P3_maxabs=p3_diff, P4_maxabs=p4_diff,
                    selected_index_exact=True, same_source_image_forward_count=1))
                if j == 7:
                    C.write(C.DOC / "C_EXTRACTION_FIRST8.json", dict(pass_parity=True, frames=probe,
                        elapsed_seconds=time.monotonic()-begun, model_forward_images=8, included_in_full_population=True))
                    print("C_PARITY8_PASS", round(time.monotonic()-begun, 3), flush=True)
            if (j+1) % 256 == 0:
                print("C_EXTRACT", j+1, total, round(time.monotonic()-begun, 1), flush=True)
    assert all(p.grad is None and not p.requires_grad for p in head.parameters())
    assert all(p.grad is None and not p.requires_grad for p in detector.yolo.model.parameters())
    assert C.sha(cp_path) == C.binding(cp_path)["sha256"]
    detector.close()
    elapsed = time.monotonic()-begun
    save_npz(C.PRIVATE / "C_SYNTH_PREDICTIONS_SEALED.npz", **out)
    C.write(C.DOC / "C_EXTRACTION_RECEIPT.json", dict(frozen_YOLO_forward_images=forward_count,
        N3_forward_images=int(out["visual_valid"].sum()), reused_P3_P4_cache=True, fresh_features_used_as_predictor=False,
        current_confidence_sidecar=True, raw_RGB_loaded_once_per_frame=True, subpix_corner_calls=subpix_calls,
        GPU_device_seconds=elapsed, peak_allocated_GPU_bytes=torch.cuda.max_memory_allocated(),
        parity=dict(points_maxabs=max_point, bbox_maxabs=max_box, P3_maxabs=max_p3, P4_maxabs=max_p4),
        frozen_model_parameters_no_grad=True, checkpoint=C.binding(cp_path), rows=total,
        outputs_sealed_before_renderer_labels=True))


def geometries():
    C.lock_ready()
    R, _, geo, selector = modules()
    packet = dict(np.load(C.PRIVATE / "C_SYNTH_PREDICTIONS_SEALED.npz", allow_pickle=False))
    n = len(packet["ids"])
    G = np.zeros((n, 2, 94), np.float32); valid = np.zeros(n, bool); current = np.full(n, -1, int)
    with C.pnp_counter("C", C.read(C.DOC / "PLAN_LOCK.json")["pnp_reservations"]["C"]) as counts:
        for j in range(n):
            G[j], valid[j], current[j], _ = candidate_features(packet["qFinal"][j], packet["bbox"][j], packet["score"][j],
                packet["kpt_conf"][j], packet["K"][j], packet["xyz"][j], packet["hw"][j], geo, selector)
            if (j+1) % 1024 == 0: print("C_SYNTH_GEOMETRY", j+1, n, sum(counts.values()), flush=True)
        real = real_features(R, geo, selector)
    save_npz(C.PRIVATE / "C_SYNTH_GEOMETRY.npz", ids=packet["ids"], G=G, valid=valid, official=current)
    save_npz(C.PRIVATE / "C_REAL_FEATURES.npz", **real)
    C.write(C.DOC / "C_GEOMETRY_RECEIPT.json", dict(synthetic_rows=n, valid_synthetic=int(valid.sum()),
        real_rows=len(real["ids"]), valid_real=int(real["valid"].sum()), runtime_only_features=True,
        both_candidates_same_points=True, predictor_GT_fields=False, actual_OpenCV_calls=sum(counts.values()),
        geometry_names=geo.names(), dimensions=94))


def real_features(R, geo, selector):
    allowed = set(C.read(C.DOC / "INPUT_CONTRACT.json")["allowed_ids"])
    prediction = {}
    source_path = C.PUBLIC / "_docs/experiments/pallet_feature_gradient_joint_20261010/PREDICTIONS.jsonl.gz"
    for r in C.rows(source_path):
        if r["id"] in allowed and r["method"] == "N3_THEN_SUBPIX":
            prediction[(int(r["seed"]), r["id"])] = {k: r[k] for k in
                ("id", "seed", "qFinal", "prediction_support", "raw_hw", "fixed_metadata", "session", "grade")}
    assert len(prediction) == len(allowed) * 3
    receipt = C.read(C.SOURCE / "_docs/experiments/pallet_dim_conditioned_p_v1/DEV_CACHE_COMPLETE.json")
    cache = {r["id"]: r for r in receipt["records"] if r["id"] in allowed}
    visual = {}; D = {}; boxes = {}; candidate = {}
    norm = C.read(C.PUBLIC / "_docs/experiments/pallet_dim_conditioned_p_v1/DIM_NORMALIZATION_LOCK.json")
    oldfeatures = R.E.old("features")
    for fid in sorted(allowed):
        row = torch.load(C.SOURCE / cache[fid]["path"], map_location="cpu", weights_only=False)
        assert row["id"] == fid and row["GT_input"] is False
        cap = row["captured"]; inp = oldfeatures.branch_inputs(cap)
        if inp is None:
            visual[fid] = (np.zeros(768, np.float32), False); candidate[fid] = None
        else:
            visual[fid] = roi_vector(cap["p3"], cap["p4"], inp["boxes"], inp["input_shape"])
            candidate[fid] = cap["candidates"][cap["selected_index"]]
        D[fid] = R.context(np.asarray(row["dimensions"])[None], np.array([row["order"]]), norm, False)[0]
    keys = sorted(prediction)
    result = dict(ids=np.array([fid for s, fid in keys]), seeds=np.array([s for s, fid in keys]),
        V=np.zeros((len(keys), 768), np.float32), D=np.zeros((len(keys), 5), np.float32),
        G=np.zeros((len(keys), 2, 94), np.float32), valid=np.zeros(len(keys), bool),
        official=np.full(len(keys), -1, int), session=np.array([prediction[k]["session"] for k in keys]),
        grade=np.array([prediction[k]["grade"] for k in keys]))
    for j, key in enumerate(keys):
        seed, fid = key; pred = prediction[key]; c = candidate[fid]
        result["V"][j] = visual[fid][0]; result["D"][j] = D[fid]
        if c is None: continue
        meta = pred["fixed_metadata"]
        result["G"][j], ok, result["official"][j], _ = candidate_features(
            np.array(pred["qFinal"], float), c["box_xyxy"], c["score"], c["keypoints_conf"],
            np.array(meta["K"]), np.array(meta["dimensions_pnp_WH_D_m"]), pred["raw_hw"], geo, selector)
        result["valid"][j] = ok and visual[fid][1]
    return result


def loaded_npz(path):
    """Decompress each requested input array once before any row loop."""
    with np.load(path, allow_pickle=False) as packet:
        return {key: packet[key] for key in packet.files}


def label_link():
    pred = loaded_npz(C.PRIVATE / "C_SYNTH_PREDICTIONS_SEALED.npz")
    geometry = loaded_npz(C.SOURCE / "challenge/yolo_pose_one_model/pallet_translation_loss_v1/GEOMETRY_SIDETABLE.npz")
    ix = {str(s): i for i, s in enumerate(geometry["stems"])}
    Xcf = geometry["Xcf"]
    labels = np.full(len(pred["ids"]), -1, int); elevations = np.full(len(labels), np.nan)
    sizes = np.full(len(labels), np.nan)
    for j, fid in enumerate(pred["ids"]):
        i = ix[str(fid)]; X = Xcf[i]; xyz = geometry["dims"][i]
        width = np.linalg.norm(X[1]-X[0]); depth = np.linalg.norm(X[4]-X[0]); height = np.linalg.norm(X[3]-X[0])
        assert np.allclose(sorted([width, depth]), sorted(xyz[[0, 2]]), atol=1e-6, rtol=0) and abs(height-xyz[1]) < 1e-6
        if abs(width-depth) > 1e-6: labels[j] = int(width > depth)
        up = (X[[0, 1, 4, 5]].mean(0)-X[[2, 3, 6, 7]].mean(0)) @ geometry["R"][i].T
        ray = -geometry["t"][i]
        elevations[j] = np.degrees(np.arcsin(np.clip(up@ray/(np.linalg.norm(up)*np.linalg.norm(ray)), -1, 1)))
        cam = X @ geometry["R"][i].T + geometry["t"][i]
        projected = cam @ pred["K"][j].T
        q = projected[:, :2] / projected[:, 2:]
        sizes[j] = np.prod(q.max(0)-q.min(0)) / np.prod(pred["hw"][j])
    save_npz(C.PRIVATE / "C_SYNTH_LABELS.npz", ids=pred["ids"], y=labels, elevation=elevations, size=sizes)
    summary = {}
    for part in ("TRAIN", "VAL", "TEST"):
        mask = pred["split"] == part
        summary[part] = dict(rows=int(mask.sum()), long=int((labels[mask] == 1).sum()), short=int((labels[mask] == 0).sum()),
            square_or_unclear=int((labels[mask] < 0).sum()), elevation_quantiles=np.quantile(elevations[mask], [0, .1, .5, .9, 1]).tolist(),
            high_elevation_ge30_count=int((elevations[mask] >= 30).sum()))
    C.write(C.DOC / "C_LABEL_AUDIT.json", dict(label="y_long==1 iff exact Xcf edge01 longer than edge04", splits=summary,
        square_or_unclear_excluded_from_fitting=True, renderer_labels_read_after_prediction_seal=True,
        canonical_dimensions_not_camera_facing=True, original_base_N3_exposure="Previously developed synthetic; test is not independent final"))


def prep_inputs(data, mean, scale, arm):
    if arm.startswith("D_"): x = data["D"]
    elif arm.startswith("G_") and arm.endswith("LOGISTIC"): x = data["G"].reshape(len(data["G"]), -1)
    elif arm.startswith("V_"): x = data["V"]
    else: x = np.concatenate([data["G"].reshape(len(data["G"]), -1), data["V"], data["D"]], axis=1)
    z = (x - mean) / scale
    if arm == "G_MLP": z[:, 188:] = 0
    if arm == "G_V_MLP": z[:, -5:] = 0
    return z.astype(np.float32)


def classification(y, p):
    from sklearn.metrics import balanced_accuracy_score, confusion_matrix, brier_score_loss
    if not len(y): return dict(n=0)
    pred = p > .5
    return dict(n=len(y), accuracy=float((pred == y).mean()), balanced_accuracy=float(balanced_accuracy_score(y, pred)),
        confusion_matrix_true_rows_short_long=confusion_matrix(y, pred, labels=[0, 1]).tolist(),
        Brier=float(brier_score_loss(y, p)))


def fit():
    C.lock_ready()
    assert C.sha(Path(__file__)) == C.read(C.DOC / "C_IO_PREFIT_LOCK.json")["code"]["sha256"]
    from sklearn.linear_model import LogisticRegression
    from sklearn.exceptions import ConvergenceWarning
    from threadpoolctl import threadpool_limits
    torch.set_num_threads(2)
    threadpool_limits(2)
    base = dict(loaded_npz(C.PRIVATE / "C_SYNTH_PREDICTIONS_SEALED.npz"))
    geom = loaded_npz(C.PRIVATE / "C_SYNTH_GEOMETRY.npz")
    labels = loaded_npz(C.PRIVATE / "C_SYNTH_LABELS.npz")
    real = dict(loaded_npz(C.PRIVATE / "C_REAL_FEATURES.npz"))
    assert np.array_equal(base["ids"], geom["ids"]) and np.array_equal(base["ids"], labels["ids"])
    base["G"] = geom["G"]
    valid = geom["valid"] & base["visual_valid"] & (labels["y"] >= 0)
    train, val = (np.flatnonzero(valid & (base["split"] == part)) for part in ("TRAIN", "VAL"))
    assert len(train) and len(val)
    assert set(base["group"][train]).isdisjoint(set(base["group"][val]))
    y = labels["y"]
    scores = {}; fits = {}; all_syn = {}; all_real = {}
    begun = time.monotonic()
    for arm in ARMS:
        if arm.startswith("D_"): raw = base["D"]
        elif arm.startswith("G_") and arm.endswith("LOGISTIC"): raw = base["G"].reshape(len(y), -1)
        elif arm.startswith("V_"): raw = base["V"]
        else: raw = np.concatenate([base["G"].reshape(len(y), -1), base["V"], base["D"]], axis=1)
        mean, scale = raw[train].mean(0), raw[train].std(0)
        scale = np.maximum(scale, 1e-6)
        X = prep_inputs(base, mean, scale, arm)
        XR = prep_inputs(real, mean, scale, arm)
        fit_start = time.monotonic()
        if arm.endswith("LOGISTIC"):
            target = y[train].copy()
            if arm == "V_SHUFFLED_LOGISTIC": target = np.random.default_rng(42).permutation(target)
            model = LogisticRegression(C=1, fit_intercept=True, solver="lbfgs", penalty="l2", class_weight=None, max_iter=1000)
            with warnings.catch_warnings(record=True) as logs:
                warnings.simplefilter("always", ConvergenceWarning)
                model.fit(X[train], target)
            all_syn[arm] = model.predict_proba(X)[:, 1]
            all_real[arm] = model.predict_proba(XR)[:, 1]
            fit_receipt = dict(kind="logistic", n_iter=int(model.n_iter_[0]), converged=not any(issubclass(w.category, ConvergenceWarning) for w in logs),
                warnings=[str(w.message) for w in logs], parameter_count=int(model.coef_.size + model.intercept_.size))
            torch.save(dict(coef=model.coef_, intercept=model.intercept_, classes=model.classes_, mean=mean, scale=scale), C.PRIVATE / (arm + ".pt"))
        else:
            torch.manual_seed(42)
            model = nn.Sequential(nn.Linear(X.shape[1], 32), nn.ReLU(), nn.Linear(32, 1))
            initial = {k: v.detach().clone() for k, v in model.state_dict().items()}
            optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
            rng = np.random.default_rng(42)
            xtrain = torch.from_numpy(X[train]); ttrain = torch.from_numpy(y[train].astype(np.float32))
            xval = torch.from_numpy(X[val]); tval = torch.from_numpy(y[val].astype(np.float32))
            best = float("inf"); best_state = None; best_epoch = 0; stale = 0; history = []; updates = 0
            for epoch in range(1, 31):
                model.train()
                shuffled = rng.permutation(len(train))
                for begin in range(0, len(train), 256):
                    ii = shuffled[begin:begin+256]
                    optimizer.zero_grad(set_to_none=True)
                    value = TF.binary_cross_entropy_with_logits(model(xtrain[ii]).squeeze(-1), ttrain[ii])
                    value.backward(); optimizer.step(); updates += 1
                model.eval()
                with torch.no_grad(): loss = float(TF.binary_cross_entropy_with_logits(model(xval).squeeze(-1), tval))
                history.append(dict(epoch=epoch, VAL_BCE=loss))
                if loss < best:
                    best = loss; best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}; best_epoch=epoch; stale=0
                else: stale += 1
                if stale >= 5: break
            model.load_state_dict(best_state); model.eval()
            with torch.no_grad():
                all_syn[arm] = model(torch.from_numpy(X)).squeeze(-1).sigmoid().numpy()
                all_real[arm] = model(torch.from_numpy(XR)).squeeze(-1).sigmoid().numpy()
            ih = hashlib.sha256()
            for key, value in sorted(initial.items()): ih.update(key.encode()); ih.update(value.numpy().tobytes())
            fit_receipt = dict(kind="MLP", input_dimensions=X.shape[1], parameter_count=sum(v.numel() for v in model.parameters()),
                initial_state_sha256=ih.hexdigest(), best_epoch=best_epoch, epochs=epoch, optimizer_updates=updates,
                earliest_best_VAL_BCE=best, history=history)
            torch.save(dict(state_dict=best_state, mean=mean, scale=scale, arm=arm), C.PRIVATE / (arm + ".pt"))
        fit_receipt.update(TRAIN_frames=len(train), VAL_frames=len(val), fit_seconds=time.monotonic()-fit_start,
            checkpoint=C.binding(C.PRIVATE / (arm + ".pt")), real_GT_fitting=False)
        fits[arm] = fit_receipt
        scores[arm] = {part: classification(y[mask], all_syn[arm][mask]) for part in ("TRAIN", "VAL", "TEST")
                      if len(mask := np.flatnonzero(valid & (base["split"] == part)))}
        print("C_FIT", arm, fit_receipt.get("best_epoch", fit_receipt.get("n_iter")), round(fit_receipt["fit_seconds"], 2), flush=True)
    for arm in ("G_V_MLP", "G_V_D_MLP"):
        assert fits[arm]["parameter_count"] == fits["G_MLP"]["parameter_count"]
        assert fits[arm]["initial_state_sha256"] == fits["G_MLP"]["initial_state_sha256"]
    prior = float(y[train].mean())
    all_syn["TRAIN_PRIOR"] = np.full(len(y), prior)
    all_real["TRAIN_PRIOR"] = np.full(len(real["ids"]), prior)
    save_npz(C.PRIVATE / "C_SYNTH_PROBABILITIES_SEALED.npz", ids=base["ids"], split=base["split"], valid=valid, **all_syn)
    save_npz(C.PRIVATE / "C_REAL_PROBABILITIES_SEALED.npz", ids=real["ids"], seeds=real["seeds"], valid=real["valid"], **all_real)
    C.write(C.DOC / "C_FITS.json", dict(fits=fits, completed_fit_count=8, synthetic_classification=scores,
        split_valid_counts={part: int((valid & (base["split"] == part)).sum()) for part in ("TRAIN", "VAL", "TEST")},
        common_fit_population=True, majority_long_probability=prior, probability_seal_before_real_truth=True,
        CPU_fit_wall_seconds=time.monotonic()-begun, high_cost_visual_extractions_during_fitting=0))


def summarize(records):
    from collections import defaultdict
    rows = list(records); by = defaultdict(list)
    for r in rows: by[r["id"]].append(r)
    frames = list(by.values())
    if not frames: return dict(frames=0, prediction_rows=0)
    axis = [np.mean([r["axis_error"] for r in rr if r["axis_error"] is not None]) for rr in frames if any(r["axis_error"] is not None for r in rr)]
    available = [np.mean([r["pose"].get("available",False) for r in rr]) for rr in frames]
    out = dict(frames=len(frames), prediction_rows=len(rows), available_seed_mean_count=float(sum(available)),
        available_prediction_rows=sum(r["pose"].get("available",False) for r in rows), coverage=float(np.mean(available)),
        strict_success=float(np.mean([np.mean([r["success"] for r in rr]) for rr in frames])),
        axis_evaluable_frames=len(axis), axis_error_rate=float(np.mean(axis)) if axis else None,
        legacy_WD_confusion=float(np.mean([np.mean([r["legacy_confusion"] for r in rr]) for rr in frames])),
        aggregation="Mean fixed seeds within frame first; moments/quantiles across frames. Missing pose success=0, UNCLEAR axis=NA.")
    for key in ("translation_cm", "rotation_deg", "yaw_deg", "IoU3D", "ADDsym_normalized"):
        values = np.array([np.mean([r["pose"][key] for r in rr if r["pose"].get("available")])
                           for rr in frames if any(r["pose"].get("available") for r in rr)], float)
        out[key] = dict(n=len(values), mean=float(values.mean()) if len(values) else None,
            SD=float(values.std(ddof=1)) if len(values)>1 else None, variance=float(values.var(ddof=1)) if len(values)>1 else None,
            median=float(np.median(values)) if len(values) else None, P90=float(np.quantile(values,.9)) if len(values) else None,
            P95=float(np.quantile(values,.95)) if len(values) else None)
    # Same existing 1001-threshold corresponding-proper-group ADDsym AUC.
    thresholds = np.linspace(0, .1, 1001)
    curves = []
    for rr in frames:
        adds = np.array([r["pose"].get("ADDsym_normalized",np.inf) if r["pose"].get("available") else np.inf for r in rr])
        curves.append((adds[:,None] <= thresholds).mean(0))
    out["ADDsym_AUC_full"] = float(np.trapz(np.mean(curves,axis=0), thresholds) / .1)
    return out


def evaluate():
    C.lock_ready()
    probs = loaded_npz(C.PRIVATE / "C_REAL_PROBABILITIES_SEALED.npz")
    # All predictor states/choices are sealed before reference-containing rows.
    metrics = list(C.rows(C.DOC / "AE_CANDIDATE_METRICS.jsonl.gz"))
    by = {(int(r["seed"]), r["id"]): r for r in metrics if r["method"] == "N3_THEN_SUBPIX"}
    assert len(by) == 519
    row_records = []
    changes = []
    arm_summaries = {}
    allowed = C.read(C.DOC / "INPUT_CONTRACT.json")
    unclear = set(allowed["unclear_allowed_ids"])
    for arm in (*ARMS, "TRAIN_PRIOR", "OFFICIAL"):
        arm_rows = []
        for j, (fid0, seed) in enumerate(zip(probs["ids"], probs["seeds"])):
            fid = str(fid0); r = by[(int(seed), fid)]
            candidates = r["candidate_metrics"]
            official = r["hyp_current"]
            p = None if arm == "OFFICIAL" else float(probs[arm][j])
            successes = [name for name in NAMES if candidates[name].get("available")]
            if len(successes) == 1: chosen = successes[0]
            elif len(successes) == 0: chosen = official
            elif arm == "OFFICIAL" or not bool(probs["valid"][j]) or p == .5: chosen = official
            else: chosen = NAMES[0] if p > .5 else NAMES[1]
            pose = candidates.get(chosen, {"available": False})
            gt = r["hyp_reference"]
            success = bool(pose.get("available") and pose["translation_cm"] < 5 and pose["rotation_deg"] < 5)
            axis_error = None if fid in unclear or chosen is None else int(chosen != gt)
            legacy = int(pose.get("available") and pose["rotation_deg"] > 45 and abs(pose["yaw_deg"]) >= 60)
            row = dict(method=arm, id=fid, seed=int(seed), session=r["session"], grade=r["grade"],
                selected_hypothesis=chosen, hyp=chosen, official_hypothesis=official, p_long=p, axis_error=axis_error,
                success=int(success), legacy_confusion=legacy, pose=pose, candidates=candidates,
                axis_reference_status=r["axis_reference_status"],
                reference_margin_px=r.get("reference_margin_px"), elevation_deg=r.get("elevation_deg"),
                distance_m=r.get("distance_m"), material=r.get("material"))
            arm_rows.append(row); row_records.append(row)
            if chosen != official:
                old = candidates.get(official, {"available": False})
                changes.append(dict(method=arm, id=fid, seed=int(seed), from_hypothesis=official, to_hypothesis=chosen,
                    axis_transition="UNCLEAR" if fid in unclear else "RECOVERY" if chosen==gt else "DAMAGE",
                    old_success=int(old.get("available") and old["translation_cm"]<5 and old["rotation_deg"]<5), new_success=int(success),
                    long_T=candidates[NAMES[0]].get("translation_cm"), long_R=candidates[NAMES[0]].get("rotation_deg"),
                    short_T=candidates[NAMES[1]].get("translation_cm"), short_R=candidates[NAMES[1]].get("rotation_deg")))
        arm_summaries[arm] = {"REAL173": summarize(arm_rows), "RECT245_INTERSECTION110": summarize([r for r in arm_rows if r["id"] in set(allowed["rect245_intersection_ids"])])}
        for subgroup, value in (("grade", ["clean", "moderate", "severe"]), ("session", sorted(set(r["session"] for r in arm_rows)))):
            arm_summaries[arm][subgroup] = {k: summarize([r for r in arm_rows if r[subgroup]==k]) for k in value if any(r[subgroup]==k for r in arm_rows)}
        arm_summaries[arm]["material"] = {k:summarize([r for r in arm_rows if r["material"]==k]) for k in sorted(set(r["material"] for r in arm_rows))}
        arm_summaries[arm]["reference_margin_lt4_descriptive"] = summarize(r for r in arm_rows if r["reference_margin_px"] is not None and r["reference_margin_px"]<4)
        arm_summaries[arm]["elevation_descriptive_fixed_bins"] = {name:summarize(r for r in arm_rows if r["elevation_deg"] is not None and low<=r["elevation_deg"]<high)
            for name,low,high in (("lt5",0,5),("5to15",5,15),("ge15",15,math.inf))}
        from sklearn.metrics import confusion_matrix, balanced_accuracy_score
        evaluable = [r for r in arm_rows if r["axis_reference_status"]!="UNCLEAR" and r["selected_hypothesis"] is not None]
        yy = np.array([int(by[(r["seed"],r["id"])]["hyp_reference"]==NAMES[0]) for r in evaluable])
        guessed = np.array([int(r["selected_hypothesis"]==NAMES[0]) for r in evaluable])
        arm_summaries[arm]["long_short_classification"] = dict(frames=len({r["id"] for r in evaluable}), prediction_rows=len(evaluable),
            accuracy=float((yy==guessed).mean()), balanced_accuracy=float(balanced_accuracy_score(yy,guessed)),
            confusion_matrix_seed_rows_true_short_long=confusion_matrix(yy,guessed,labels=[0,1]).tolist(),
            Brier=None if arm=="OFFICIAL" else float(np.mean((np.array([r["p_long"] for r in evaluable])-yy)**2)),
            seed_rows_not_independent_for_CI=True, fallback_included=True)
    with (C.DOC / "C_FRAME_METRICS.jsonl.gz").open("wb") as f:
        import gzip
        with gzip.open(f, "wt", compresslevel=6) as z:
            for r in row_records: z.write(json.dumps(C.clean(r), ensure_ascii=False, separators=(",",":"), allow_nan=False)+"\n")
    path = C.DOC / "C_CHANGED_SELECTIONS.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(changes[0]) if changes else ["method", "id", "seed"])
        writer.writeheader(); writer.writerows(changes)
    C.write(C.DOC / "C_RESULTS.json", dict(status="EXPLORATORY_ALL_FIXED_ARMS_REPORTED", summaries=arm_summaries,
        changed_selection_rows=len(changes), unchanged_corner_and_candidate_cache=True, real_pnp_calls_for_scoring=0,
        seed_predictions_not_independent_frames=True, full_frames=173, sessions=9, axis_reference_frames=170,
        axis_error_rate_definition="selected long/short name != reference name; excludes UNCLEAR3",
        legacy_WD_confusion_definition="R>45 degrees AND absolute yaw>=60 degrees; separate from direct classifier axis error"))


def synthetic_evaluate():
    """Solve both final8-corner candidates once per TEST frame, then reuse them."""
    C.lock_ready()
    from scripts.research.pallet_wd_info_fast_20261011.geometry_analysis import pose_api
    from scripts.research.pallet_wd_hypothesis_diag_20261010 import solver
    pose = pose_api(C.SOURCE); solver.configure(pose)
    pred = loaded_npz(C.PRIVATE / "C_SYNTH_PREDICTIONS_SEALED.npz")
    geo = loaded_npz(C.PRIVATE / "C_SYNTH_GEOMETRY.npz")
    probs = loaded_npz(C.PRIVATE / "C_SYNTH_PROBABILITIES_SEALED.npz")
    labels = loaded_npz(C.PRIVATE / "C_SYNTH_LABELS.npz")
    exact = loaded_npz(C.SOURCE / "challenge/yolo_pose_one_model/pallet_translation_loss_v1/GEOMETRY_SIDETABLE.npz")
    ix = {str(fid): i for i, fid in enumerate(exact["stems"])}
    test = np.flatnonzero(pred["split"] == "TEST")
    rows = []
    prior_calls = C.read(C.DOC / "C_GEOMETRY_RECEIPT.json")["actual_OpenCV_calls"]
    with C.pnp_counter("C", C.read(C.DOC / "PLAN_LOCK.json")["pnp_reservations"]["C"]) as calls:
        for j in test:
            fid = str(pred["ids"][j]); i = ix[fid]
            truth = dict(R=exact["R"][i], t=exact["t"][i], xyz=pred["xyz"][j],
                         body_R=exact["R"][i], body_xyz=pred["xyz"][j], order=int(pred["order"][j]))
            subset = np.flatnonzero(np.isfinite(pred["qFinal"][j, :8]).all(-1)).tolist()
            candidates = {}
            for name in NAMES:
                fitted = solver.fit(pred["qFinal"][j], pred["K"][j], pred["xyz"][j], name, subset, True)
                candidates[name] = pose.metric((fid, fitted["actual_pose"], truth))
            assert prior_calls + sum(calls.values()) <= 31000
            official = NAMES[int(geo["official"][j])] if int(geo["official"][j]) >= 0 else None
            target = NAMES[0] if int(labels["y"][j]) == 1 else NAMES[1]
            available = [name for name in NAMES if candidates[name].get("available")]
            for arm in (*ARMS, "TRAIN_PRIOR", "OFFICIAL"):
                p = None if arm == "OFFICIAL" else float(probs[arm][j])
                if len(available) == 1: chosen = available[0]
                elif len(available) == 0: chosen = official
                elif arm == "OFFICIAL" or not bool(probs["valid"][j]) or p == .5: chosen = official
                else: chosen = NAMES[0] if p > .5 else NAMES[1]
                m = candidates.get(chosen, {"available": False})
                rows.append(dict(method=arm, id=fid, seed=1, session=str(pred["group"][j]),
                    selected_hypothesis=chosen, p_long=p, pose=m,
                    axis_error=int(chosen != target) if chosen is not None and labels["y"][j]>=0 else None,
                    success=int(m.get("available") and m["translation_cm"] < 5 and m["rotation_deg"] < 5),
                    legacy_confusion=int(m.get("available") and m["rotation_deg"]>45 and abs(m["yaw_deg"])>=60)))
    C.write(C.DOC / "C_SYNTH_TEST_RESULTS.json", dict(frames=len(test), synthetic_renderer_group_count=len(set(pred["group"][test])),
        independent_final_test=False, each_candidate_fitted_once=True, actual_OpenCV_calls=sum(calls.values()),
        total_C_OpenCV_calls=prior_calls+sum(calls.values()), summaries={arm: summarize(r for r in rows if r["method"]==arm) for arm in (*ARMS,"TRAIN_PRIOR","OFFICIAL")}))
    import gzip
    with gzip.open(C.DOC / "C_SYNTH_TEST_ROWS.jsonl.gz", "wt", compresslevel=6) as f:
        for r in rows: f.write(json.dumps(C.clean(r), ensure_ascii=False, separators=(",",":"), allow_nan=False)+"\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("phase", choices=("lock", "extract", "geometry", "labels", "fit", "evaluate", "synthetic-evaluate"))
    args = p.parse_args()
    torch.set_num_threads(2); cv2.setNumThreads(1)
    if args.phase == "lock": freeze_code()
    elif args.phase == "extract": extract_synthetic()
    elif args.phase == "geometry": geometries()
    elif args.phase == "labels": label_link()
    elif args.phase == "fit": fit()
    elif args.phase == "synthetic-evaluate": synthetic_evaluate()
    else: evaluate()


if __name__ == "__main__":
    main()
