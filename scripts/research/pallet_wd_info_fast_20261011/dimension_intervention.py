"""B1: fixed seed1 N3 logits and frozen visual inputs; correct PnP geometry.

The original N3 metadata residual is independent of image features. Replacing
only that residual reuses the stored actual full logits, avoiding a new CNN.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.research.pallet_wd_info_fast_20261011 import common as C
from scripts.research.pallet_wd_info_fast_20261011.classifier import modules, summarize
from scripts.research.pallet_wd_info_fast_20261011.geometry_analysis import pose_api, truth_real


def residual(head, context):
    z = torch.as_tensor(np.array(context, copy=True), dtype=torch.float32).reshape(1, 5)
    emb = head.metadata_encoder(z)
    role = head.role_embedding.weight[None, :, None].expand(1, 8, 222, 8)
    disp = (head.displacements/.08)[None, None].expand(1, 8, 222, 2)
    null = torch.zeros((1, 8, 222, 1)); null[:, :, -1] = 1
    x = torch.cat([emb[:, None, None].expand(1, 8, 222, 16), role, disp, null], -1)
    return head.metadata_scorer(x).squeeze(-1)[0]


def run():
    C.lock_ready()
    C.write(C.DOC / "B1_CODE_LOCK.json", dict(code=C.binding(Path(__file__)), seed=1, arms=["CORRECT", "TRAIN_MEAN", "NORMALIZED_ZERO", "TRAIN_SHUFFLE"],
        limits=dict(new_OpenCV_calls=3200, new_CNN_forwards=0, fitting=0), correct_PnP_dimensions=True,
        shuffled_donor="hash('20261011|'+ID) modulo fixed original TRAIN dimensions; source population not reference pose",
        source_reconstruction="full actual stored logits + (new_metadata_residual - correct_metadata_residual), all original 222 candidates",
        outputs_saved_before_reference_scoring=True))
    R, _, _, _ = modules()
    allowed = set(C.read(C.DOC / "INPUT_CONTRACT.json")["allowed_ids"])
    rows = {}
    for r in C.rows(C.PUBLIC / "_docs/experiments/pallet_feature_gradient_joint_20261010/PREDICTIONS.jsonl.gz"):
        if r["id"] in allowed and int(r["seed"]) == 1 and r["method"] == "N3_DIM_SYM":
            rows[r["id"]] = {k: r[k] for k in ("id", "session", "grade", "q0", "qFinal", "pose", "fixed_metadata", "raw_hw")}
    assert len(rows) == 173
    norm = C.read(C.PUBLIC / "_docs/experiments/pallet_dim_conditioned_p_v1/DIM_NORMALIZATION_LOCK.json")
    side = np.load(C.SOURCE / "data/pallet/results/pallet_dim_conditioned_p_v1/DIMENSION_SIDECAR.npz", allow_pickle=False)
    source = C.read(C.SOURCE / "data/pallet/results/pallet_line_pose_v1/SOURCE_MANIFEST.json")["records"]
    train = np.array([r["partition"] == "train" for r in source])
    train_dims = side["dimensions"][train]
    mean_dims = train_dims.mean(0)
    cp_path = C.SOURCE / "data/pallet/results/pallet_dim_conditioned_p_v1/runs/N3_DIM_SYM_seed1/last.pt"
    cp_binding = C.binding(cp_path)
    cp = torch.load(cp_path, map_location="cpu", weights_only=False)
    head = R.model("N3_DIM_SYM", cp["config"]).eval().requires_grad_(False)
    head.load_state_dict(cp["model_state_dict"])
    extractor = R.E.old("features")
    cache = C.read(C.SOURCE / "_docs/experiments/pallet_dim_conditioned_p_v1/DEV_CACHE_COMPLETE.json")
    cby = {r["id"]: r for r in cache["records"] if r["id"] in allowed}
    predictions = []; diagnostics = []
    begun = time.monotonic()
    max_correct_parity = 0.
    with torch.no_grad():
        for fid in sorted(allowed):
            r = rows[fid]
            packet = torch.load(C.SOURCE / cby[fid]["path"], map_location="cpu", weights_only=False)
            inp = extractor.branch_inputs(packet["captured"])
            if inp is None:
                for arm in ("CORRECT", "TRAIN_MEAN", "NORMALIZED_ZERO", "TRAIN_SHUFFLE"):
                    predictions.append(dict(method=arm, id=fid, seed=1, session=r["session"], grade=r["grade"], q=np.array(r["qFinal"], float), diagnostic=None))
                continue
            logits_path = C.SOURCE / "data/pallet/results/pallet_dim_conditioned_p_v1/DEV_logits/N3_DIM_SYM_seed1" / (fid.replace(":", "__") + ".npz")
            actual = np.load(logits_path, allow_pickle=False)
            logits = torch.from_numpy(actual["logits"].copy())
            support = torch.from_numpy(actual["support"].copy())
            assert logits.shape == (8, 222) and support.shape == (8,)
            dimensions = np.asarray(packet["dimensions"])
            z_correct = R.context(dimensions[None], np.array([packet["order"]]), norm, False)[0]
            z_mean = R.context(mean_dims[None], np.array([packet["order"]]), norm, False)[0]
            donor = int(hashlib.sha256(("20261011|" + fid).encode()).hexdigest(), 16) % len(train_dims)
            z_shuffle = R.context(train_dims[donor:donor+1], np.array([packet["order"]]), norm, False)[0]
            old = residual(head, z_correct)
            probability = logits.softmax(-1)
            box = torch.as_tensor(inp["boxes"]).float()
            diagonal = torch.linalg.vector_norm(box[2:] - box[:2]).clamp_min(1)
            displacement = diagonal * head.displacements
            points = torch.as_tensor(inp["points"])[None].float()
            cap = .01 * math.hypot(*packet["raw_hw"]) * inp["gain"]
            for arm, z in (("CORRECT", z_correct), ("TRAIN_MEAN", z_mean), ("NORMALIZED_ZERO", np.zeros(5, np.float32)), ("TRAIN_SHUFFLE", z_shuffle)):
                modified = logits if arm == "CORRECT" else logits + (residual(head, z) - old)
                o = dict(logits=modified[None], points_raw=points,
                    candidate_displacements=displacement[None], point_support=support[None])
                qnet = R.decode(o, 1, 1, cap)[0].numpy()
                q = np.asarray(r["q0"], float) + (qnet.astype(float) - np.asarray(inp["points"], float))/inp["gain"]
                q[8] = np.asarray(r["q0"], float)[8]
                if arm == "CORRECT":
                    error = np.linalg.norm(q - np.asarray(r["qFinal"], float), axis=-1)
                    max_correct_parity = max(max_correct_parity, float(np.max(error[np.isfinite(error)])))
                    assert max_correct_parity <= 1e-4, "Stored full-logit decode parity failed"
                new_probability = modified.softmax(-1)
                centered_delta = (modified-modified.mean(-1, keepdim=True)) - (logits-logits.mean(-1, keepdim=True))
                unclipped = (new_probability[..., None]*displacement[None]).sum(-2)
                diag = dict(centered_logit_maxabs=float(centered_delta.abs().max()),
                    centered_logit_RMS=float(centered_delta.square().mean().sqrt()),
                    probability_L1_mean=float((new_probability-probability).abs().sum(-1)[support].mean()),
                    argmax_changed=int(((modified.argmax(-1)!=logits.argmax(-1)) & support).sum()),
                    supported_corners=int(support.sum()),
                    cap_active=int(((unclipped.norm(dim=-1)>cap) & support).sum()),
                    coordinate_delta_px=np.linalg.norm(q[:8]-np.asarray(r["qFinal"], float)[:8],axis=-1)[support.numpy()].tolist(),
                    donor_original_TRAIN_index=donor if arm=="TRAIN_SHUFFLE" else None)
                predictions.append(dict(method=arm, id=fid, seed=1, session=r["session"], grade=r["grade"], q=q, diagnostic=diag))
                diagnostics.append(dict(method=arm,id=fid,**diag))
    private = C.PRIVATE / "B1_PREDICTIONS_SEALED.json"
    C.write(private, predictions)
    C.write(C.DOC / "B1_PREDICTION_SEAL.json", dict(rows=len(predictions), source_checkpoint=cp_binding,
        full_logit_decode_max_error_px=max_correct_parity, sealed_before_reference_scoring=True, new_CNN_forwards=0,
        mean_dimensions_normalized_context=z_mean.tolist(), mean_dimensions_equals_zero=bool(np.allclose(z_mean,0,atol=1e-10,rtol=0))))
    # Read reference labels strictly after all counterfactual coordinates are sealed.
    reference = {r["id"]: r for r in C.rows(C.PUBLIC / "_docs/experiments/pallet_wd_hypothesis_diag_20261010/STAGE1_ROWS_REAL.jsonl.gz")
                 if r["id"] in allowed and r["method"]=="N3_DIM_SYM" and r["seed"]==1}
    truth = truth_real(C.SOURCE, reference)
    pose = pose_api(C.SOURCE)
    scored = []
    with C.pnp_counter("B1", 3200) as count:
        for row in predictions:
            fid = row["id"]; base = rows[fid]; meta = base["fixed_metadata"]
            if row["method"] == "CORRECT":
                result = dict(reference[fid]["pose"]["S0"])
                hypothesis = reference[fid]["hypS0"]
            else:
                actual = pose.infer(row["q"], np.asarray(meta["K"]), np.asarray(meta["dimensions_pnp_WH_D_m"]), False)
                result = pose.metric((fid,actual,truth[fid])); hypothesis=actual.get("selected_hypothesis")
            target = reference[fid]["hypOracle"]
            uncertain = fid in set(C.read(C.DOC / "INPUT_CONTRACT.json")["unclear_allowed_ids"])
            scored.append(dict(method="B1_"+row["method"],id=fid,seed=1,session=row["session"],grade=row["grade"],
                pose=result, selected_hypothesis=hypothesis, axis_error=None if uncertain or hypothesis is None else int(hypothesis!=target),
                success=int(result.get("available") and result["translation_cm"]<5 and result["rotation_deg"]<5),
                legacy_confusion=int(result.get("available") and result["rotation_deg"]>45 and abs(result["yaw_deg"])>=60),
                diagnostic=row["diagnostic"]))
    summaries = {}
    # Same 173 whole population, seed1 and every method; no outcome-based filtering.
    for arm in ("CORRECT","TRAIN_MEAN","NORMALIZED_ZERO","TRAIN_SHUFFLE"):
        rr = [r for r in scored if r["method"] == "B1_"+arm]
        dd = [r for r in diagnostics if r["method"] == arm]
        coordinates = [x for r in dd for x in r["coordinate_delta_px"]]
        summaries[arm] = dict(frames=len(rr), metrics=summarize(rr),
            coordinate_delta_px=dict(n=len(coordinates),mean=float(np.mean(coordinates)),median=float(np.median(coordinates)),P90=float(np.quantile(coordinates,.9))),
            centered_logit_maxabs=max(r["centered_logit_maxabs"] for r in dd), probability_L1_mean=float(np.mean([r["probability_L1_mean"] for r in dd])),
            argmax_changed=sum(r["argmax_changed"] for r in dd), supported_corners=sum(r["supported_corners"] for r in dd),
            cap_active=sum(r["cap_active"] for r in dd),
            WD_choice_switches=sum(r["selected_hypothesis"]!=reference[r["id"]]["hypS0"] for r in rr))
    with gzip.open(C.DOC / "B1_FRAME_ROWS.jsonl.gz", "wt", compresslevel=6) as f:
        for r in scored: f.write(json.dumps(C.clean(r),ensure_ascii=False,separators=(",",":"),allow_nan=False)+"\n")
    C.write(C.DOC / "B1_RESULTS.json", dict(status="SENSITIVE_BUT_BENEFIT_UNRESOLVED", summaries=summaries,
        frames=173,sessions=9,seed=1, mean_vs_zero_not_equivalent=not np.allclose(z_mean,0,atol=1e-10,rtol=0),
        actual_OpenCV_calls=sum(count.values()),new_neural_backbone_forwards=0,new_fits=0,
        correct_PnP_dimensions_every_condition=True, wall_seconds=time.monotonic()-begun,
        interpretation="Counterfactual dimension response is not an accuracy benefit proof; original fixed checkpoints/visual inputs unchanged"))
    print("B1_DONE",len(scored),sum(count.values()),round(time.monotonic()-begun,2),flush=True)


if __name__ == "__main__":
    torch.set_num_threads(2);cv2.setNumThreads(1)
    run()
