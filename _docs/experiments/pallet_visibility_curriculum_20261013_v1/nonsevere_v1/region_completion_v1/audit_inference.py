"""Read only sealed predictions and prediction-derived inputs, never labels.

This audit recomputes objective, geometric rank, pose motion, per-anchor
protection, region penalties, selected candidates and mixed readout without
calling the optimizer, proposal selector or region selector.
"""
from collections import Counter
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from scripts.research.pallet_visibility_curriculum_20261013_v1.curriculum import CurriculumConfig

HERE = Path(__file__).resolve().parent
COHORT = HERE.parent
ANCHOR = COHORT / "anchored_completion_v1"
CHECKS = Counter()
FAILURES = []
EXTREMA = Counter()
NOTES = []


def read(path):
    path = Path(path)
    if any(x in str(path).lower() for x in ("scoring", "scored_rows", "coordinate_rows", "radius_fit_rows", "results.json")):
        raise RuntimeError("Label/outcome files are outside this inference audit")
    return json.loads(path.read_text())


def rows(path):
    if Path(path).name != "PREDICTIONS.jsonl.gz":
        raise RuntimeError("Only prediction JSONL records are allowed")
    with gzip.open(path, "rt") as stream:
        for line in stream:
            yield json.loads(line)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def array_sha(value, kind):
    a = np.ascontiguousarray(np.asarray(value, dtype="<f8"))
    return hashlib.sha256(kind.encode()+str(a.shape).encode()+a.tobytes()).hexdigest()


def check(kind, condition, context):
    CHECKS[kind] += 1
    if not bool(condition):
        FAILURES.append(dict(check=kind, context=context))


def near(a, b, atol=1e-8):
    return bool(np.allclose(np.asarray(a, float), np.asarray(b, float), atol=atol, rtol=0., equal_nan=True))


def corners(extents):
    a, b, c = np.asarray(extents)/2
    return np.array([[-a,-b,-c], [a,-b,-c], [a,b,-c], [-a,b,-c],
                     [-a,-b,c], [a,-b,c], [a,b,c], [-a,b,c]])


def project(xyz, R, t, K):
    camera = xyz @ np.asarray(R).T+np.asarray(t)
    homogeneous = camera @ K.T
    return homogeneous[:, :2]/homogeneous[:, 2:], camera


def huber(error, radius):
    return np.where(error <= radius, error**2, 2*radius*error-radius**2)


def objective(xyz, K, q, R, t, R0, t0, pool, weights, cfg):
    uv, _ = project(xyz, R, t, K)
    residual = np.linalg.norm(uv[pool]-q[pool], axis=1)
    depth = max(np.linalg.norm(t0), np.linalg.norm(np.ptp(xyz, axis=0)), 1e-6)
    rv = cv2.Rodrigues(np.asarray(R) @ R0.T)[0].ravel()
    prior = cfg.prior_strength_px**2 * (
        np.sum((rv/np.deg2rad(cfg.rotation_prior_deg))**2)
        +np.sum(((np.asarray(t)-t0)/depth/cfg.translation_prior_fraction)**2))
    return float((np.sum(huber(residual, cfg.robust_px)*weights[pool])+prior)
                 /max(float(weights[pool].sum()), 1e-12))


def shape(points):
    s = np.linalg.svd(points-points.mean(0), compute_uv=False)
    return float(s[1]/max(s[0], 1e-12)), float(s[1])


def geometry(xyz, K, q, R, t, ids, dof, cfg):
    ids = np.asarray(ids, int)
    if len(ids) < 3:
        return False, 0, float("inf")
    ratio, spread = shape(q[ids])
    camera = xyz[ids] @ R.T+t
    h = camera @ K.T
    D = (K[None,:2,:]*h[:,2,None,None]-h[:,:2,None]*K[None,2:3,:])/h[:,2,None,None]**2
    if dof == 3:
        J = D.reshape(-1,3)
    elif len(ids) == 3:
        X = xyz[ids] @ R.T
        skew = np.zeros((len(ids),3,3))
        skew[:,0,1],skew[:,0,2] = -X[:,2],X[:,1]
        skew[:,1,0],skew[:,1,2] = X[:,2],-X[:,0]
        skew[:,2,0],skew[:,2,1] = -X[:,1],X[:,0]
        rotation = np.einsum("nij,njk->nik", D, -skew)
        J = np.concatenate([rotation,D],axis=2).reshape(-1,6)
    else:
        # The frozen >=4 gate uses OpenCV's absolute Rodrigues parameters.
        # Recompute that documented parameterization, separately from fitting.
        _, J = cv2.projectPoints(np.ascontiguousarray(xyz[ids]), cv2.Rodrigues(R)[0], t, K, None)
        J = J[:,:6]
    J = J/np.maximum(np.linalg.norm(J,axis=0),1e-12)[None]
    singular = np.linalg.svd(J, compute_uv=False)
    rank = int(np.sum(singular > singular[0]*1e-9))
    condition = float(singular[0]/max(singular[-1],1e-15))
    objspread = shape(xyz[ids])[0]
    usable = (rank == dof and condition <= cfg.maximum_jacobian_condition
        and ratio >= cfg.minimum_spread_ratio and spread >= cfg.minimum_spread_px
        and (dof == 3 or objspread > 1e-8))
    return bool(usable), rank, condition


def anchor_guard(xyz,K,q,R,t,base,cfg,context, reference_initial=False, track_selected=False):
    ids = base["anchor_ids"]
    ref = base["anchor_reference"]
    if ref is None:
        if not reference_initial:
            return True
        ref = base["boundary_seed"]["stages"][0]
    initial_uv,_=project(xyz,ref["R"],ref["t"],K)
    uv,_=project(xyz,R,t,K)
    shift=np.linalg.norm(uv[ids]-initial_uv[ids],axis=1)
    errors=np.linalg.norm(uv[ids]-q[ids],axis=1)
    old_errors=np.linalg.norm(initial_uv[ids]-q[ids],axis=1)
    increase=huber(errors,cfg.robust_px)-huber(old_errors,cfg.robust_px)
    if len(ids) and track_selected:
        EXTREMA["maximum_new_selected_anchor_shift_px"]=max(EXTREMA["maximum_new_selected_anchor_shift_px"],float(shift.max()))
        EXTREMA["maximum_new_selected_anchor_huber_increase_px2"]=max(EXTREMA["maximum_new_selected_anchor_huber_increase_px2"],float(increase.max()))
    return bool(np.all(shift <= 2.+1e-8) and np.all(increase <= 1.+1e-8))


def motion(xyz,K,R,t,R0,t0,cfg):
    angle=float(np.degrees(np.arccos(np.clip((np.trace(R @ R0.T)-1)/2,-1,1))))
    translation=float(np.linalg.norm(t-t0)/max(np.linalg.norm(t0),1e-6))
    uv,camera=project(xyz,R,t,K)
    uv0,_=project(xyz,R0,t0,K)
    shift=float(np.linalg.norm(uv-uv0,axis=1).max())
    return bool((camera[:,2] > 1e-6).all() and angle <= cfg.maximum_rotation_deg
                and translation <= cfg.maximum_translation_fraction and shift <= cfg.maximum_projection_shift_px)


def region_penalty(candidate, regions,xyz,K,context):
    uv,_=project(xyz,candidate["R"],candidate["t"],K)
    losses=[]
    for r in regions:
        norm=float(np.linalg.norm((uv[r["native_corner"]]-r["center_xy"])/r["half_axes_px"]))
        losses.append(max(norm-1.,0.)**2)
    value=float(np.mean(losses)) if losses else 0.
    check("region_normalized_penalty_recomputed",near(value,candidate["region"]["normalized_penalty"]),context)
    check("region_adds_zero_precise_support",candidate["region"]["independent_precise_support_count"] == 0,context)
    return value


def audit():
    protocol,model=read(HERE/"PROTOCOL.json"),read(HERE/"REGION_MODEL.json")
    source_hashes={p:sha(p) for p in protocol["source_sha256"]}
    check("all_protocol_bound_sources_unchanged",source_hashes == protocol["source_sha256"],"protocol")
    check("empirical_model_no_joint_coverage",model["calibrated_joint_coverage"] is False and model["calibrated_probability"] is False,"model")
    inputs=read(COHORT/"PREDICTION_INPUTS.json")
    check("prediction_inputs_have_no_labels",inputs["GT_input"] is False and inputs["human_visibility_input"] is False,"inputs")
    frames={f["id"]:f for f in inputs["frames"]}
    selected=read(ANCHOR/"SELECTION_CAL.json")["selected"]
    image_cache={}
    summaries={}
    seals={}
    immutable_fields=["raw_points","measured_points","source_measured_points","immutable_pool_ids",
        "frozen_pool_weights","anchor_ids","anchor_measurements","anchor_weights","self_hidden",
        "configuration","states","boundary_seed","eligible_supplemental_ids","supplemental_order"]
    for group in protocol["groups"]:
        seal=read(HERE/group/"PREDICTION_SEAL.json")
        check("prediction_seal_hash",seal["predictions_sha256"] == sha(HERE/group/"PREDICTIONS.jsonl.gz"),group)
        check("prediction_seal_before_scoring",seal["before_scoring"] is True,group)
        check("prediction_seal_protocol_model",seal["protocol_sha256"] == sha(HERE/"PROTOCOL.json") and seal["region_model_sha256"] == sha(HERE/"REGION_MODEL.json"),group)
        seals[group]=dict(seal_sha256=sha(HERE/group/"PREDICTION_SEAL.json"),**seal)
        old={r["id"]:r for r in rows(ANCHOR/group/"PREDICTIONS.jsonl.gz")
             if r["method"] == (selected if group == "CAL_N3" else "SELECTED")}
        grouped={}
        for r in rows(HERE/group/"PREDICTIONS.jsonl.gz"):
            grouped.setdefault(r["id"],{})[r["method"]]=r
        check("group_exact_frame_count",len(grouped) == protocol["frames"][group],group)
        check("group_exact_row_count",sum(len(v) for v in grouped.values()) == seal["rows"],group)
        count=Counter()
        for key in ("PROPOSALS_new_selections", "FINITE_VISIBLE_new_selections",
                    "FINITE_REGION_new_selections", "FINITE_pair_different_selected_poses",
                    "INITIAL_ninth_center_source_label_missing"):
            count[key]=0
        for fid,arms in grouped.items():
            context=f"{group}:{fid}"
            check("five_declared_methods",set(arms) == set(protocol["methods"]),context)
            base=arms["BASE"]["solution"]
            check("V1_original_solution_exact",base == old[fid]["solution"],context)
            frame=frames[fid]
            if fid not in image_cache:
                imagepath=ROOT/frame["image"]
                check("actual_image_file_sha",sha(imagepath) == frame["image_sha256"],context)
                im=cv2.imread(str(imagepath))
                check("actual_image_canvas",list(im.shape[:2]) == frame["raw_hw"],context)
                pixel=hashlib.sha256(b"BGR_PIXEL_SHA256_v1"+str(im.shape).encode()+np.ascontiguousarray(im).tobytes()).hexdigest()
                image_cache[fid]=(pixel,list(im.shape[:2]))
            pixel,canvas=image_cache[fid]
            initial=arms["BASE"]["initial_pose"]
            xyz,K=corners(initial["cf_extents"]),np.asarray(frame["K"],float)
            R0,t0=np.asarray(initial["R_cf"]),np.asarray(initial["centroid"])
            q=np.asarray(base["measured_points"],float)[:8]
            raw=np.asarray(base["raw_points"],float)
            pool=base["immutable_pool_ids"]
            w=np.zeros(8);w[pool]=base["frozen_pool_weights"]
            cfg=CurriculumConfig(**base["configuration"]["pose"])
            initial_cost=objective(xyz,K,q,R0,t0,R0,t0,pool,w,cfg)
            current_cost=objective(xyz,K,q,base["R"],base["t"],R0,t0,pool,w,cfg)
            check("original_initial_prior_bound",near(R0,base["boundary_seed"]["stages"][0]["R"],0.) and near(t0,base["boundary_seed"]["stages"][0]["t"],0.),context)
            finite0=arms["FINITE_VISIBLE"]["solution"]["region_audit"]
            finite1=arms["FINITE_REGION"]["solution"]["region_audit"]
            if (arms["FINITE_VISIBLE"]["solution"]["R"] != arms["FINITE_REGION"]["solution"]["R"]
                    or arms["FINITE_VISIBLE"]["solution"]["t"] != arms["FINITE_REGION"]["solution"]["t"]):
                count["FINITE_pair_different_selected_poses"]+=1
            check("finite_candidate_count_identical",len(finite0["candidates"]) == len(finite1["candidates"]),context)
            for x,y in zip(finite0["candidates"],finite1["candidates"]):
                identity=(x["name"] == y["name"] and x.get("R") == y.get("R") and x.get("t") == y.get("t")
                    and x.get("precise_used_ids") == y.get("precise_used_ids") and x.get("estimated_dof") == y.get("estimated_dof"))
                check("finite_same_pose_pool_support",identity,context+":"+x["name"])
                for key in ("usable","geometry","positive_depth","motion_pass","anchor_guard","visible_objective_pass","objective","region","support_mode"):
                    check("finite_identical_gates_and_evidence",x.get(key) == y.get(key),context+":"+x["name"]+":"+key)
            check("finite_identical_coarse_regions",arms["FINITE_VISIBLE"]["coarse_regions"] == arms["FINITE_REGION"]["coarse_regions"],context)
            count["generated_p3p_roots"] += finite1["generation"].get("roots",0)
            for method,row in arms.items():
                rctx=context+":"+method
                check("row_no_GT_inputs",row["GT_inputs"] is False,rctx)
                check("native_raw_queries_unchanged",row["original_points"] == base["raw_points"],rctx)
                check("registered_dimensions_preserved",row["actual_pose"]["cf_extents"] == initial["cf_extents"],rctx)
                check("original_initial_pose_same",row["initial_pose"] == initial,rctx)
                pose=row["actual_pose"]
                R,t=np.asarray(pose["R_cf"]),np.asarray(pose["centroid"])
                uv,camera=project(xyz,R,t,K)
                check("proper_rotation_positive_depth",near(R.T@R,np.eye(3),1e-6) and abs(np.linalg.det(R)-1)<1e-6 and np.all(camera[:,2]>1e-6),rctx)
                check("physical_basis_preserved",near(R.T@np.asarray(pose["R_physical"]),R0.T@np.asarray(initial["R_physical"]),1e-10),rctx)
                check("rigid_model_readout_from_one_pose",near(row["keypoint_readout"]["rigid_model_corners"],uv,1e-10),rctx)
                if method == "INITIAL":
                    check("initial_pose_exact",pose == initial,rctx)
                    check("initial_center_carrier_preserved",near(np.asarray(row["keypoint_readout"]["reported_points"],float)[8:],raw[8:],0.),rctx)
                    if len(row["keypoint_readout"]["reported_points"]) != len(row["keypoint_readout"]["point_sources"]):
                        count["INITIAL_ninth_center_source_label_missing"]+=1
                    continue
                s=row["solution"]
                for key in immutable_fields:
                    check("immutable_V1_field",s[key] == base[key],rctx+":"+key)
                for k,(record,original_record) in enumerate(zip(s["keypoint_status"],base["keypoint_status"])):
                    check("immutable_native_point_status_evidence",{a:b for a,b in record.items() if a!="used_for_final_pose"}
                          == {a:b for a,b in original_record.items() if a!="used_for_final_pose"},rctx+":"+str(k))
                check("immutable_observed_measurement_map",{
                    k:{a:b for a,b in record.items() if a!="used_for_final_pose"}
                    for k,record in s["observed_measurements_by_id"].items()} == {
                    k:{a:b for a,b in record.items() if a!="used_for_final_pose"}
                    for k,record in base["observed_measurements_by_id"].items()},rctx)
                if s["anchor_reference"] != base["anchor_reference"]:
                    check("only_explicit_initial_anchor_reference_allowed",method.startswith("FINITE_") and base["anchor_reference"] is None
                        and s["anchor_reference"]["source"] == "declared_initial_reference_without_accepted_seed"
                        and near(s["anchor_reference"]["R"],R0,0.) and near(s["anchor_reference"]["t"],t0,0.),rctx)
                else:
                    check("frozen_anchor_reference_unchanged",True,rctx)
                check("selected_solution_actual_pose_match",near(s["R"],R,0.) and near(s["t"],t,0.),rctx)
                check("hidden_projection_only_invariants",s["provenance"]["projected_points_used_as_measurements"] is False
                      and s["provenance"]["excluded_points_used_for_fit"] is False
                      and s["provenance"]["completed_points_used_for_fit"] is False,rctx)
                used=s["used_ids"]
                check("used_support_is_original_pool",set(used) <= set(pool) and not any(base["self_hidden"][k] for k in used),rctx)
                for k in used:
                    record=s["keypoint_status"][k]
                    check("excluded_status_never_fitted",not record.get("conditional_truncation") and not record.get("independent_other_object_region_supported")
                        and record["state"] not in ("SELF_OCCLUDED","TRUNCATED","UNKNOWN","UNKNOWN_OR_OCCLUSION","OTHER_OBJECT_SUSPECTED","OTHER_OBJECT_OCCLUDED"),rctx+":"+str(k))
                mixed=deepcopy(uv.tolist()+base["raw_points"][8:])
                sources=["RIGID_MODEL_COMPLETION"]*8+(["ORIGINAL_CENTER_METADATA"] if len(raw)==9 else [])
                preserved=[]
                for k,record in enumerate(base["keypoint_status"]):
                    role=record["measurement_role"]
                    if k in base["anchor_ids"] and role == "IMAGE_BOUNDARY_ANCHOR":
                        source="IMAGE_BOUNDARY_ANCHOR_MEASUREMENT"
                    elif k in used and role == "RGB_FACE_JUNCTION_INTERIOR":
                        source="USED_RGB_JUNCTION_MEASUREMENT"
                    elif k in used and role in ("DETECTOR_VISIBLE_MODEL_INTERIOR","DETECTOR_VISIBLE_WEAK_BOUNDARY"):
                        source="USED_DETECTOR_VISIBLE_MEASUREMENT"
                    else:
                        continue
                    mixed[k]=s["measured_points"][k];sources[k]=source;preserved.append(k)
                readout=row["keypoint_readout"]
                check("mixed_measurements_preserved_exact",near(readout["reported_points"],mixed,1e-10) and readout["point_sources"] == sources
                    and readout["preserved_measurement_ids"] == preserved,rctx)
                check("mixed_not_one_rigid_pose_or_PnP_input",readout["mixed_points_are_one_rigid_pose"] is False and readout["provenance"]["mixed_readout_must_not_be_fed_to_PnP"] is True,rctx)
                count[f"{method}_accepted"]+=int(s["accepted"])
                count[f"{method}_dof_{s['estimated_dof']}"]+=1
                count[f"{method}_preserved_measurements"]+=len(preserved)
                if method == "PROPOSALS":
                    check("proposal_retains_V1_support_and_dof",used == base["used_ids"] and s["estimated_dof"] == base["estimated_dof"],rctx)
                    if s["estimated_dof"] == 3:
                        check("proposal_three_dof_R_exact",near(R,R0,0.),rctx)
                    if s["proposal_selected"]:
                        cost=objective(xyz,K,q,R,t,R0,t0,pool,w,cfg)
                        check("proposal_dual_strict_gain",current_cost-cost>1e-7 and initial_cost-cost>1e-7,rctx)
                        check("new_proposal_guard_recomputed",motion(xyz,K,R,t,R0,t0,cfg) and geometry(xyz,K,q,R,t,used,s["estimated_dof"],cfg)[0]
                            and anchor_guard(xyz,K,q,R,t,base,cfg,rctx,track_selected=True),rctx)
                        count["PROPOSALS_new_selections"]+=1
                    else:
                        check("proposal_all_rejected_exact_V1",s["R"]==base["R"] and s["t"]==base["t"] and s["final_guard"]==base["final_guard"],rctx)
                if not method.startswith("FINITE_"):
                    continue
                regions=row["coarse_regions"]
                regionids=[r["native_corner"] for r in regions]
                eligible_regions=[k for k in range(8) if k not in pool and np.isfinite(raw[k]).all() and not np.all(raw[k]==-1)]
                check("coarse_regions_exclude_precise_pool",regionids == eligible_regions,rctx)
                validq=np.isfinite(raw[:8]).all(1) & ~np.all(raw[:8]==-1,axis=1)
                scale=max(float(np.linalg.norm(np.ptp(raw[:8][validq],axis=0))) if validq.any() else 0.,16.)
                for region in regions:
                    k=region["native_corner"];category="INITIAL_MODEL_SELF" if base["self_hidden"][k] else "OTHER_UNOBSERVED"
                    radius=scale*model["categories"][category]["normalized_radius"]
                    check("actual_region_input_bindings",region["source"] == "IMAGE_REGRESSED" and region["raw_query_xy"] == base["raw_points"][k]
                        and region["center_xy"] == base["raw_points"][k] and region["query_sha256"] == array_sha(raw,"native_query_v1")
                        and region["image_binding"] == dict(sha256=pixel,kind="BGR_PIXEL_SHA256",verified_pixels=True,image_hw=canvas)
                        and region["geometry_binding"] == dict(K_sha256=array_sha(K,"K_v1"),xyz_sha256=array_sha(xyz,"native_xyz_v1")),rctx+":"+str(k))
                    check("region_digest_and_frozen_radius",region["region_sha256"] == digest({a:b for a,b in region.items() if a!="region_sha256"})
                        and near(region["half_axes_px"],[radius,radius],1e-10) and region["hard_policy"] is None
                        and region["calibration_model_sha256"] == sha(HERE/"REGION_MODEL.json") and region["default_coverage_guaranteed"] is False,rctx+":"+str(k))
                audit=s["region_audit"];weight=audit["region_weight"]
                check("finite_fixed_weight",weight == (0. if method == "FINITE_VISIBLE" else .10),rctx)
                check("p3p_roots_finite_only",audit["generation"].get("roots",0) <= 4 and audit["regions_are_precise_correspondences"] is False,rctx)
                best_index=0;best_score=current_cost
                for index,candidate in enumerate(audit["candidates"]):
                    cctx=rctx+":"+candidate["name"]
                    if "R" not in candidate:
                        check("unavailable_candidates_not_selected",candidate.get("usable") is False,cctx)
                        continue
                    cR,ct=np.asarray(candidate["R"]),np.asarray(candidate["t"])
                    cost=objective(xyz,K,q,cR,ct,R0,t0,pool,w,cfg)
                    penalty=region_penalty(candidate,regions,xyz,K,cctx)
                    score=cost+weight*cfg.robust_px**2*penalty
                    check("candidate_objective_score_recomputed",near(cost,candidate["objective"],1e-8) and near(score,candidate["combined_score"],1e-8),cctx)
                    if index == 0:
                        best_score=score
                        continue
                    ids,dof=candidate["precise_used_ids"],candidate["estimated_dof"]
                    geom,rank,condition=geometry(xyz,K,q,cR,ct,ids,dof,cfg)
                    visible=cost <= current_cost+cfg.validation_tolerance_px2 and cost <= initial_cost+cfg.validation_tolerance_px2
                    guard=anchor_guard(xyz,K,q,cR,ct,base,cfg,cctx,True)
                    expected=bool(geom and motion(xyz,K,cR,ct,R0,t0,cfg) and guard and visible and candidate["region"]["hard_valid"])
                    check("finite_candidate_admissibility_recomputed",expected == candidate["usable"] and visible == candidate["visible_objective_pass"],cctx)
                    check("finite_candidate_actual_rank_condition",rank == candidate["geometry"]["jacobian_rank"] and near(condition,candidate["geometry"]["jacobian_condition"],1e-7),cctx)
                    if candidate["name"].startswith("P3P_ROOT_"):
                        check("P3P_precise_support_exact_triple",ids == audit["generation"]["native_triple"] and len(ids)==3 and dof==6 and set(ids)<=set(base["anchor_ids"]),cctx)
                        count["P3P_candidates_admissible"]+=int(expected)
                    if expected and score < best_score-cfg.minimum_gain_px2:
                        best_index,best_score=index,score
                chosen=audit["candidates"][best_index]
                check("finite_selected_by_frozen_score_only",audit["selected"] == chosen["name"] and near(R,chosen["R"],0.) and near(t,chosen["t"],0.),rctx)
                if best_index:
                    count[f"{method}_new_selections"]+=1
                    count[f"{method}_selected_{chosen['name']}"]+=1
                    check("new_region_selected_anchor_protection",anchor_guard(xyz,K,q,R,t,base,cfg,rctx,True,True),rctx)
                    strict=initial_cost-objective(xyz,K,q,R,t,R0,t0,pool,w,cfg)>cfg.minimum_gain_px2
                    check("strict_gain_flag_not_tolerance_claim",s["final_guard"]["selected_checkpoint_passes_full_pool_guard"] == strict
                          and s["final_guard"]["selected_checkpoint_passes_visible_tolerance_guard"] is True,rctx)
                    count[f"{method}_new_without_strict_initial_gain"]+=int(not strict)
                    if chosen["name"].startswith("P3P_ROOT_"):
                        check("selected_three_points_full_dof_explicit",len(used)==3 and s["estimated_dof"] == 6
                             and audit["original_three_point_V1_policy_changed_only_in_this_opt_in_solver"] is True,rctx)
                else:
                    check("finite_BASE_preserves_V1_pose",s["R"]==base["R"] and s["t"]==base["t"],rctx)
        summaries[group]=dict(frames=len(grouped),rows=sum(len(v) for v in grouped.values()),counts=dict(count))
    missing_center_sources=sum(g["counts"]["INITIAL_ninth_center_source_label_missing"] for g in summaries.values())
    if missing_center_sources:
        NOTES.append(dict(kind="NON_NUMERICAL_INITIAL_OUTPUT_METADATA",affected_rows=missing_center_sources,
            detail="INITIAL reports nine coordinates including the unchanged center but only eight corner source labels. Pose, eight-corner projection and ninth coordinate are correct; this does not affect selection or any fitted measurement. Sealed experiment source/output are preserved."))
    check("all_protocol_sources_still_unchanged_after_audit",all(sha(p)==h for p,h in protocol["source_sha256"].items()),"protocol_end")
    result=dict(schema="independent_region_inference_audit_v1",status="PASS" if not FAILURES else "FAIL",
        GT_scoring_labels_or_results_read=False,predictions_not_changed=True,runtime_sources_not_changed=True,
        protocol_sha256=sha(HERE/"PROTOCOL.json"),region_model_sha256=sha(HERE/"REGION_MODEL.json"),
        runner_sha256=sha(__file__),seals=seals,groups=summaries,total_prediction_rows=sum(g["rows"] for g in summaries.values()),
        checks=dict(CHECKS),check_count=sum(CHECKS.values()),failures=FAILURES,extrema=dict(EXTREMA),metadata_notes=NOTES,
        first_pass_audit_preserved="INFERENCE_AUDIT_FIRST_PASS.json",
        first_pass_bookkeeping_correction="First pass extrema included rejected candidates; final extrema correctly cover newly selected poses only. Added explicit unchanged-status and center-metadata checks. No inference source, prediction, threshold or policy changed.",
        limitations=["These are software/provenance checks, not pose-accuracy or semantic-visibility claims",
            "The empirical CAL model is already-seen in-sample tolerance, not calibrated coverage",
            "P3P has finite ambiguous roots; full local rank does not prove global uniqueness",
            "Mixed image/model readout is not one exact rigid projection and is not additional PnP evidence"])
    output=HERE/"INFERENCE_AUDIT.json"
    if output.exists():
        raise FileExistsError("Preserve existing audit")
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps({k:result[k] for k in ("status","total_prediction_rows","check_count","groups","failures","extrema")},indent=2))
    if FAILURES:
        raise SystemExit(1)


if __name__ == "__main__":
    cv2.setNumThreads(1)
    audit()
