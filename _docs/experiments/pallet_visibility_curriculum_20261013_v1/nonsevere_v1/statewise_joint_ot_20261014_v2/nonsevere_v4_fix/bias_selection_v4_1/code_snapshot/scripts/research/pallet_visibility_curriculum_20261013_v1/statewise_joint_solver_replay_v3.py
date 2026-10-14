"""Blind solver revision using exactly the sealed v2 RGB measurements.

The revision freezes q/covariance, not an earlier rigid seed projection.
It checks final B/I geometry/canvas and allows a explicitly ambiguous joint
pose family for two B/I plus O/T regions. No classifier or region is refit.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from copy import deepcopy
from multiprocessing import get_context
from pathlib import Path
import shutil
import time

import numpy as np

from . import common as C
from .joint_ot_pnp import project_native,solve_joint_ot_pnp
from .statewise_joint_experiment_v2 import STUDY,source_bindings,crop_bindings


def _pose_and_readout(runtime,solution):
    original=runtime["original_coarse_pose"]
    seed=solution.get("corrected_BI_seed") or {}
    accepted=bool(solution["accepted"])
    if accepted:
        R,t=np.asarray(solution["R"]),np.asarray(solution["t"])
        source="all_joint_OT_feasible_pose_family_representative" if (solution.get("ambiguity") or {}).get("detected") else "all_joint_OT_feasible_corrected_BI_SE3"
    elif seed.get("visible_support_guards_pass") and not seed.get("contains_underconstrained_pose_family"):
        R,t=np.asarray(seed["R"]),np.asarray(seed["t"])
        source="explicit_valid_corrected_BI_only_seed_fallback_S_output_only"
    else:
        R,t=np.asarray(original["R_cf"]),np.asarray(original["centroid"])
        source="explicit_registered_coarse_pose_fallback_S_output_only"
    Q=np.asarray(original["R_cf"]).T@np.asarray(original["R_physical"])
    pose=dict(available=True,R_cf=R.tolist(),R_physical=(R@Q).tolist(),centroid=t.tolist(),
        cf_extents=original["cf_extents"],physical_registration_Q=Q.tolist(),
        source=source,joint_feasible=accepted,fallback=not accepted,fallback_pose_does_not_solve_OT=not accepted)
    geometry=runtime["initial_geometry"]
    projection,_=project_native(np.asarray(geometry["native_xyz"]),R,t,np.asarray(geometry["K"]))
    reported,sources=[None]*8,["UNRESOLVED"]*8
    for m in runtime["frozen_visible_measurements"]:
        i=m["native_corner"];reported[i]=deepcopy(m["xy"]);sources[i]="FROZEN_CORRECTED_CURRENT_RGB_BI_MEASUREMENT"
    for i,s in enumerate(runtime["states"]):
        if s=="S":
            reported[i]=projection[i].tolist();sources[i]="FINAL_JOINT_RIGID_S_COMPLETION" if accepted else "EXPLICIT_FALLBACK_RIGID_S_PROJECTION"
        elif s in ("O","T") and accepted:
            reported[i]=projection[i].tolist();sources[i]="AMBIGUOUS_JOINT_POSE_FAMILY_PROJECTION" if (solution.get("ambiguity") or {}).get("detected") else "ALL_OT_JOINT_FEASIBLE_RIGID_COMPLETION"
    original_points=runtime["original_raw_points"]
    if len(original_points)==9:reported.append(deepcopy(original_points[8]));sources.append("ORIGINAL_NINTH_CENTER_METADATA")
    return pose,dict(reported_points=reported,point_sources=sources,rigid_model_corners=projection.tolist(),
        native_order_preserved=True,ninth_center_metadata_unchanged=True,BI_observations_preserved=True,
        mixed_points_are_one_rigid_pose=False,OT_solved=solution.get("OT_solved",False),
        OT_constraints_satisfied=solution.get("OT_constraints_satisfied",False),OT_unique_solution_claimed=False,
        OT_feasible_hypothesis_available=solution.get("OT_feasible_hypothesis_available",False),
        pose_family_candidates=deepcopy(solution.get("pose_family_candidates",[])),
        U_unresolved_ids=[i for i,s in enumerate(runtime["states"]) if s=="U"])


def _run(task):
    row,config=task;runtime=row["runtime"]
    initial=row["initial_pose"];X=C.cuboid(*initial["cf_extents"])
    measurements=deepcopy(runtime["frozen_visible_measurements"])
    before=deepcopy(measurements)
    starts=[dict(R=initial["R_cf"],t=initial["centroid"])]
    solution=solve_joint_ot_pnp(X,row["K"],runtime["states"],measurements,
        runtime["ot_domains"]["domains"],row["raw_hw"],initializations=starts,config=config)
    if measurements!=before:raise AssertionError("Frozen observations changed")
    pose,readout=_pose_and_readout(runtime,solution)
    runtime.update(accepted=bool(solution["accepted"]),joint_pose=solution,solution=solution,
        actual_pose=pose,keypoint_readout=readout,method="STATEWISE_JOINT_OT_V3")
    runtime["fallback_audit"]=dict(applied=not solution["accepted"],pose_source=pose["source"],
        reason=solution.get("fallback_reason"),O_T_reported_as_solved=solution.get("OT_solved",False),
        S_projection_is_image_measurement=False,corrected_BI_seed_used_as_OT_solution=False)
    runtime["ambiguity_audit"]=dict(material_ambiguity=bool((solution.get("ambiguity") or {}).get("detected")),
        accepted_pose_is_one_joint_feasible_branch=solution["accepted"],globally_unique_pose_proved=False,
        solver_ambiguity=solution.get("ambiguity"))
    runtime["provenance"].update(fresh_RGB_forwards_in_this_replay=0,
        RGB_state_BI_domain_source="exact_immutable_sealed_evaluation_v2",seed_projection_is_fixed=False,
        final_BI_geometric_visibility_and_canvas_checked=True)
    bi_states=["U" if s in ("O","T") else s for s in runtime["states"]]
    bi_solution=solve_joint_ot_pnp(X,row["K"],bi_states,deepcopy(before),[],row["raw_hw"],
                                 initializations=starts,config=config)
    bi_pose,_=_pose_and_readout(runtime,bi_solution)
    bi_pose["source"]="CORRECTED_BI_ONLY_PNP" if bi_solution["accepted"] else (
        "VALID_CORRECTED_BI_SEED_FALLBACK" if (bi_solution.get("corrected_BI_seed") or {}).get("visible_support_guards_pass") and not (bi_solution.get("corrected_BI_seed") or {}).get("contains_underconstrained_pose_family") else "ORIGINAL_PNP_FALLBACK")
    bi_pose["joint_feasible"]=False
    bi_pose["fallback_pose_does_not_solve_OT"]=True
    row["BI_only"]=dict(actual_pose=bi_pose,solution=bi_solution,states=bi_states,
                        frozen_visible_measurements=deepcopy(before))
    return row


def predict(output,input_run,workers=3):
    output=Path(output).resolve();input_run=Path(input_run).resolve()
    if output.exists():raise FileExistsError("Preserve previous replay")
    if not isinstance(workers,int) or not 1<=workers<=3:raise ValueError("One to three workers required")
    old_seal=C.read(input_run/"PREDICTION_SEAL.json");old_protocol=C.read(input_run/"PROTOCOL.json")
    if C.sha(input_run/"PREDICTIONS.jsonl.gz")!=old_seal["predictions_sha256"]:
        raise ValueError("Cached RGB prediction seal changed")
    if C.sha(input_run/"PROTOCOL.json")!=old_seal["protocol_sha256"]:raise ValueError("Cached input protocol changed")
    rows=[]
    for row in C.rows(input_run/"PREDICTIONS.jsonl.gz"):
        # Keep all exact RGB/state/measurement/domain fields, omit old solver
        # journals before sending independent workers their small records.
        runtime=row["runtime"]
        for key in ("joint_pose","solution","actual_pose","keypoint_readout","fallback_audit","ambiguity_audit"):
            runtime.pop(key,None)
        row.pop("BI_only",None);rows.append(row)
    if [r["id"] for r in rows]!=old_protocol["ids"] or len(rows)!=98:raise ValueError("Complete ordered98 cache required")
    config=dict(max_p3p_triples=56,global_rotation_starts=24,max_seed_nfev=160,max_joint_iterations=250,
        maximum_union_branches=256,maximum_visible_mahalanobis=3.,analytic_jacobian=True,
        protect_seed_projection=False,enforce_visible_geometry=True,geometry_tolerance_deg=2.,
        enforce_visible_canvas=True)
    output.mkdir(parents=True)
    protocol=deepcopy(old_protocol)
    protocol.update(schema="STATEWISE_JOINT_BLIND_SOLVER_REPLAY_V3",solver_configuration=config,
        source_sha256=source_bindings(),crop_source_sha256=crop_bindings(),independent_frame_workers=workers,
        input_prediction_seal=dict(path=str((input_run/"PREDICTION_SEAL.json").relative_to(C.ROOT)),
            sha256=C.sha(input_run/"PREDICTION_SEAL.json"),predictions_sha256=old_seal["predictions_sha256"]),
        unchanged_state_model_BI_and_domains=True,new_classifier_fit=False,new_RGB_forwards=False,
        revision_reason=["Freeze observed B/I q and covariance; rigid seed is not an observation",
            "Final directly visible B/I must be nonself and inframe under the registered model",
            "Two B/I plus all O/T may constrain a pose family; do not reject before joint feasibility"],
        protocol_scope="reuse already seen exploratory TEST; no independent postselection claim")
    C.write(output/"PROTOCOL.json",protocol)
    for relative,digest in protocol["source_sha256"].items():
        target=output/"code_snapshot"/relative;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(C.ROOT/relative,target)
        if C.sha(target)!=digest:raise ValueError("Frozen source snapshot mismatch")
    start=time.monotonic();by_id={}
    with ProcessPoolExecutor(max_workers=workers,mp_context=get_context("spawn")) as pool:
        futures={pool.submit(_run,(r,config)):r["id"] for r in rows}
        for future in as_completed(futures):
            row=future.result()
            if row["id"]!=futures[future] or row["id"] in by_id:raise ValueError("Worker input ID mismatch")
            by_id[row["id"]]=row
            C.write(output/"LIVE_PROGRESS.json",dict(completed=len(by_id),total=len(rows),last_id=row["id"],
                states=row["runtime"]["states"],accepted=row["runtime"]["accepted"],elapsed_seconds=time.monotonic()-start))
            print(f"{len(by_id)}/{len(rows)} {row['id']} accepted={row['runtime']['accepted']}",flush=True)
    ordered=[by_id[r["id"]] for r in rows]
    if source_bindings()!=protocol["source_sha256"] or crop_bindings()!=protocol["crop_source_sha256"]:
        raise ValueError("Source changed during replay")
    C.save_rows(output/"PREDICTIONS.jsonl.gz",ordered)
    C.write(output/"PREDICTION_SEAL.json",dict(schema="STATEWISE_JOINT_SEAL_V3",before_scoring=True,
        predictions_sha256=C.sha(output/"PREDICTIONS.jsonl.gz"),protocol_sha256=C.sha(output/"PROTOCOL.json"),
        source_sha256=protocol["source_sha256"],rows=len(ordered),severe_rows=0,elapsed_seconds=time.monotonic()-start))


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--output",required=True,type=Path)
    parser.add_argument("--input-run",required=True,type=Path);parser.add_argument("--workers",default=3,type=int)
    args=parser.parse_args();predict(args.output,args.input_run,args.workers)


if __name__=="__main__":main()
