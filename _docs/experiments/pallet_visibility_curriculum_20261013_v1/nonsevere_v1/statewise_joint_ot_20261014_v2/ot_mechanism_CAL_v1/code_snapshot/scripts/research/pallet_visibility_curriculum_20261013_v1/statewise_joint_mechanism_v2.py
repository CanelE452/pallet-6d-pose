"""Nondeployable CAL visibility-oracle mechanism diagnostic on real RGB crops.

Human O/V/S metadata supplies only visibility. S and conditional T still come
first from fixed initial geometry. Current RGB supplies B/I corrections. Every
arm shares those frozen observations and registered/global initializations.
No annotation coordinates are read before the complete prediction seal.
"""
from __future__ import annotations

import argparse
import builtins
from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
import io
from pathlib import Path
import shutil
import time

import cv2
import numpy as np

from . import common as C
from .joint_ot_pnp import project_native, solve_joint_ot_pnp
from .state_visibility_v2 import geometry_state_gates
from .statewise_joint_api_v2 import StatewiseJointV2
from .statewise_masks_v2 import StatewiseMaskProvider


STUDY = C.DOC / "nonsevere_v1/statewise_joint_ot_20261014_v2"
SOLVER = dict(max_p3p_triples=56, global_rotation_starts=24, max_seed_nfev=160,
    max_joint_iterations=250, maximum_union_branches=256, maximum_visible_mahalanobis=3.,
    analytic_jacobian=True)


@contextmanager
def no_coordinate_annotation_reads():
    original = builtins.open,io.open,Path.open
    forbidden = ("SCORING", "TARGETS", "CAL_INPUTS", "DATA_CONTRACT", "STATIC_VISIBILITY", "ORACLE_CAL_INPUTS", "FINAL_")
    def wrap(function):
        def guarded(path,*args,**kwargs):
            if any(token in str(path) for token in forbidden):
                raise AssertionError("Mechanism prediction attempted coordinate-annotation read: "+str(path))
            return function(path,*args,**kwargs)
        return guarded
    builtins.open,io.open,Path.open=map(wrap,original)
    try:
        yield
    finally:
        builtins.open,io.open,Path.open=original


def visibility_oracle(human_states):
    human=tuple(human_states)
    if len(human)!=8 or any(s not in {"O","V","S","T","U"} for s in human):
        raise ValueError("Known CAL visibility metadata must contain eight native O/V/S states")
    def produce(image,raw,hidden,roles,projected,model,*,pallet_mask=None,independent_truncation=None):
        records,eligible=geometry_state_gates(image,raw,hidden,roles,projected,
            independent_truncation=independent_truncation)
        for k in eligible:
            state=roles[k] if human[k]=="V" else "O" if human[k]=="O" else "U"
            records[k].update(state=state,abstained=state=="U",rgb_evaluated=False,p_visible=None,
                reason="known_CAL_human_visibility_oracle_only")
        # Missing/raw-OOB nonS queries and initial T hypotheses stay gated.
        # Human S never overrides the fixed initial registered S assignment.
        return dict(states=[r["state"] for r in records],records=records,rgb_queried_ids=[],
            p_visible=[None]*8,diagnostic_oracle=True,
            diagnostics=dict(visibility_oracle=True,deployable=False,primary_evaluation=False,
                human_visibility_input=True,GT_coordinates_input=False,state_RGB_queries=0,
                B_I_correction_still_current_RGB=True,S_fixed_initial_geometry_first=True))
    return produce


def domain_audit(point,domain,tolerance=2e-6):
    """Independent saved-domain inequalities, without a coordinate target."""
    q=np.asarray(point,float)
    slacks=[]
    if domain["shape"]=="ellipse":
        angle=np.deg2rad(domain.get("angle_deg",0.))
        rotation=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
        z=(q-np.asarray(domain["center_xy"]))@rotation/np.asarray(domain["half_axes_px"])
        slacks.append(1.-float(z@z))
    elif domain["shape"]=="box":
        x0,y0,x1,y1=domain["bounds_xyxy"]
        slacks.extend([q[0]-x0,x1-q[0],q[1]-y0,y1-q[1]])
    if domain["state"]=="T":
        h,w=domain["canvas_hw"]
        sides={"left":-q[0],"right":q[0]-w,"top":-q[1],"bottom":q[1]-h}
        side=domain.get("outside_canvas_side","unknown")
        if isinstance(side,(list,tuple)):
            slacks.append(max(sides[s] for s in side))
        else:
            slacks.append(max(sides.values()) if side=="unknown" else sides[side])
    for a,b,c in domain.get("halfplanes",[]):
        slacks.append(c-a*q[0]-b*q[1])
    return dict(native_id=domain["native_corner"],state=domain["state"],
        feasible=bool(np.isfinite(q).all() and slacks and min(slacks)>=-tolerance),
        minimum_slack=float(min(slacks)) if slacks else None,projected_xy=q.tolist())


def _distinct_candidate_poses(solution,scale):
    from scipy.spatial.transform import Rotation
    poses=[]
    for candidate in solution.get("candidates",[]):
        if not candidate.get("joint_feasible") or "R" not in candidate:
            continue
        R,t=np.asarray(candidate["R"]),np.asarray(candidate["t"])
        if any(Rotation.from_matrix(R@old[0].T).magnitude()<1e-5 and
               np.linalg.norm(t-old[1])<1e-5*scale for old in poses):
            continue
        poses.append((R,t))
    return poses


def postseal_mechanism_summary(rows):
    """Compare real finite solves using only predictions/domain inequalities."""
    summary=[]
    for row in rows:
        runtime=row["runtime"]
        xyz=np.asarray(runtime["initial_geometry"]["native_xyz"])
        camera=np.asarray(row["K"])
        domains=runtime["ot_domains"]["domains"]
        scale=float(np.linalg.norm(np.ptp(xyz,axis=0)))
        arms=[]
        for name,solution in row["arms"].items():
            cross=[]
            for R,t in _distinct_candidate_poses(solution,scale):
                projected,_=project_native(xyz,R,t,camera)
                audit=[domain_audit(projected[d["native_corner"]],d) for d in domains]
                cross.append(dict(all_original_domains_feasible=all(a["feasible"] for a in audit),audit=audit))
            accepted_audit=[]
            if solution["accepted"]:
                projected,_=project_native(xyz,solution["R"],solution["t"],camera)
                accepted_audit=[domain_audit(projected[d["native_corner"]],d) for d in domains]
            arms.append(dict(arm=name,accepted=solution["accepted"],status=solution["status"],
                fallback_reason=solution.get("fallback_reason"),
                finite_feasible_distinct_local_poses=len(cross),
                those_also_satisfying_all_original_domains=sum(p["all_original_domains_feasible"] for p in cross),
                chosen_pose_satisfies_all_original_domains=all(a["feasible"] for a in accepted_audit) if accepted_audit else None,
                chosen_pose_all_domain_audit=accepted_audit,
                corrected_BI_local_minima=solution.get("search",{}).get("corrected_BI_local_minima"),
                searched_initializations_per_branch=solution.get("search",{}).get("joint_initializations_per_branch"),
                union_branches=solution.get("search",{}).get("T_union_branches_searched"),
                finite_candidate_cross_domain_audit=cross))
        summary.append(dict(id=row["id"],states=runtime["states"],BI_count=len(runtime["frozen_visible_measurements"]),
            OT_ids=[d["native_corner"] for d in domains],arms=arms,
            uniqueness_or_global_infeasibility_proved=False))
    return summary


def run(args):
    import torch
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    out=Path(args.output)
    out.mkdir(parents=True,exist_ok=False)
    prereg_path=STUDY/"ot_domains/REAL_CROP_PREREGISTRATION.json"
    prereg=C.read(prereg_path)
    paths=[STUDY/"ot_domains"/p["crop_prediction_input"] for p in prereg["plans"]]
    frames=[C.read(path) for path in paths]
    if len(frames)!=3 or any(f["partition"]!="CAL" or f["GT_coordinates_used"] for f in frames):
        raise AssertionError("Only three existing preregistered CAL actual crops, no coordinate targets")
    model_path=STUDY/"state_visibility/bounded_ownership_spatial_v3/MODEL.json"
    ot_path=STUDY/"ot_domains/EXTERNAL_O_CAL_MAX_MODEL_N3_SUBPIX.json"
    descriptor=C.read(model_path)
    ot_model=C.read(ot_path)
    mask_config=C.DOC/"nonsevere_v1/BOUNDARY_OBSERVATION_CONFIG.json"
    provider=StatewiseMaskProvider.from_configs(descriptor,mask_config,args.device)
    code=Path(__file__).parent
    sources=[Path(__file__)]+[code/name for name in ("statewise_joint_api_v2.py","state_visibility_v2.py",
        "statewise_masks_v2.py","visible_bi_correction_v2.py","joint_ot_pnp.py","joint_ot_jacobian_v2.py",
        "ot_domains_v2.py","appearance.py","segmentation.py","common.py")]
    source_bindings={str(p.relative_to(C.ROOT)):C.sha(p) for p in sources}
    protocol=dict(schema="CAL_VISIBILITY_ORACLE_OT_MECHANISM_V2",CAL_frames=3,TEST_frames=0,severe_frames=0,
        diagnostic_oracle=True,deployable=False,primary_evaluation=False,
        human_visibility_input=True,GT_coordinate_input=False,source_human_states="existing original_human_states in preregistered realcrop inputs",
        state_head_executed=False,state_RGB_queries=0,S_fixed_initial_geometry_first=True,
        conditional_T_keeps_initial_geometry_gate=True,raw_OOB_is_not_recast_as_a_measured_point=True,
        B_I_source="fresh actual current RGB Tiny/SAM masks and distinct B/I image correctors",
        same_frozen_BI_all_arms=True,same_registered_global_initializations_all_arms=True,
        all_corrected_BI_roots_retained_by_same_core_solver=True,
        arms="ALL_OT, NO_OT (O/T toU), each single original O/T domain (otherO/T toU)",
        solver_configuration=SOLVER,device=args.device,OT_model_sha256=C.sha(ot_path),
        unused_state_model_descriptor_sha256=C.sha(model_path),mask_models=provider.bindings,
        crop_prereg_sha256=C.sha(prereg_path),input_sha256={str(p.relative_to(C.ROOT)):C.sha(p) for p in paths},
        image_file_sha256={f["image"]:C.sha(C.ROOT/f["image"]) for f in frames},
        source_sha256=source_bindings,
        scoring_rule="Seal all complete predictions/allarms before any CAL coordinate targets; no manufactured constraint effect",
        limitations=["Human visibility oracle removes a production-classifier failure; mechanism evidence only",
            "T remains conditional until postseal source annotations are transformed",
            "Actual crop replays translated frozen upstream predictions, not a fresh crop backbone forward",
            "Finite multistart results are not a global uniqueness/infeasibility proof"])
    for source in sources:
        snapshot=out/"code_snapshot"/source.relative_to(C.ROOT)
        snapshot.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,snapshot)
    C.write(out/"PROTOCOL.json",protocol)
    input_rows=[]
    for f in frames:
        image=cv2.imread(str(C.ROOT/f["image"]))
        input_rows.append(dict(id=f["record_id"],source_record_id=f["source_record_id"],partition="CAL",
            image=f["image"],image_sha256=C.sha(C.ROOT/f["image"]),raw_hw=list(image.shape[:2]),K=f["K"]))
    C.write(out/"DIAGNOSTIC_IMAGE_INPUTS.json",dict(frames=input_rows,GT_coordinates_used=False))
    start=time.monotonic()
    rows=[]
    with no_coordinate_annotation_reads():
        for index,f in enumerate(frames):
            image=cv2.imread(str(C.ROOT/f["image"]))
            oracle=visibility_oracle(f["original_human_states"])
            runtime=StatewiseJointV2(None,ot_model,provider,
                config=dict(solver=SOLVER,prediction_source="N3_SUBPIX",record_id=f["record_id"],device=args.device),
                state_producer=oracle).refine(image,f["raw_points"],f["K"],C.cuboid(*f["initial_pose"]["cf_extents"]),
                    f["initial_pose"],crop_transform=f["transform"])
            measurements=deepcopy(runtime["frozen_visible_measurements"])
            frozen=deepcopy(measurements)
            geometry=runtime["initial_geometry"]
            xyz=np.asarray(geometry["native_xyz"])
            initializations=[dict(name="REGISTERED_COARSE_INITIALIZATION_ONLY",R=geometry["R_cf"],t=geometry["centroid"])]
            domains=runtime["ot_domains"]["domains"]
            arms={"ALL_OT":runtime["joint_pose"]}
            subsets=[("NO_OT",[])]+[("ONLY_%s%d" % (d["state"],d["native_corner"]),[d]) for d in domains]
            for name,subset in subsets:
                active={d["native_corner"] for d in subset}
                states=["U" if s in {"O","T"} and k not in active else s for k,s in enumerate(runtime["states"])]
                arms[name]=solve_joint_ot_pnp(xyz,f["K"],states,deepcopy(measurements),deepcopy(subset),image.shape[:2],
                    initializations=deepcopy(initializations),config=SOLVER)
                if measurements!=frozen:
                    raise AssertionError("Mechanism arm mutated frozen B/I measurements")
                if arms[name].get("corrected_BI_seed")!=arms["ALL_OT"].get("corrected_BI_seed"):
                    raise AssertionError("Different arm changed corrected-only seed/root reference")
                if arms[name].get("search",{}).get("corrected_BI_local_minima")!=arms["ALL_OT"].get("search",{}).get("corrected_BI_local_minima"):
                    raise AssertionError("Different arm discarded B/I local roots before joint constraints")
            rows.append(dict(id=f["record_id"],source_record_id=f["source_record_id"],partition="CAL",
                cohort="NONDEPLOYABLE_HUMAN_VISIBILITY_ORACLE_REAL_CROP_MECHANISM",image=f["image"],K=f["K"],
                raw_hw=list(image.shape[:2]),crop_transform=f["transform"],original_human_states=f["original_human_states"],
                diagnostic_oracle=True,runtime=runtime,arms=arms))
            C.write(out/"LIVE_PROGRESS.json",dict(completed=index+1,total=3,id=f["record_id"],states=runtime["states"],
                BI_count=len(measurements),OT_ids=[d["native_corner"] for d in domains],
                arm_accepted={name:a["accepted"] for name,a in arms.items()},elapsed_seconds=time.monotonic()-start))
            print("MECHANISM_CAL",f["record_id"],runtime["states"],"BI",len(measurements),
                {name:a["accepted"] for name,a in arms.items()},flush=True)
    if {str(p.relative_to(C.ROOT)):C.sha(p) for p in sources}!=source_bindings:
        raise AssertionError("Executed mechanism sources changed; do not seal or score")
    C.save_rows(out/"PREDICTIONS.jsonl.gz",rows)
    C.write(out/"PREDICTION_SEAL.json",dict(schema="CAL_ORACLE_MECHANISM_PREDICTION_SEAL_V2",before_scoring=True,
        rows=len(rows),predictions_sha256=C.sha(out/"PREDICTIONS.jsonl.gz"),protocol_sha256=C.sha(out/"PROTOCOL.json"),
        source_sha256=source_bindings,GT_coordinate_reads=0,TEST_reads=0,FINAL_reads=0,
        diagnostic_oracle=True,primary_evaluation=False,elapsed_seconds=time.monotonic()-start))
    summary=postseal_mechanism_summary(rows)
    C.write(out/"POSTSEAL_MECHANISM_ONLY.json",dict(schema="PREDICTION_ONLY_CAL_MECHANISM_COMPARISON_V2",
        prediction_seal_sha256=C.sha(out/"PREDICTION_SEAL.json"),GT_coordinate_reads=0,
        diagnostic_oracle=True,rows=summary,
        interpretation="Counts describe finite continuous arm solves; zero change/redundancy/failure must be reported as observed"))
    print("MECHANISM_ALL_ARMS_SEALED",out,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=STUDY/"ot_mechanism_CAL_v1")
    parser.add_argument("--device",default="cpu")
    run(parser.parse_args())


if __name__=="__main__":
    main()
