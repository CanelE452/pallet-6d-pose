"""Prediction-sealed fresh RGB evaluation of the actual statewise runtime.

Predict never reads annotation targets. Score is a separate invocation that
verifies the complete prediction/source seal before opening locked labels.
Historical TEST is reused exploratory evidence, not a new unseen test.
"""
import argparse
from collections import Counter,defaultdict
from copy import deepcopy
from pathlib import Path
import time

import cv2
import numpy as np

from . import common as C
from .joint_ot_pnp import solve_joint_ot_pnp
from .scoring import distribution,pose_metric,rotation_y
from .state_visibility_v2 import load_state_model
from .statewise_joint_api_v2 import StatewiseJointV2
from .statewise_masks_v2 import StatewiseMaskProvider


COHORT=C.DOC/"nonsevere_v1"
STUDY=COHORT/"statewise_joint_ot_20261014_v2"
SOURCE_FILES=("statewise_joint_experiment_v2.py","statewise_joint_api_v2.py","statewise_masks_v2.py",
    "state_visibility_v2.py","visible_bi_correction_v2.py","joint_ot_pnp.py","ot_domains_v2.py",
    "joint_ot_jacobian_v2.py","statewise_joint_parallel_v2.py","common.py","appearance.py","boundary_refine.py",
    "interior_observation.py","joint_observation.py","segmentation.py","scoring.py")


def source_bindings():
    root=Path(__file__).parent
    result={str((root/name).relative_to(C.ROOT)):C.sha(root/name) for name in SOURCE_FILES}
    for name in ("uncertainty.py","foundation.py"):
        path=root.parent/"pallet_foundation_mask_20261013_v1"/name
        result[str(path.relative_to(C.ROOT))]=C.sha(path)
    return result


def _bi_only(result,frame,config):
    """Same corrected observations, independently solve without all OT domains."""
    states=["U" if s in ("O","T") else s for s in result["states"]]
    coarse=frame["initial_pose"]
    solved=solve_joint_ot_pnp(C.cuboid(*coarse["cf_extents"]),frame["K"],states,
        deepcopy(result["frozen_visible_measurements"]),[],frame["raw_hw"],
        initializations=[dict(R=coarse["R_cf"],t=coarse["centroid"])],config=config)
    Q=np.asarray(coarse["R_cf"]).T@np.asarray(coarse["R_physical"])
    if solved["accepted"]:
        pose=dict(available=True,R_cf=solved["R"],R_physical=(np.asarray(solved["R"])@Q).tolist(),
                  centroid=solved["t"],cf_extents=coarse["cf_extents"],source="CORRECTED_BI_ONLY_PNP")
    else:
        seed=solved.get("corrected_BI_seed") or {}
        if seed.get("R") is not None and seed.get("t") is not None and seed.get("visible_measurement_budgets_pass"):
            pose=dict(available=True,R_cf=seed["R"],R_physical=(np.asarray(seed["R"])@Q).tolist(),
                      centroid=seed["t"],cf_extents=coarse["cf_extents"],source="CORRECTED_BI_SEED_FALLBACK")
        else:pose=dict(coarse,source="ORIGINAL_PNP_FALLBACK")
    return dict(actual_pose=pose,solution=solved,states=states,
        frozen_visible_measurements=deepcopy(result["frozen_visible_measurements"]))


def _frames():
    source=C.read(COHORT/"PREDICTION_INPUTS.json")
    protocol=C.read(COHORT/"COHORT_PROTOCOL.json")
    frames=source["frames"]
    if len(frames)!=95 or Counter(f["partition"] for f in frames)!={"CAL":46,"TEST":49}:
        raise ValueError("Complete frozen nonsevere95 cohort required")
    if set(f["id"] for f in frames)&set(protocol["excluded_severe_ids"]):
        raise ValueError("A severe record entered primary predictions")
    result=[]
    for f in frames:
        result.append(dict(id=f["id"],source_record_id=f["id"],partition=f["partition"],session=f["session"],
            image=f["image"],image_sha256=f["image_sha256"],K=f["K"],raw_hw=f["raw_hw"],
            raw_points=f["points"]["N3_SUBPIX"],initial_pose=f["initial_poses"]["N3_SUBPIX"],
            cohort="PRIMARY_NONSEVERE",crop_transform=None))
    plans=C.read(STUDY/"ot_domains/REAL_CROP_PREREGISTRATION.json")["plans"]
    for plan in plans:
        f=C.read(STUDY/"ot_domains"/plan["crop_prediction_input"])
        result.append(dict(id=f["record_id"],source_record_id=f["source_record_id"],partition="CAL",
            session=f["source_record_id"].split(":")[0],image=f["image"],image_sha256=None,
            K=f["K"],raw_hw=f["transform"]["canvas_hw"],raw_points=f["raw_points"],
            initial_pose=f["initial_pose"],cohort="REAL_RGB_CROP_CONDITIONED_REPLAY",crop_transform=f["transform"]))
    return result


def crop_bindings():
    directory=STUDY/"ot_domains"
    prereg=directory/"REAL_CROP_PREREGISTRATION.json"
    paths=[prereg]
    for plan in C.read(prereg)["plans"]:
        paths.extend([directory/plan["crop_prediction_input"],directory/plan["crop_image"]])
    return {str(path.relative_to(C.ROOT)):C.sha(path) for path in paths}


def predict(output,model_path,device="cuda"):
    import torch
    cv2.setNumThreads(1);torch.set_num_threads(1)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    output=Path(output).resolve();model_path=Path(model_path).resolve()
    if output.exists():raise FileExistsError("Preserve prior frozen evaluations: "+str(output))
    frames=_frames()
    model=load_state_model(str(model_path),device)
    # Mask ownership is separately bound; RGB26 models need no network/Tiny
    # features but their B corrector still gets a real current target mask.
    mask_descriptor=C.read(STUDY/"state_visibility/MODEL.json")
    provider=StatewiseMaskProvider.from_configs(mask_descriptor,COHORT/"BOUNDARY_OBSERVATION_CONFIG.json",device)
    ot_path=STUDY/"ot_domains/EXTERNAL_O_CAL_MAX_MODEL_N3_SUBPIX.json"
    ot_model=C.read(ot_path)
    solver_config=dict(max_p3p_triples=56,global_rotation_starts=24,max_seed_nfev=160,
        max_joint_iterations=250,maximum_union_branches=256,maximum_visible_mahalanobis=3.)
    output.mkdir(parents=True)
    protocol=dict(schema="STATEWISE_JOINT_RGB_EVALUATION_V2",pipeline_contract="../METHOD_CONTRACT_KO.md",
        primary_population=dict(CAL=46,TEST=49,severe=0),natural_T_labels=0,
        TEST_status="REUSED_EXPLORATORY_NOT_NEW_UNSEEN",upstream_source="N3_SUBPIX frozen image predictions",
        arms=["ORIGINAL_PNP","FROZEN_BI_ONLY_PNP","ALL_OT_JOINT_SE3"],
        alternative_arms_reuse_exact_same_BI=True,OT_joint_per_corner_hard_constraints=True,
        conditional_T_uses_unknown_side_union=True,solver_configuration=solver_config,
        S_image_queries_allowed=0,S_region_constraints_allowed=0,
        O_domain_centers_are_point_correspondences=False,
        crop_status="actual RGB slices; translated frozen upstream q/K/coarse, no fresh backbone crop forward",
        crop_primary_evaluation=False,annotation_targets_used_for_prediction=False,
        state_model=dict(path=str(model_path.relative_to(C.ROOT)),sha256=C.sha(model_path),
            O_enabled=model.descriptor.get("external_O_enabled",True),CAL_crossfit_AUC=model.descriptor.get("crossfit_AUC")),
        OT_model=dict(path=str(ot_path.relative_to(C.ROOT)),sha256=C.sha(ot_path)),
        mask_models=provider.bindings,input_sha256=C.sha(COHORT/"PREDICTION_INPUTS.json"),
        source_sha256=source_bindings(),crop_source_sha256=crop_bindings(),ids=[f["id"] for f in frames])
    C.write(output/"PROTOCOL.json",protocol)
    rows=[];start=time.monotonic()
    for i,frame in enumerate(frames):
        image_path=C.ROOT/frame["image"]
        if frame["image_sha256"] and C.sha(image_path)!=frame["image_sha256"]:
            raise ValueError("Image file binding changed: "+frame["id"])
        image=cv2.imread(str(image_path))
        if image is None or list(image.shape[:2])!=frame["raw_hw"]:raise ValueError("Actual image canvas mismatch")
        runtime=StatewiseJointV2(model,ot_model,provider,
            config=dict(solver=solver_config,prediction_source="N3_SUBPIX",record_id=frame["id"],device=device))
        result=runtime.refine(image,frame["raw_points"],frame["K"],C.cuboid(*frame["initial_pose"]["cf_extents"]),
                             frame["initial_pose"],crop_transform=frame["crop_transform"])
        bi=_bi_only(result,frame,solver_config)
        if result["frozen_visible_measurements"]!=bi["frozen_visible_measurements"]:
            raise AssertionError("BI changed between joint and BI-only arms")
        rows.append(dict(id=frame["id"],source_record_id=frame["source_record_id"],partition=frame["partition"],
            session=frame["session"],cohort=frame["cohort"],image=frame["image"],K=frame["K"],raw_hw=frame["raw_hw"],
            crop_transform=frame["crop_transform"],initial_pose=frame["initial_pose"],
            runtime=result,BI_only=bi))
        C.write(output/"LIVE_PROGRESS.json",dict(completed=i+1,total=len(frames),last_id=frame["id"],
            states=result["states"],accepted=result["accepted"],elapsed_seconds=time.monotonic()-start))
        print(f"{i+1}/{len(frames)} {frame['id']} states={''.join(result['states'])} accepted={result['accepted']}",flush=True)
    C.save_rows(output/"PREDICTIONS.jsonl.gz",rows)
    if source_bindings()!=protocol["source_sha256"]:raise ValueError("Source changed during prediction; do not score")
    if crop_bindings()!=protocol["crop_source_sha256"]:raise ValueError("Crop inputs changed during prediction")
    C.write(output/"PREDICTION_SEAL.json",dict(schema="STATEWISE_JOINT_SEAL_V2",before_scoring=True,
        predictions_sha256=C.sha(output/"PREDICTIONS.jsonl.gz"),protocol_sha256=C.sha(output/"PROTOCOL.json"),
        source_sha256=protocol["source_sha256"],rows=len(rows),severe_rows=0,elapsed_seconds=time.monotonic()-start))


def _model_compatible(coarse,truth,mapping):
    Q=np.asarray(coarse["R_cf"]).T@np.asarray(coarse["R_physical"])
    Qref=np.asarray(truth["body_R"]).T@np.asarray(truth["R"])
    native_physical=C.cuboid(*coarse["cf_extents"])@Q
    mapped=C.cuboid(*truth["body_xyz"])@Qref
    mapped=mapped[np.asarray(mapping,int)]
    gap=min(float(np.linalg.norm(native_physical@rotation_y(j*2*np.pi/truth["order"]).T-mapped,axis=1).max())
            for j in range(truth["order"]))
    return gap<1e-6,gap


def score(output):
    output=Path(output).resolve();seal=C.read(output/"PREDICTION_SEAL.json")
    protocol=C.read(output/"PROTOCOL.json")
    if not seal["before_scoring"] or C.sha(output/"PREDICTIONS.jsonl.gz")!=seal["predictions_sha256"]:
        raise ValueError("Prediction seal invalid")
    if C.sha(output/"PROTOCOL.json")!=seal["protocol_sha256"] or source_bindings()!=seal["source_sha256"]:
        raise ValueError("Protocol/source seal changed")
    if crop_bindings()!=protocol["crop_source_sha256"]:raise ValueError("Crop source seal changed")
    rows=list(C.rows(output/"PREDICTIONS.jsonl.gz"))
    if [r["id"] for r in rows]!=protocol["ids"]:raise ValueError("Complete sealed population mismatch")
    # Annotation-coordinate access starts here, after every prediction seal.
    labels={}
    for part,path in (("CAL",COHORT/"OOF_pose_v1/ISOLATED_CAL_SCORING.json"),
                      ("TEST",COHORT/"test_reused_v1/ISOLATED_TEST_SCORING.json")):
        labels.update(C.read(path)["rows"])
    values=[];matrix=defaultdict(Counter);pixels=defaultdict(lambda:defaultdict(list));crops=[]
    for row in rows:
        labeled=labels[row["source_record_id"]];metadata=labeled["metadata"];target=labeled["target"]
        truth=target["truth"];mapping=metadata["native_to_reference_corner_ids"]
        compatible,gap=_model_compatible(row["initial_pose"],truth,mapping)
        runtime=row["runtime"];cohort=row["cohort"];part=row["partition"]
        pose_arms={"ORIGINAL_PNP":row["initial_pose"],"FROZEN_BI_ONLY_PNP":row["BI_only"]["actual_pose"],
                   "ALL_OT_JOINT_SE3":runtime["actual_pose"]}
        metrics={arm:pose_metric(pose,truth) for arm,pose in pose_arms.items()}
        human=metadata["human_states_native"]
        if cohort=="PRIMARY_NONSEVERE":
            for a,p in zip(human,runtime["states"]):matrix[part][(a,p)]+=1
        gt=np.asarray(target["gt"],float)[:8][np.asarray(mapping,int)]
        valid=np.asarray(target["valid"],bool)[:8][np.asarray(mapping,int)]
        if row["crop_transform"]:
            x0,y0,_,_=row["crop_transform"]["crop_xyxy"];gt=gt-np.array([x0,y0])
            h,w=row["raw_hw"]
            actual_T=[i for i in range(8) if valid[i] and human[i]=="DIRECT_VISIBLE"
                      and (gt[i,0]<0 or gt[i,0]>=w or gt[i,1]<0 or gt[i,1]>=h)]
            crops.append(dict(id=row["id"],actual_transformed_visible_T_ids=actual_T,
                predicted_T_ids=[i for i,s in enumerate(runtime["states"]) if s=="T"],states=runtime["states"],
                native_gt=gt.tolist(),native_gt_valid=valid.tolist(),joint_accepted=runtime["accepted"],
                source_visible_annotation_not_independent_pose_certificate=True))
        if cohort=="PRIMARY_NONSEVERE" and compatible:
            q=np.asarray(runtime["original_raw_points"],float)
            meas=np.asarray(runtime["visible_correction"]["measured_points"],float)
            rigid=np.asarray(runtime["keypoint_readout"]["rigid_model_corners"],float)
            for i,s in enumerate(runtime["states"]):
                if valid[i] and human[i]=="DIRECT_VISIBLE" and s in ("B","I"):
                    for arm,points in (("BACKBONE",q),("FROZEN_BI",meas),("FINAL_RIGID",rigid)):
                        pixels[part][arm].append(float(np.linalg.norm(points[i]-gt[i])))
        joint=runtime["joint_pose"];bi=row["BI_only"]["solution"]
        delta_R=float(np.linalg.norm(np.asarray(runtime["actual_pose"]["R_cf"])-np.asarray(row["BI_only"]["actual_pose"]["R_cf"])))
        delta_t=float(np.linalg.norm(np.asarray(runtime["actual_pose"]["centroid"])-np.asarray(row["BI_only"]["actual_pose"]["centroid"])))
        values.append(dict(id=row["id"],partition=part,cohort=cohort,metrics=metrics,states=runtime["states"],
            compatible_pixel_model=compatible,physical_native_gap_m=gap,joint_accepted=runtime["accepted"],
            joint_status=joint["status"],joint_OT_count=len(runtime["ot_domains"]["domains"]),
            total_OT_states=sum(s in ("O","T") for s in runtime["states"]),
            all_OT_feasible=bool(joint["accepted"] and all(a["feasible"] for a in joint["joint_constraint_audit"])),
            BI_count=len(runtime["frozen_visible_measurements"]),
            BI_corrected_ids=[r["native_corner"] for r in runtime["visible_correction"]["records"] if r["corrected"]],
            material_ambiguity=runtime["ambiguity_audit"]["material_ambiguity"],
            joint_changes_BI_only_pose=delta_R>1e-6 or delta_t>1e-7,
            pose_difference_from_BI_only=dict(matrix_norm=delta_R,translation_m=delta_t),
            visible_protection_audit=joint.get("visible_protection_audit"),
            human_states_native=human,native_gt=gt.tolist(),native_gt_valid=valid.tolist()))
    summary={}
    for part in ("CAL","TEST"):
        group=[v for v in values if v["partition"]==part and v["cohort"]=="PRIMARY_NONSEVERE"]
        summary[part]=dict(frames=len(group),joint_accepted=sum(v["joint_accepted"] for v in group),
            with_OT=sum(v["total_OT_states"]>0 for v in group),
            joint_changes_BI_only_pose=sum(v["joint_changes_BI_only_pose"] for v in group if v["total_OT_states"]),
            fallback_status_counts=dict(Counter(v["joint_status"] for v in group)),
            material_ambiguity=sum(v["material_ambiguity"] for v in group),
            BI_corrections=sum(len(v["BI_corrected_ids"]) for v in group),
            pose={arm:{name:distribution([float(v["metrics"][arm][name]) for v in group])
                       for name in ("translation_cm","rotation_deg","ADDsym_cm","success_5cm_5deg")}
                  for arm in ("ORIGINAL_PNP","FROZEN_BI_ONLY_PNP","ALL_OT_JOINT_SE3")},
            state_confusion=[dict(actual=a,predicted=p,n=n) for (a,p),n in sorted(matrix[part].items())],
            compatible_visible_pixel_metrics={arm:distribution(x) for arm,x in pixels[part].items()})
    C.write(output/"POSTSEAL_SCORING.json",dict(schema="STATEWISE_JOINT_POSTSEAL_SCORING_V2",rows=values,crops=crops,
        summary=summary,prediction_seal_sha256=C.sha(output/"PREDICTION_SEAL.json"),
        limitations=["TEST already reused; exploratory only","Pose targets are historical derived reference poses",
            "Pixel comparison limited to preexisting physically compatible native mapping; no GT reindex or branch choice",
            "Natural T absent; crops replay frozen upstream predictions rather than retrain/rerun backbone",
            "No automatic O success claimed when CAL O gate disables output","Finite multi-start local solve is not global proof"]))
    print(C.finite(summary))


def main():
    parser=argparse.ArgumentParser();parser.add_argument("action",choices=("predict","score"))
    parser.add_argument("--output",type=Path,required=True);parser.add_argument("--model",type=Path)
    parser.add_argument("--device",default="cuda");args=parser.parse_args()
    if args.action=="predict":
        if args.model is None:parser.error("predict requires --model")
        predict(args.output,args.model,args.device)
    else:score(args.output)


if __name__=="__main__":main()
