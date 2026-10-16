"""Independent-frame execution of the unchanged statewise experiment.

Each spawned process loads frozen models and reads only prediction inputs.
The master seals the complete ordered population before the separate scorer.
No pose starts, domains or solver constraints are removed to gain speed.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from multiprocessing import get_context
from pathlib import Path
import shutil
import time

import cv2
from . import common as C
from .state_visibility_v2 import load_state_model
from .statewise_joint_api_v2 import StatewiseJointV2
from .statewise_masks_v2 import StatewiseMaskProvider
from .statewise_joint_experiment_v2 import COHORT,STUDY,_frames,_bi_only,crop_bindings,source_bindings


_WORKER_CONTEXT=None


def _init_worker(model_path,ot_path,device,solver_config):
    global _WORKER_CONTEXT
    import torch
    cv2.setNumThreads(1);torch.set_num_threads(1)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    model=load_state_model(model_path,device)
    provider=StatewiseMaskProvider.from_configs(C.read(STUDY/"state_visibility/MODEL.json"),
                                               COHORT/"BOUNDARY_OBSERVATION_CONFIG.json",device)
    _WORKER_CONTEXT=(model,C.read(ot_path),provider,device,solver_config)


def _run_frame(frame):
    model,ot_model,provider,device,solver_config=_WORKER_CONTEXT
    image_path=C.ROOT/frame["image"]
    if frame["image_sha256"] and C.sha(image_path)!=frame["image_sha256"]:
        raise ValueError("Image binding changed: "+frame["id"])
    image=cv2.imread(str(image_path))
    if image is None or list(image.shape[:2])!=frame["raw_hw"]:raise ValueError("Canvas mismatch")
    runtime=StatewiseJointV2(model,ot_model,provider,config=dict(solver=solver_config,
        prediction_source="N3_SUBPIX",record_id=frame["id"],device=device))
    result=runtime.refine(image,frame["raw_points"],frame["K"],C.cuboid(*frame["initial_pose"]["cf_extents"]),
                         frame["initial_pose"],crop_transform=frame["crop_transform"])
    bi=_bi_only(result,frame,solver_config)
    if result["frozen_visible_measurements"]!=bi["frozen_visible_measurements"]:
        raise AssertionError("BI changed between joint and BI-only arms")
    return dict(id=frame["id"],source_record_id=frame["source_record_id"],partition=frame["partition"],
        session=frame["session"],cohort=frame["cohort"],image=frame["image"],K=frame["K"],raw_hw=frame["raw_hw"],
        crop_transform=frame["crop_transform"],initial_pose=frame["initial_pose"],runtime=result,BI_only=bi)


def predict(output,model_path,device="cuda",workers=3):
    output=Path(output).resolve();model_path=Path(model_path).resolve()
    if output.exists():raise FileExistsError("Preserve prior frozen evaluation")
    if int(workers)!=workers or not 1<=workers<=3:raise ValueError("One to three independent workers required")
    frames=_frames();model_descriptor=C.read(model_path)
    ot_path=STUDY/"ot_domains/EXTERNAL_O_CAL_MAX_MODEL_N3_SUBPIX.json"
    mask_descriptor=C.read(STUDY/"state_visibility/MODEL.json")
    sam=C.read(COHORT/"BOUNDARY_OBSERVATION_CONFIG.json")
    solver_config=dict(max_p3p_triples=56,global_rotation_starts=24,max_seed_nfev=160,
        max_joint_iterations=250,maximum_union_branches=256,maximum_visible_mahalanobis=3.,
        analytic_jacobian=True)
    output.mkdir(parents=True)
    protocol=dict(schema="STATEWISE_JOINT_RGB_EVALUATION_V2",pipeline_contract="../METHOD_CONTRACT_KO.md",
        primary_population=dict(CAL=46,TEST=49,severe=0),natural_T_labels=0,
        TEST_status="REUSED_EXPLORATORY_NOT_NEW_UNSEEN",upstream_source="N3_SUBPIX frozen image predictions",
        arms=["ORIGINAL_PNP","FROZEN_BI_ONLY_PNP","ALL_OT_JOINT_SE3"],
        alternative_arms_reuse_exact_same_BI=True,OT_joint_per_corner_hard_constraints=True,
        conditional_T_uses_unknown_side_union=True,solver_configuration=solver_config,
        S_image_queries_allowed=0,S_region_constraints_allowed=0,O_domain_centers_are_point_correspondences=False,
        crop_status="actual RGB slices; translated frozen upstream q/K/coarse, no fresh backbone crop forward",
        crop_primary_evaluation=False,annotation_targets_used_for_prediction=False,
        state_model=dict(path=str(model_path.relative_to(C.ROOT)),sha256=C.sha(model_path),
            O_enabled=model_descriptor.get("external_O_enabled",True),CAL_crossfit_AUC=model_descriptor.get("crossfit_AUC")),
        OT_model=dict(path=str(ot_path.relative_to(C.ROOT)),sha256=C.sha(ot_path)),
        mask_models=dict(tiny_sha256=mask_descriptor["pallet_segmenter"]["weights_sha256"],
                         sam_sha256=sam["model_sha256"]["mobile_sam_checkpoint"]),
        input_sha256=C.sha(COHORT/"PREDICTION_INPUTS.json"),source_sha256=source_bindings(),
        crop_source_sha256=crop_bindings(),ids=[f["id"] for f in frames],independent_frame_workers=workers,
        exact_derivative_amendment=dict(prior_numeric_attempt="../evaluation_v1/ABORTED_BEFORE_SCORING.json",
            same_frozen_state_head=True,same_domains_and_hard_constraints=True,
            all_P3P_global_starts_preserved=True,TEST_annotations_not_read=True))
    C.write(output/"PROTOCOL.json",protocol)
    # Executed source snapshots make future diagnostic fixes auditable without
    # changing this run's sealed implementation.
    for relative,digest in protocol["source_sha256"].items():
        source=C.ROOT/relative;destination=output/"code_snapshot"/relative
        destination.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,destination)
        if C.sha(destination)!=digest:raise ValueError("Source snapshot mismatch")
    start=time.monotonic();by_id={}
    with ProcessPoolExecutor(max_workers=workers,mp_context=get_context("spawn"),initializer=_init_worker,
        initargs=(str(model_path),str(ot_path),device,solver_config)) as pool:
        futures={pool.submit(_run_frame,f):f["id"] for f in frames}
        for future in as_completed(futures):
            row=future.result()
            if row["id"]!=futures[future] or row["id"] in by_id:raise ValueError("Worker native record binding mismatch")
            by_id[row["id"]]=row
            completed=len(by_id)
            C.write(output/"LIVE_PROGRESS.json",dict(completed=completed,total=len(frames),last_id=row["id"],
                states=row["runtime"]["states"],accepted=row["runtime"]["accepted"],
                elapsed_seconds=time.monotonic()-start,completed_ids=sorted(by_id)))
            print(f"{completed}/{len(frames)} {row['id']} states={''.join(row['runtime']['states'])} accepted={row['runtime']['accepted']}",flush=True)
    rows=[by_id[f["id"]] for f in frames]
    if source_bindings()!=protocol["source_sha256"] or crop_bindings()!=protocol["crop_source_sha256"]:
        raise ValueError("Source changed during prediction; do not score")
    C.save_rows(output/"PREDICTIONS.jsonl.gz",rows)
    C.write(output/"PREDICTION_SEAL.json",dict(schema="STATEWISE_JOINT_SEAL_V2",before_scoring=True,
        predictions_sha256=C.sha(output/"PREDICTIONS.jsonl.gz"),protocol_sha256=C.sha(output/"PROTOCOL.json"),
        source_sha256=protocol["source_sha256"],rows=len(rows),severe_rows=0,
        elapsed_seconds=time.monotonic()-start,workers=workers))


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--model",type=Path,required=True);parser.add_argument("--device",default="cuda")
    parser.add_argument("--workers",type=int,default=3);args=parser.parse_args()
    predict(args.output,args.model,args.device,args.workers)


if __name__=="__main__":main()
