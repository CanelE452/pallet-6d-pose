from copy import deepcopy
from pathlib import Path
import hashlib
import shutil
import sys
import time
sys.path.insert(0,"/home/minjae/Documents/github/pallet-pose")

import cv2
import numpy as np
import torch
from threadpoolctl import threadpool_limits

from scripts.research.pallet_visibility_curriculum_20261013_v1 import common as C
from scripts.research.pallet_visibility_curriculum_20261013_v1.joint_ot_pnp import solve_joint_ot_pnp
from scripts.research.pallet_visibility_curriculum_20261013_v1.state_visibility_v2 import load_state_model
from scripts.research.pallet_visibility_curriculum_20261013_v1.statewise_joint_api_v2 import StatewiseJointV2
from scripts.research.pallet_visibility_curriculum_20261013_v1.statewise_masks_v2 import StatewiseMaskProvider
from scripts.research.pallet_visibility_curriculum_20261013_v1.statewise_joint_experiment_v2 import source_bindings

COHORT=C.DOC/"nonsevere_v1"
STUDY=COHORT/"statewise_joint_ot_20261014_v2"
OUTPUT=STUDY/"runtime_smoke_current_v3"
V2=STUDY/"evaluation_v2"
FIRST=STUDY/"runtime_smoke_guarded_v2"
CONFIG=dict(max_p3p_triples=56,global_rotation_starts=24,max_seed_nfev=160,max_joint_iterations=250,
    maximum_union_branches=256,maximum_visible_mahalanobis=3.,analytic_jacobian=True,
    protect_seed_projection=False,enforce_visible_geometry=True,geometry_tolerance_deg=2.,enforce_visible_canvas=True)


def array_sha(array):
    a=np.ascontiguousarray(array);h=hashlib.sha256(str((a.shape,str(a.dtype))).encode());h.update(a.tobytes());return h.hexdigest()


class Capture:
    def __init__(self,provider):self.provider=provider;self.calls=[]
    def __call__(self,image,**kwargs):
        bundle=self.provider(image,**kwargs)
        self.calls.append(dict(prompt_native_ids=deepcopy(kwargs["prompt_native_ids"]),
            prompt_points=kwargs["prompt_points"].tolist(),prompt_box_xyxy=deepcopy(kwargs["prompt_box_xyxy"]),
            state_mask_sha256=array_sha(bundle["state_pallet_mask"]),BI_mask_sha256=array_sha(bundle["pallet_mask"]),
            provenance=deepcopy(bundle.get("provenance",{}))))
        return bundle


def sealed_rows(directory):
    seal=C.read(directory/"PREDICTION_SEAL.json")
    assert seal["before_scoring"] and C.sha(directory/"PREDICTIONS.jsonl.gz")==seal["predictions_sha256"]
    assert C.sha(directory/"PROTOCOL.json")==seal["protocol_sha256"]
    return list(C.rows(directory/"PREDICTIONS.jsonl.gz"))


def main():
    cv2.setNumThreads(1);torch.set_num_threads(1)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    OUTPUT.mkdir(parents=True,exist_ok=False)
    cache={r["id"]:r for r in sealed_rows(V2)}
    first_cached=sealed_rows(FIRST)[0]["runtime"]
    v2_protocol=C.read(V2/"PROTOCOL.json")
    primary_path=C.ROOT/v2_protocol["state_model"]["path"]
    assert C.sha(primary_path)==v2_protocol["state_model"]["sha256"]
    first_path=STUDY/"state_visibility/MODEL.json"
    mask_descriptor=C.read(first_path)
    ot_path=STUDY/"ot_domains/EXTERNAL_O_CAL_MAX_MODEL_N3_SUBPIX.json"
    ot_model=C.read(ot_path)
    specs=[dict(case="primary_head_outside_CAL",record_id="eval_outside:1778651519235162880",model_path=primary_path,cache="evaluation_v2"),
           dict(case="primary_head_night_CAL",record_id="eval_night08:1779449488068910592",model_path=primary_path,cache="evaluation_v2"),
           dict(case="first_head_night_seen_CAL_diagnostic",record_id="eval_night08:1779449488068910592",model_path=first_path,cache="runtime_smoke_guarded_v2")]
    source_sha=source_bindings()
    for relative,digest in source_sha.items():
        p=OUTPUT/"code_snapshot"/relative;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(C.ROOT/relative,p)
        assert C.sha(p)==digest
    shutil.copyfile(__file__,OUTPUT/"runtime_smoke_run.py")
    protocol=dict(schema="FRESH_CURRENT_STATEWISE_RUNTIME_SMOKE_V3",solver_configuration=CONFIG,
        source_sha256=source_sha,script_sha256=C.sha(OUTPUT/"runtime_smoke_run.py"),
        cases=[{**s,"model_path":str(s["model_path"].relative_to(C.ROOT)),"model_sha256":C.sha(s["model_path"])}for s in specs],
        upstream_cache_seals={"evaluation_v2":C.sha(V2/"PREDICTION_SEAL.json"),"runtime_smoke_guarded_v2":C.sha(FIRST/"PREDICTION_SEAL.json")},
        OT_model_sha256=C.sha(ot_path),scope="fresh current-RGB forwards and API contract parity on preselected CAL examples",
        direct_observations_frozen_before_pose=True,all_individual_OT_joint=True,S_image_queries_allowed=0,
        GT_reads=0,scoring_outputs_read=False,first_model_case_is_seen_CAL_diagnostic=True,
        pose_family_representative_R_parity_not_required=True,cached_replay_initialization_name="SUPPLIED_START_0",
        API_initialization_name="REGISTERED_COARSE_INITIALIZATION_ONLY")
    C.write(OUTPUT/"PROTOCOL.json",protocol)
    models={};rows=[];checks=[];start=time.monotonic()
    for spec in specs:
        row=cache[spec["record_id"]];assert row["partition"]=="CAL"
        frozen=row["runtime"] if spec["cache"]=="evaluation_v2" else first_cached
        input_path=C.ROOT/row["image"]
        image=cv2.imread(str(input_path));assert list(image.shape[:2])==row["raw_hw"]
        # The sealed source input also binds the original file bytes.
        frame=next(f for f in C.read(COHORT/"PREDICTION_INPUTS.json")["frames"] if f["id"]==spec["record_id"])
        assert C.sha(input_path)==frame["image_sha256"]
        path=spec["model_path"]
        if path not in models:models[path]=load_state_model(str(path),"cuda")
        capture=Capture(StatewiseMaskProvider.from_configs(mask_descriptor,COHORT/"BOUNDARY_OBSERVATION_CONFIG.json","cuda"))
        runtime=StatewiseJointV2(models[path],ot_model,capture,
            config=dict(solver=CONFIG,prediction_source="N3_SUBPIX",record_id=spec["record_id"],device="cuda"))
        raw=np.array(frozen["original_raw_points"],float);original_raw=raw.copy()
        coarse=deepcopy(frozen["original_coarse_pose"]);before_coarse=deepcopy(coarse)
        X=C.cuboid(*coarse["cf_extents"])
        case_start=time.monotonic()
        original=runtime.refine(image,raw,row["K"],X,coarse)
        self_ids=[i for i,s in enumerate(original["states"])if s=="S"]
        assert self_ids
        perturbed_raw=raw.copy()
        for j,i in enumerate(self_ids):perturbed_raw[i]=[1e8+j*12345.,-1e8-j*34567.]
        perturbed=runtime.refine(image,perturbed_raw,row["K"],X,coarse)
        required=["states","state_observation","mask_audit","visible_correction","ot_domains",
                  "frozen_visible_measurements","joint_pose","actual_pose","keypoint_readout"]
        invariance={field:original[field]==perturbed[field]for field in required}
        masks_equal=len(capture.calls)==2 and capture.calls[0]==capture.calls[1]
        assert masks_equal and all(invariance.values())
        assert all(i not in capture.calls[0]["prompt_native_ids"]for i in self_ids)
        cache_parity={field:original[field]==frozen[field]for field in["states","frozen_visible_measurements","ot_domains","visible_correction"]}
        # Reproduce the blind replay's unnamed supplied start on the same
        # immutable cache. Alternative initialization labels may affect a
        # representative choice in an ambiguous family, never the observations.
        cached_solution=solve_joint_ot_pnp(X,row["K"],frozen["states"],deepcopy(frozen["frozen_visible_measurements"]),
            deepcopy(frozen["ot_domains"]["domains"]),row["raw_hw"],
            initializations=[dict(R=coarse["R_cf"],t=coarse["centroid"])],config=CONFIG)
        joint=original["joint_pose"];accepted=bool(joint["accepted"])
        if accepted:
            assert all(a["feasible"]for a in joint["joint_constraint_audit"])
            assert all(a["visible_support_guards_pass"]for a in joint["visible_protection_audit"])
            for m in original["frozen_visible_measurements"]:
                assert original["keypoint_readout"]["reported_points"][m["native_corner"]]==m["xy"]
            if len(original["frozen_visible_measurements"])==2:
                assert joint["status"]=="joint_feasible_pose_family" and joint["ambiguity"]["detected"]
                assert original["keypoint_readout"]["OT_solved"] is False
                assert original["keypoint_readout"]["OT_unique_solution_claimed"] is False
        Q=np.array(coarse["R_cf"]).T@np.array(coarse["R_physical"])
        np.testing.assert_allclose(original["actual_pose"]["R_physical"],np.array(original["actual_pose"]["R_cf"])@Q,atol=1e-12,rtol=0)
        assert original["keypoint_readout"]["reported_points"][8]==raw[8].tolist()
        np.testing.assert_array_equal(raw,original_raw);assert coarse==before_coarse
        comparison=dict(accepted_same=accepted==cached_solution["accepted"],status_API=joint["status"],status_cached=cached_solution["status"],
            representative_exact_R_required=False if (joint.get("ambiguity")or{}).get("detected") else True)
        if accepted and cached_solution["accepted"]:
            comparison.update(objective_difference=abs(joint["objective"]-cached_solution["objective"]),
                rotation_matrix_difference=float(np.linalg.norm(np.array(joint["R"])-np.array(cached_solution["R"]))),
                translation_difference_m=float(np.linalg.norm(np.array(joint["t"])-np.array(cached_solution["t"]))),
                cached_all_constraints_pass=all(a["feasible"]for a in cached_solution["joint_constraint_audit"]))
        checks.append(dict(case=spec["case"],record_id=spec["record_id"],states=original["states"],S_ids=self_ids,
            O_T_ids=[d["native_corner"]for d in original["ot_domains"]["domains"]],BI_count=len(original["frozen_visible_measurements"]),
            accepted=accepted,status=joint["status"],material_ambiguity=original["ambiguity_audit"]["material_ambiguity"],
            cache_field_exact_parity=cache_parity,Sraw_perturb_field_exact_invariance=invariance,
            both_Tiny_SAM_mask_exact_invariance=masks_equal,mask_capture=capture.calls[0],
            direct_observations_and_registration_preserved=True,API_vs_cached_replay=comparison,
            elapsed_seconds=time.monotonic()-case_start))
        for index,(variant,value)in enumerate([("ORIGINAL",original),("S_RAW_EXTREME",perturbed)]):
            rows.append(dict(id=spec["case"]+"/"+variant,source_record_id=spec["record_id"],partition="CAL",variant=variant,
                model_sha256=C.sha(path),runtime=value,mask_capture=capture.calls[index]))
        rows.append(dict(id=spec["case"]+"/CACHED_CURRENT_SOLVER",source_record_id=spec["record_id"],partition="CAL",
                         variant="CACHED_CURRENT_SOLVER",solution=cached_solution))
        print({key:checks[-1][key]for key in["case","states","BI_count","accepted","status","material_ambiguity","cache_field_exact_parity","elapsed_seconds"]},flush=True)
    C.save_rows(OUTPUT/"PREDICTIONS.jsonl.gz",rows)
    assert source_bindings()==source_sha
    C.write(OUTPUT/"PREDICTION_SEAL.json",dict(schema="FRESH_CURRENT_RUNTIME_SEAL_V3",before_scoring=True,
        GT_reads=0,score_invoked=False,rows=len(rows),predictions_sha256=C.sha(OUTPUT/"PREDICTIONS.jsonl.gz"),
        protocol_sha256=C.sha(OUTPUT/"PROTOCOL.json"),source_sha256=source_sha,elapsed_seconds=time.monotonic()-start))
    C.write(OUTPUT/"CONTRACT_CHECKS.json",dict(schema="FRESH_CURRENT_RUNTIME_POSTSEAL_CHECKS_V3",cases=checks,
        prediction_seal_sha256=C.sha(OUTPUT/"PREDICTION_SEAL.json"),GT_reads=0,scoring_outputs_read=False,
        all_cache_upstream_fields_exactly_equal=all(all(c["cache_field_exact_parity"].values())for c in checks),
        all_S_raw_perturbations_exactly_invariant=all(all(c["Sraw_perturb_field_exact_invariance"].values())for c in checks),
        elapsed_seconds=time.monotonic()-start,
        limitations=["CAL-only executable smoke, not an independent generalization result",
            "First state model case explicitly uses already seen CAL",
            "Pose-family representative equality is not a unique-pose assertion",
            "Classifier and region errors are not repaired using labels or coordinate targets"]))
    print(dict(prediction_sealed=True,output=str(OUTPUT),elapsed_seconds=time.monotonic()-start),flush=True)

if __name__=="__main__":
    with threadpool_limits(limits=1):main()
