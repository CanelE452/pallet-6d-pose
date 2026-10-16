from copy import deepcopy
import hashlib
from pathlib import Path
import shutil
import sys
import time

sys.path.insert(0, "/home/minjae/Documents/github/pallet-pose")

import cv2
import numpy as np
import torch

from scripts.research.pallet_visibility_curriculum_20261013_v1 import common as C
from scripts.research.pallet_visibility_curriculum_20261013_v1.state_visibility_v2 import load_state_model
from scripts.research.pallet_visibility_curriculum_20261013_v1.statewise_joint_api_v2 import StatewiseJointV2
from scripts.research.pallet_visibility_curriculum_20261013_v1.statewise_masks_v2 import StatewiseMaskProvider

COHORT = C.DOC / "nonsevere_v1"
STUDY = COHORT / "statewise_joint_ot_20261014_v2"
OUTPUT = STUDY / "runtime_smoke_guarded_v2"
FRAME_ID = "eval_night08:1779449488068910592"
RUNTIME_ROOT = Path("scripts/research/pallet_visibility_curriculum_20261013_v1")
SOURCES = [RUNTIME_ROOT/name for name in (
    "statewise_joint_api_v2.py", "statewise_masks_v2.py", "state_visibility_v2.py",
    "visible_bi_correction_v2.py", "joint_ot_pnp.py", "ot_domains_v2.py", "common.py",
    "joint_observation.py", "appearance.py", "boundary_refine.py", "interior_observation.py", "segmentation.py")]
SOURCES += [Path("scripts/research/pallet_foundation_mask_20261013_v1")/name
            for name in ("foundation.py", "uncertainty.py")]


def array_sha(array):
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256(str((value.shape, str(value.dtype))).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


class Capture:
    def __init__(self, provider):
        self.provider, self.calls = provider, []

    def __call__(self, image, **kwargs):
        bundle = self.provider(image, **kwargs)
        self.calls.append(dict(prompt_native_ids=kwargs["prompt_native_ids"],
            prompt_points=kwargs["prompt_points"].tolist(), prompt_box_xyxy=kwargs["prompt_box_xyxy"],
            state_mask_sha256=array_sha(bundle["state_pallet_mask"]),
            BI_mask_sha256=array_sha(bundle["pallet_mask"]),
            provenance=deepcopy(bundle.get("provenance", {}))))
        return bundle


def main():
    cv2.setNumThreads(1); torch.set_num_threads(1)
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.allow_tf32=False; torch.backends.cuda.matmul.allow_tf32=False
    OUTPUT.mkdir(parents=True, exist_ok=False)
    input_path = COHORT/"PREDICTION_INPUTS.json"
    frame = next(f for f in C.read(input_path)["frames"] if f["id"] == FRAME_ID)
    assert frame["partition"] == "CAL"
    image_path = C.ROOT/frame["image"]
    assert C.sha(image_path) == frame["image_sha256"]
    image = cv2.imread(str(image_path)); assert list(image.shape[:2]) == frame["raw_hw"]
    model_path = STUDY/"state_visibility/MODEL.json"
    freeze_path = STUDY/"state_visibility/MODEL_FREEZE.json"
    freeze = C.read(freeze_path)
    assert freeze["complete"] and C.sha(model_path) == freeze["model"]["sha256"]
    model = load_state_model(str(model_path), "cuda")
    descriptor = C.read(model_path)
    provider = StatewiseMaskProvider.from_configs(descriptor, COHORT/"BOUNDARY_OBSERVATION_CONFIG.json", "cuda")
    capture = Capture(provider)
    ot_path = STUDY/"ot_domains/EXTERNAL_O_CAL_MAX_MODEL_N3_SUBPIX.json"
    ot_model = C.read(ot_path)
    solver_cfg = dict(max_p3p_triples=56, global_rotation_starts=24, max_seed_nfev=160,
                      max_joint_iterations=250, maximum_union_branches=256, maximum_visible_mahalanobis=3.)
    source_bindings = {str(path): C.sha(C.ROOT/path) for path in SOURCES}
    for path in SOURCES:
        snapshot = OUTPUT/"code_snapshot"/path
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(C.ROOT/path, snapshot)
    shutil.copyfile(__file__, OUTPUT/"runtime_smoke_run.py")
    protocol = dict(schema="STATEWISE_REAL_RUNTIME_SMOKE_GUARDED_V2", record_id=FRAME_ID,
        partition="CAL", scope="first frozen state classifier, seen CAL exploratory contract smoke",
        purpose="fresh Tiny+SAM+actual state+BI correction+simultaneous allOT constraints and Sraw perturb canary",
        state_model_sha256=C.sha(model_path), model_freeze_sha256=C.sha(freeze_path),
        model_parameters_sha256=C.sha(model_path.parent/descriptor["arrays_file"]),
        OT_model_sha256=C.sha(ot_path), mask_models=provider.bindings,
        prediction_input_sha256=C.sha(input_path), image_file_sha256=C.sha(image_path),
        source_sha256=source_bindings, run_script_sha256=C.sha(OUTPUT/"runtime_smoke_run.py"),
        solver_configuration=solver_cfg, variants=["ORIGINAL", "S_RAW_EXTREME"],
        GT_reads=0, score_invoked=False, no_new_unseen_generalization_claim=True,
        no_CAL_or_TEST_target_coordinate_reads=True)
    C.write(OUTPUT/"PROTOCOL.json", protocol)
    coarse = deepcopy(frame["initial_poses"]["N3_SUBPIX"])
    raw = np.array(frame["points"]["N3_SUBPIX"], float)
    runtime = StatewiseJointV2(model,ot_model,capture,
        config=dict(solver=solver_cfg,prediction_source="N3_SUBPIX",record_id=FRAME_ID,device="cuda"))
    start = time.monotonic()
    original = runtime.refine(image,raw,frame["K"],C.cuboid(*coarse["cf_extents"]),coarse)
    self_ids = [i for i,s in enumerate(original["states"]) if s == "S"]
    assert self_ids
    perturbed_raw = raw.copy()
    for index,i in enumerate(self_ids): perturbed_raw[i] = [1e8+index*12345,-1e8-index*34567]
    perturbed = runtime.refine(image,perturbed_raw,frame["K"],C.cuboid(*coarse["cf_extents"]),coarse)
    assert len(capture.calls) == 2 and capture.calls[0] == capture.calls[1]
    assert all(i not in capture.calls[0]["prompt_native_ids"] for i in self_ids)
    for field in ("states", "state_observation", "mask_audit", "visible_correction", "ot_domains",
                  "frozen_visible_measurements", "joint_pose", "actual_pose", "keypoint_readout"):
        assert original[field] == perturbed[field], field
    np.testing.assert_array_equal(raw,np.array(frame["points"]["N3_SUBPIX"]))
    assert coarse == frame["initial_poses"]["N3_SUBPIX"]
    assert original["accepted"]
    joint = original["joint_pose"]
    domain_ids = [d["native_corner"] for d in original["ot_domains"]["domains"]]
    assert set(domain_ids) == {i for i,s in enumerate(original["states"]) if s in {"O","T"}}
    assert len(domain_ids) >= 2 and all(a["feasible"] for a in joint["joint_constraint_audit"])
    assert set(domain_ids) == {a["native_corner"] for a in joint["joint_constraint_audit"]}
    assert all(a["measurement_budget_pass"] and a["protection_pass"] for a in joint["visible_protection_audit"])
    for measurement in original["frozen_visible_measurements"]:
        native_id=measurement["native_corner"]
        preserved=next(m for m in joint["visible_observations"] if m["native_corner"]==native_id)
        assert all(preserved[k]==measurement[k] for k in ("xy","covariance_px2","protection_px"))
        assert original["keypoint_readout"]["reported_points"][native_id] == measurement["xy"]
    assert original["keypoint_readout"]["reported_points"][8] == raw[8].tolist()
    Q=np.array(coarse["R_cf"]).T @ np.array(coarse["R_physical"])
    np.testing.assert_allclose(original["actual_pose"]["R_physical"],np.array(original["actual_pose"]["R_cf"])@Q,atol=1e-12)
    rows=[dict(id=FRAME_ID+"/"+variant,source_record_id=FRAME_ID,variant=variant,
               mask_capture=capture.calls[index],runtime=value)
          for index,(variant,value) in enumerate(zip(protocol["variants"],[original,perturbed]))]
    C.save_rows(OUTPUT/"PREDICTIONS.jsonl.gz",rows)
    changed = [str(path) for path in SOURCES if C.sha(C.ROOT/path)!=source_bindings[str(path)]]
    if changed: raise RuntimeError("Executed sources changed before prediction seal: "+str(changed))
    C.write(OUTPUT/"PREDICTION_SEAL.json",dict(schema="STATEWISE_REAL_RUNTIME_SMOKE_GUARDED_SEAL_V2",before_scoring=True,
        score_invoked=False,GT_reads=0,rows=2,predictions_sha256=C.sha(OUTPUT/"PREDICTIONS.jsonl.gz"),
        protocol_sha256=C.sha(OUTPUT/"PROTOCOL.json"),source_sha256=source_bindings,
        state_model_sha256=C.sha(model_path),OT_model_sha256=C.sha(ot_path),elapsed_seconds=time.monotonic()-start))
    checks=dict(schema="STATEWISE_RUNTIME_POSTSEAL_GUARDED_CONTRACT_CHECKS_V2",
        prediction_seal_sha256=C.sha(OUTPUT/"PREDICTION_SEAL.json"),GT_reads=0,
        states=original["states"],O_T_domain_ids=domain_ids,S_ids=self_ids,
        all_joint_OT_constraints_pass=True,original_and_Sperturbed_masks_exactly_equal=True,
        both_Tiny_and_SAM_mask_sha256=capture.calls[0],
        Sraw_extreme_perturb_changes_state_BI_domains_pose=False,B_I_q_covariance_budget_frozen=True,
        ninth_center_metadata_preserved=True,physical_registration_Q_preserved=True,
        joint_accepted=original["accepted"],material_ambiguity=original["ambiguity_audit"]["material_ambiguity"],
        visible_protection_audit=joint.get("visible_protection_audit"),
        joint_constraint_audit=joint["joint_constraint_audit"],
        search=joint["search"],elapsed_seconds=time.monotonic()-start,
        limitation="seen CAL first-head smoke validates executable contract, not generalization or probabilistic coverage")
    C.write(OUTPUT/"CONTRACT_CHECKS.json",checks)
    print({k:checks[k] for k in ("states","O_T_domain_ids","S_ids","joint_accepted","material_ambiguity","elapsed_seconds")},flush=True)

if __name__=="__main__": main()
