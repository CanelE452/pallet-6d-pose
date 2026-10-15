"""Freeze predictor bindings and the two mechanistic hypotheses before runs."""
from pathlib import Path
from datetime import datetime,timezone
import argparse
from scripts.research.pallet_minimal_bi_ablation_v1 import batch as B
from scripts.research.pallet_minimal_bi_ablation_v1 import runtime as RT
from .score import SELECTION_RULE

def freeze(folder,prior):
    folder,prior=Path(folder),Path(prior);code=Path(__file__).resolve().parents[3]
    old=B.read(prior/'run_v2/PROTOCOL.json')
    source=dict(old['source_sha256']);source.update(old['legacy_source_sha256'])
    # Only inference definitions are frozen here. Reporting/checker code cannot
    # alter the already frozen method or select a new rule after seeing PRIMARY.
    for name in ('__init__.py','e_inputs.py','guard.py','run.py','freeze.py','score.py'):
        p=Path(__file__).with_name(name);source[str(p.relative_to(code))]=RT.sha256(p)
    for name,h in source.items():
        if RT.sha256(code/name)!=h:raise ValueError('Inherited source changed: '+name)
    inputs={str(p):RT.sha256(p) for p in (
        prior/'inputs_v2/INPUTS.json',prior/'inputs_v2/INPUT_SEAL.json',
        prior/'run_v2/PROTOCOL.json',prior/'run_v2/PREDICTION_SEAL.json',prior/'run_v2/PREDICTIONS.jsonl.gz',
        prior/'run_v2/OBSERVATION_SEAL.json',folder/'e_inputs/INPUTS.json',folder/'e_inputs/INPUT_SEAL.json')}
    protocol=dict(schema='GUARDED_BI_BOUNDED_FINAL_METHOD_TRIAL_V1',UTC=datetime.now(timezone.utc).isoformat(),
        parent_commit='7ea3899ea2ac2e26619df81924a49d11237673e4',
        algorithms=['A','B'],initializers=['RAW_YOLO_own_F','N3_SUBPIX_own_F','SAME_FROZEN_E'],
        A=dict(minimum_unique_BI=4,visible_rank=6,require_accepted=True,
            reject_ambiguity_detected=True,reject_local_underdetermined=True,reject_two_BI_pose_family=True,
            genuine_RGB_moved_BI_minimum=1,moved_tolerance_px=1e-8,failure='return_exact_own_baseline'),
        B=dict(includes='A',evidence='own supported real observed ray points and their saved confidence',
            projected_edge='same native corner to saved native neighbor; infinite 2D line',
            loss='confidence-weighted Huber perpendicular distance',huber_delta_px=1.25,
            each_ray='candidate_cost<=initial_cost+1e-9',
            aggregate='initial_cost-candidate_cost>1e-9',minimum_valid_rays=1,
            reused_fitting_evidence=True,independent_image_holdout=False,failure='return_exact_own_baseline'),
        selection_rule=SELECTION_RULE,selection_before_new_PRIMARY_scores=True,
        no_more_candidates_or_threshold_tuning_after_CAL=True,
        success='translation_cm<=5 AND rotation_deg<=5',historical_success='strict <5 and <5',
        populations=dict(CAL=46,TEST=49,REMAINDER=143,PRIMARY=192),all_rows_and_failures_retained=True,
        E_points='old N3 real observations remapped by unique physical-vertex bijection, not E projections',
        E_state_masks_BI='fresh own E geometry, prompts, state and mask forwards',
        S='excluded before all new state/mask-prompt/local-BI/PnP image observations; final model projection only',
        O_T_U='abstain from new BI-only exact 2D correspondences; no new region evidence in this follow-up',
        ambiguity_rejection_is_global_uniqueness_proof=False,
        labels_or_defaults_or_paper_changed=False,new_training_or_learning_seeds_or_data=False,
        prior_PRIMARY_results_already_development_exposed=True,new_independent_test=False,
        no_automatic_operating_default_change=True,
        caps=dict(gpu_process_seconds=3600,cpu_solver_analysis_seconds=7200,pose_workers=3,threads=1),
        immutable_inputs_sha256=inputs,inference_source_sha256=source)
    B.write(folder/'METHOD_PROTOCOL.json',protocol)
    print(RT.sha256(folder/'METHOD_PROTOCOL.json'),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--folder',type=Path,required=True);p.add_argument('--prior',type=Path,required=True)
    a=p.parse_args();freeze(a.folder,a.prior)
