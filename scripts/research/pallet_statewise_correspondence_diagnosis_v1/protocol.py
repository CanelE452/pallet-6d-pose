"""Freeze source/input identity and the six-CAL cost panel before any solve."""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import argparse
import time

from scripts.research.pallet_minimal_bi_ablation_v1.batch import pilot_ids
from .common import ARMS, digest, read, sha, write, record_stage


def freeze(source_root, guarded_dir, out):
    started = time.monotonic()
    source_root, guarded_dir, out = map(Path, (source_root, guarded_dir, out))
    frames = read(guarded_dir/'e_inputs/INPUTS.json')['frames']
    assert len(frames) == 238 and len({f['id'] for f in frames}) == 238
    assert Counter(f['partition'] for f in frames) == {'CAL':46,'TEST':49,'REMAINDER':143}
    bindings = []
    for index, frame in enumerate(frames):
        image = Path(frame['image'])
        image_digest = sha(image)
        assert image_digest == frame['image_sha256'], frame['id']
        files = {kind:guarded_dir/f'e_run/{kind}/{index:03d}_E_FROZEN.json'
                 for kind in ('observations','poses')}
        values = {k:read(p) for k,p in files.items()}
        assert all(v['id'] == frame['id'] for v in values.values())
        bindings.append(dict(id=frame['id'], index=index, partition=frame['partition'],
            session=frame['session'], cluster_id=frame['cluster_id'], raw_hw=frame['raw_hw'],
            RGB_sha256=image_digest, K=frame['K'], xyz=frame['xyz'],
            E0=frame['arms']['E_FROZEN']['coarse_pose'],
            native_vertex_mapping=frame['arms']['E_FROZEN']['native_vertex_mapping'],
            observation_sha256=sha(files['observations']), cached_pose_sha256=sha(files['poses'])))
    source_files = {}
    folders = ['pallet_statewise_correspondence_diagnosis_v1','pallet_minimal_bi_ablation_v1',
               'pallet_guarded_bi_final_v1','pallet_visibility_curriculum_20261013_v1',
               'pallet_posefix_gap_audit_v1','pallet_sensors_submission_v1']
    for name in folders:
        for path in sorted((source_root/'scripts/research'/name).glob('*.py')):
            source_files[str(path.relative_to(source_root))] = sha(path)
    manifest = dict(schema='STATEWISE_DIAGNOSTIC_INPUT_CONTRACT_V1',
        created_UTC=datetime.now(timezone.utc).isoformat(), base_main='96360c61418c435602875f5f34ca8ea4c694fbdc',
        instruction_sha256=sha(source_root/'_docs/experiments/pallet_statewise_correspondence_diagnosis_v1/USER_TASK.md'),
        ordered_ids=[f['id'] for f in frames], rows=len(frames), frames=bindings,
        source_sha256=source_files, input_sha256=sha(guarded_dir/'e_inputs/INPUTS.json'),
        GT_inference_inputs=False, reference_use='SEPARATE_POSTSEAL_SCORING_PROCESS_ONLY',
        distortion='unchanged existing image/K/zero-distortion convention; no new undistortion',
        units='metres; pixel observations; physical registered C2 only; no 90deg equivalence',
        model_origin='existing cuboid centre',
        semantic_owner_status='SEMANTIC_OWNER_UNVERIFIED unless independent evidence exists')
    protocol = dict(schema='STATEWISE_FIXED_FOUR_PATH_REPLAY_V1',
        ordered_ids=manifest['ordered_ids'], pilot_ids=pilot_ids(frames),
        pilot_rule='sorted CAL session round robin; SHA256(ID) ascending within session; no reference access',
        arms=list(ARMS), same_packet_pairs=[['REOBS_MULTI','REOBS_SINGLE'],['TYPED_MULTI','TYPED_SINGLE']],
        REOBS_MULTI='exact RT.solve with actual cached E-after B/I observations and original budgets',
        REOBS_SINGLE='one SLSQP from E0; same original BI soft_l1/covariances/constraints; fixed E0 native registration',
        TYPED_MULTI='exact original G10 family search, G13 typed refinement where active, frozen G10 family selection; logging only instrumentation',
        TYPED_SINGLE='same selected-family original backend inner objective/constraints from one E0 start; preserve original prior reference',
        typed_outer_policy='NOT_ISOLATED if G10 family score differs from G13 inner objective; report all differences',
        typed_union_branches=dict(actual_T_domains=0, actual_branch_histogram={'1':238},
                                  policy='do not alter union feasible set'),
        parity=dict(atol=1e-9,rtol=0.,exact=['states','IDs','source','selected_family','accepted','fallback'],
                    relax_after_result=False),
        costs=dict(max_top_level_slots=952,max_logging_replays=238,max_cpu_stage_elapsed_seconds=7200,
                   max_gpu_process_elapsed_seconds=3600,max_heavy_workers=3,max_gpu_processes=1),
        fixed_comparisons=['TYPED_SINGLE-E0','TYPED_SINGLE-TYPED_MULTI','TYPED_SINGLE-REOBS_SINGLE'],
        bootstrap=dict(replicates=10000,seed=20260917,unit='actual original recording cluster_id',
                       paired=True,independent_validation=False),
        no_new_training=True,no_new_data=True,no_new_seed=True,no_threshold_or_loss_tuning=True,
        unavailable=dict(binary='failure in full denominator',continuous='NA; available and common available N reported'),
        success='translation_cm <= 5 AND physical_C2_rotation_deg <= 5',
        GT_inputs=False, source_manifest_sha256=digest(manifest))
    write(out/'SOURCE_AND_INPUT_MANIFEST.json',manifest)
    write(out/'REPLAY_PROTOCOL.json',protocol)
    record_stage(out,'identity_freeze',started,frames=238,new_pose_calls=0,worker_cpu_seconds=None)
    return dict(pilot_ids=protocol['pilot_ids'],manifest_sha256=sha(out/'SOURCE_AND_INPUT_MANIFEST.json'),
                protocol_sha256=sha(out/'REPLAY_PROTOCOL.json'))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--source-root',required=True); p.add_argument('--guarded-dir',required=True)
    p.add_argument('--output',required=True)
    a = p.parse_args()
    print(freeze(a.source_root,a.guarded_dir,a.output))
