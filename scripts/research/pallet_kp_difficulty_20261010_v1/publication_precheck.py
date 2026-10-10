"""Immutable-input publication ledger and noncircular public manifest."""
import argparse
from datetime import datetime,timezone
from pathlib import Path
import subprocess
from ..pallet_observation_refiner_20261009_v1 import common as C

DOC=C.WORKTREE/'_docs/experiments/pallet_kp_difficulty_20261010_v1'
PRIVATE=Path('/dev/shm/pallet-kp-difficulty-private-20261010')
PUBLICATION_REPO=Path('/dev/shm/pallet-kp-publication-20261010')

def precheck():
    assert not (DOC/'BUILD_LEDGER.json').exists()
    old=C.read(DOC/'PRIOR_PUBLICATION_BINDINGS.json')
    for b in old['protected_files']:
        p=C.WORKTREE/b['path'];assert C.sha(p)==b['sha256'] and p.stat().st_size==b['bytes']
    source=C.source_state();assert source==C.read(PRIVATE/'SOURCE_STATE_BEFORE_PUBLICATION.json')
    retention=C.read(C.DOC/'PRIVATE_CHECKPOINT_RETENTION.json')
    for b in retention['retained']:assert C.sha(C.ROOT/b['path'])==b['sha256']
    runtime=C.read(DOC/'RUNTIME.json');assert runtime['complete'] and runtime['execution']['pipeline_calls_complete']==600
    causal=C.read(DOC/'CAUSAL_POSE_EXECUTION.json');pilot=C.read(DOC/'CPU_PILOT.json')
    source_counts=C.read(DOC/'SOURCE_DIAGNOSTIC_EXECUTION_COUNTS.json')
    ledger=dict(schema='kp_difficulty_actual_followup_build_v1',complete=True,
        started_utc=old['started_utc'],completed_before_publication_utc=datetime.now(timezone.utc).isoformat(),
        baseline_publication_commit=old['baseline_commit'],protected_original_public_files=111,
        protected_original_all_unchanged=True,source_tracked_state_unchanged=True,retained_checkpoints=retention['retained'],
        original_frozen_training_updates=9000,new_training_updates=0,new_RGB=0,new_manual_annotations=0,
        accuracy=dict(new_pose_paths=2552,unlocked_actual_replay_paths=1276,initial_pose_reuse=True,
            new_detector_forwards=0,new_head_forwards_real=0,counts=causal['counts'],
            geometry_sealed_before_GT=True,unlocked_coordinate_pose_and_metrics_max_difference=0.,
            initial_pose_reuse_not_latency_measurement=True,receipt=C.binding(DOC/'CAUSAL_POSE_EXECUTION.json')),
        pilot=dict(new_pose_paths=104,unlocked_actual_replay_paths=104,counts=pilot['counts'],GT_scoring=0),
        source=source_counts,
        fresh_actual_whole_path_runtime=dict(pipeline_calls=600,measured=520,warmup=80,
            model_forwards=runtime['model_forwards'],primitive_and_pipeline_counts=runtime['execution'],
            cached_coordinates_used_for_latency=False,other_numeric_workloads_stopped=True,
            interference_snapshots=len(runtime['environment']['interference_snapshots']),receipt=C.binding(DOC/'RUNTIME.json')),
        root_review=dict(derived_PNG_files=3,real_image_cases=4,panel_count=12,
            existing_RGB_decode_exposures=4,new_inference_pose_fits_rays=0,statistical_selection_posthoc=True),
        documented_repairs=[dict(stage='CPU pilot panel selection',error='panel entries are dicts, not hashable frame IDs',
            before_numeric_calls=True,additional_pose_detector_head_rays=0,fix='extract entry.id; unchanged fixed26 panel'),
            dict(stage='source no-ray geometry final stdout',error='JSON stdout serialization after completed128-row writes',
                 numerical_outputs_altered=False,repeated_geometry=False,fix='final stdout integer serialization; code binding refreshed'),
            dict(stage='source logit verifier binding',error='semantic row hash was initially compared to compressed-file hash',
                 source_data_changed=False,first_failed_attempt_query_replay_calls=0,fix='validate semantic row and file hash separately'),
            dict(stage='report read-only review',fix='clarified BASE-based case selection; exact586 old pose omissions; actual CLI guards and source/runtime keys',
                 performance_configuration_changed=False)],
        no_performance_based_setting_seed_model_reselection=True,
        missing_work=['no revised-supervision training','no new independent holdout validation',
            'original E6 four controlled RGB variants remain unexecuted','original fixed controls586 R/t remain unstored'],
        scientific_verdict='both fixed primary and secondary failed N3_SUBPIX joint operational mean improvement',
        old_experiment_execution_not_counted_as_new=True)
    C.write(DOC/'BUILD_LEDGER.json',ledger)
    snapshot=DOC/'NUMERICAL_PREPUBLICATION_CHECKS.json'
    assert not snapshot.exists()
    snapshot.write_bytes((DOC/'REVIEW_CHECKS.json').read_bytes())
    remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/main','refs/heads/research/observation-refiner-robust-pnp-20261009'],cwd=PUBLICATION_REPO,text=True)
    refs={line.split()[1]:line.split()[0] for line in remote.splitlines()}
    assert refs['refs/heads/main']=='7e92fdcefb0a37bdee0aef95d3e86c572e23b967'
    assert refs['refs/heads/research/observation-refiner-robust-pnp-20261009']==old['baseline_commit']
    C.write(DOC/'PUBLICATION_PRECHECK.json',dict(passed=True,generated_utc=datetime.now(timezone.utc).isoformat(),
        baseline_commit=old['baseline_commit'],remote_before=refs,source_state=source,
        protected_original111_unchanged=True,retained_checkpoints_unchanged=True,
        branch='research/observation-refiner-robust-pnp-20261009',main_unchanged=True,force_push=False,
        publication_repository=str(PUBLICATION_REPO),
        storage='Independent shared tmpfs clone for normal commit/push: SSD free space fell to4.8MB during work. Old objects readonly via alternates; new objects and outputs in tmpfs. No user data cleanup.',
        source_original_repository=str(C.ROOT),source_files_not_edited=True,
        numerical_checks=C.binding(snapshot),numerical_checks_note='Immutable prepublication numerical receipt; final REVIEW_CHECKS is regenerable.'))
    print('PUBLICATION_PRECHECK_PASS')

def manifest():
    code=C.WORKTREE/'scripts/research/pallet_kp_difficulty_20261010_v1'
    excluded={'REVIEW_MANIFEST.json','REVIEW_CHECKS.json','PUBLICATION.json'}
    paths=[p for p in DOC.rglob('*') if p.is_file() and p.name not in excluded]
    paths +=list(code.glob('*.py'))
    assert not any('__pycache__' in str(p) for p in paths)
    C.write(DOC/'REVIEW_MANIFEST.json',dict(schema='kp_difficulty_public_review_manifest_v1',complete=True,
        original_protected_files=111,files=[C.binding(p) for p in sorted(paths)],
        exclusions=dict(REVIEW_MANIFEST='self hash cycle',REVIEW_CHECKS='regenerable verifier receipt',
            PUBLICATION='post-push receipt records immutable payload commit; cannot include own future commit hash'),
        no_raw_original_RGB_or_weights=True))
    print('MANIFEST',len(paths))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['precheck','manifest'])
    a=p.parse_args();precheck() if a.stage=='precheck' else manifest()
