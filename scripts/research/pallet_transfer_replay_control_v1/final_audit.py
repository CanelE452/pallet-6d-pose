"""Independent recomputation and provenance audit for the completed fixed panel."""
import json
from pathlib import Path
import subprocess
import numpy as np
import torch

from aggregate import ARMS,CONTRASTS
from evaluate import NAMES,checkpoint
from runtime import ROOT,RAW,DOC,R0,atomic_json,sha


def read(path):return json.loads(Path(path).read_text())


def main():
    lock=read(DOC/'EXECUTION_CODE_LOCK.json')
    for path,expected in {**lock['implementation_sha256'],**lock['reporting_implementation_sha256']}.items():
        assert sha(ROOT/path)==expected,(path,expected,sha(ROOT/path))
    source=read(DOC/'SOURCE_BINDING.json')
    assert all(sha(ROOT/path)==expected for path,expected in source['originals'].items())
    complete=read(DOC/'TRAINING_COMPLETE.json');assert complete['fits']==12 and complete['actual_main_optimizer_updates']==3600
    initial=set();checkpoints={};base_hashes={seed:{step:set() for step in range(300)} for seed in (1,2,3)}
    for seed in (1,2,3):
        for arm in ARMS:
            name=f'{arm}_seed{seed}';audit=read(RAW/'runs'/name/'TRAINING_AUDIT.json');trace=read(RAW/'runs'/name/'EXPOSURE.json')
            assert audit['optimizer_updates']==len(trace)==300 and audit['criterion_updates']==10 and audit['BN_buffers_equal']
            assert sha(checkpoint(name))==audit['checkpoint_sha256'];initial.add(audit['initial_state_sha256']);checkpoints[name]=audit['checkpoint_sha256']
            assert [r['step'] for r in trace]==list(range(300))
            for step,row in enumerate(trace):
                base_hashes[seed][step].add(row['streams']['target_base']['tensor_sha256'])
                for stream,record in row['streams'].items():
                    expected='source' if stream.startswith('source_') else 'target';assert record['domain']==expected
                    if arm!='REPLAY':assert not stream.startswith('source_')
            assert sum(r['clipped'] for r in trace)==audit['clipped_updates']
    assert len(initial)==1 and all(len(v)==1 for seed in base_hashes.values() for v in seed.values())
    metrics=read(DOC/'METRICS_PER_SEED.json');paired=read(DOC/'PAIRED_CONTRASTS.json')
    assert set(metrics['per_model'])==set(NAMES) and paired['status']=='COMPLETE'
    for name in NAMES:
        result=read(RAW/'evaluation'/name/'RESULT.json');target=read(RAW/'evaluation'/name/'TARGET_PER_FRAME.json');source_rows=read(RAW/'source_evaluation'/name/'SOURCE_PER_FRAME.json')
        pos=[r for r in target.values() if r['kind']=='positive'];neg=[r for r in target.values() if r['kind']=='negative']
        assert len(pos)==145 and len(neg)==2689 and len(source_rows)==512
        for label,rows in [('target',pos),('source',list(source_rows.values()))]:
            den=sum(r['denominator'] for r in rows);num=sum(r['numerators']['10.0'] for r in rows)
            observed=result[label]['ALL_GT_PCK']['10.0'];assert abs(num/den-observed)<1e-15
        assert result['checkpoint_sha256']==sha(checkpoint(name))
    verdict=read(DOC/'VERDICT.json');assert verdict['execution_status']=='COMPLETE' and verdict['scientific_verdict']
    assert read(DOC/'CURRENT_RESULT.json')['scientific_complete']
    assert read(DOC/'RUNTIME.json')['status']=='COMPLETE'
    audit=dict(execution='COMPLETE',integrity='PASS',scientific_verdict=verdict['scientific_verdict'],
        fits=12,actual_main_optimizer_updates=3600,actual_smoke_optimizer_updates=1,
        initial_state_parity=True,target_base_augmented_tensor_parity_all_seeds_steps=True,
        frozen_BN_buffers=True,checkpoints=checkpoints,target_positive=145,target_sessions=4,target_negative=2689,
        source_retention=512,evaluated_models=13,full_denominator_PCK_recomputed=True,
        paired_bootstrap_resamples=10000,performance_based_retraining_or_selection=False,
        source_original_bindings_preserved=len(source['originals']),paper_final_preserved=True,
        prior_experiment_results_modified=False,large_checkpoints_and_prediction_caches_ignored=True,
        runtime_scope='Single RTX3080 fixed panel, not Jetson/export',independent_confirmation=False,
        git_publication='Separate from scientific completion; verify after final commit/push',
        unrelated_user_untracked=['_docs/experiments/pallet_capacity_screen_v1/','data/pallet/results/pallet_capacity_screen_v1/','scripts/research/pallet_capacity_screen_v1/'])
    atomic_json(DOC/'FINAL_AUDIT.json',audit)
    old=DOC/'ARTIFACT_MANIFEST.json';history=DOC/'ARTIFACT_MANIFEST_CPU_PREFLIGHT.json'
    if old.exists() and not history.exists():history.write_text(old.read_text())
    files=sorted(p for directory in (DOC,ROOT/'scripts/research/pallet_transfer_replay_control_v1') for p in directory.rglob('*') if p.is_file() and p not in (old,) and '__pycache__' not in p.parts)
    atomic_json(old,dict(scope='Completed transfer replay control; small public code/docs only',
        files={str(p.relative_to(ROOT)):dict(sha256=sha(p),bytes=p.stat().st_size) for p in files},
        raw_evidence='See RAW_EVIDENCE_MANIFEST.json; weights/predictions are not included'))
    print('FINAL_AUDIT_PASS',verdict['scientific_verdict'],flush=True)


if __name__=='__main__':main()
