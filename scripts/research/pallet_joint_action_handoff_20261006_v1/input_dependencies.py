"""Bind actual A supervision, normalization, baseline and real-feature inputs.

The 73.7 GB source feature arrays have their separate full-content receipt.
This small supplemental receipt covers inputs absent from A_protocol.json;
it does not claim independent accuracy for reconstructed DEV references.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import json
import time
from .run import ROOT, DOC, CODE, read, sha, write


def generate(source_root):
    source = Path(source_root).resolve()
    path = DOC / 'INPUT_DEPENDENCIES.json'
    if path.exists():
        old = read(path)
        if old['source_root'] != str(source):
            raise ValueError('Input source root differs from locked receipt')
        for item in old['inputs']:
            if sha(item['path']) != item['sha256']:
                raise ValueError('Input dependency changed: ' + item['path'])
        return {'status': 'REUSED_VALIDATED', 'inputs_verified': len(old['inputs'])}
    started = time.perf_counter()
    paths = {}
    def add(p, role):
        p = Path(p).resolve()
        paths.setdefault(p, set()).add(role)
    for name in ('DIM_NORMALIZATION_LOCK.json', 'TRAIN_PROTOCOL_LOCK.json',
                 'CALIBRATION_AND_SELECTION.json'):
        add(source / '_docs/experiments/pallet_dim_conditioned_p_v1' / name, 'inherited input/selection contract')
    add(source / 'data/pallet/results/pallet_posefix_replay_diagnosis_v1/TARGETS.json', 'shared supervised evaluation targets')
    for seed in (1, 2, 3):
        add(source / f'data/pallet/results/pallet_final_ml_contribution_test_v1/B_line_vs_point/order_seed{seed}.npy', 'identical paired-arm training order')
        for split, prefix in (('SYNTH_HELDOUT', 'heldout'), ('REAL_DEV', 'REAL_DEV')):
            add(source / f'data/pallet/results/pallet_dim_conditioned_p_v1/predictions/{split}/N3_DIM_SYM_seed{seed}.json', 'unchanged N3 comparator predictions')
            add(source / f'data/pallet/results/pallet_posefix_replay_diagnosis_v1/predictions/seed{seed}_{prefix}.npz', 'uncapped RAW first-pass PoseFix comparator; array points[:,0,1]')
    axis = read(source / 'data/pallet/results/paper_pose_metric_closure_v1/AXIS_REVIEW_MANIFEST.json')
    for frame in axis['frames_list']:
        add(source / f'data/pallet/results/pallet_dim_conditioned_p_v1/DEV_cache/{frame["frame_id"]}.pt', 'original frozen real detector/features and metadata')
        add(source / frame['annotation'], 'actual real camera intrinsics read by evaluator')
    for rel in ('scripts/research/pallet_dim_conditioned_p_v1/pose.py',
                'scripts/research/pallet_dim_conditioned_p_v1/eval_math.py',
                'scripts/research/pallet_dim_conditioned_p_v1/refiner.py',
                'scripts/research/pallet_dim_conditioned_p_v1/dcp_env.py',
                'scripts/research/pallet_final_ml_contribution_test_v1/generic_point_refiner.py'):
        add(ROOT / rel, 'inherited executable adapter in latest-remote worktree')
    add(Path(__file__), 'receipt generator')
    entries = [{'path': str(p), 'relative_to_source': str(p.relative_to(source)) if p.is_relative_to(source) else None,
                'roles': sorted(roles), 'sha256': sha(p), 'bytes': p.stat().st_size}
               for p, roles in sorted(paths.items())]
    result = {'schema': 'joint_action_additional_input_dependencies_v1', 'status': 'DONE',
              'source_root': str(source), 'inputs': entries, 'input_count': len(entries),
              'bytes_hashed': sum(e['bytes'] for e in entries), 'seconds': time.perf_counter() - started,
              'source_feature_full_content_receipt': 'SOURCE_CACHE_HASHES.json',
              'scope': 'Supplement A_protocol and component receipts; inputs are read-only, caches/checkpoints remain external; no raw data copied',
              'independent_physical_reference': False}
    write(path, result)
    return {'status': 'DONE', 'inputs_hashed': len(entries), 'bytes_hashed': result['bytes_hashed'], 'seconds': result['seconds']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(generate(args.source_root), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
