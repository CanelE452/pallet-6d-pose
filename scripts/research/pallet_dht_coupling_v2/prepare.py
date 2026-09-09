"""Bind completed controls and revalidate immutable synthetic inputs for v2."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[3]
OLD = ROOT / 'data/pallet/results/pallet_dht_joint_v1'
SOURCE = ROOT / 'data/pallet/results/pallet_line_pose_v1/SOURCE_MANIFEST.json'
CONTROL_ARMS = ('point_only', 'hough_features', 'hough_joint')
NEW_ARMS = ('balanced', 'pcgrad', 'balanced_pcgrad', 'incidence')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def bound(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': sha(path)}


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.pending.json')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def check(condition, message):
    if not condition:
        raise ValueError(message)


def controls(out):
    protocol = read(OLD / 'TRAIN_PROTOCOL.json')
    check(protocol['source_manifest_sha256'] == sha(SOURCE), 'Synthetic manifest changed')
    check(protocol['initialization']['sha256'] == sha(protocol['initialization']['path']), 'R0 changed')
    inputs = {str(OLD / 'TRAIN_PROTOCOL.json'): sha(OLD / 'TRAIN_PROTOCOL.json')}
    for path, digest in protocol['source_code_sha256'].items():
        check(sha(path) == digest, f'Original training source changed: {path}')
        inputs[path] = digest
    rows, traces = [], {}
    for seed in (1, 2, 3):
        for arm in CONTROL_ARMS:
            name = f'{arm}_seed{seed}'
            train_dir, eval_dir = OLD / 'runs' / name, OLD / 'evaluation' / name
            trained, evaluated = read(train_dir / 'COMPLETION.json'), read(eval_dir / 'COMPLETION.json')
            check(trained['complete'] and trained['PASS'] and trained['stage'] == 'main', f'Unfinished control {name}')
            check(trained['optimizer_steps'] == 6998 and trained['epochs_completed'] == 2, 'Control budget changed')
            check(trained['train_frames'] == 55980 and trained['val_frames'] == 4020, 'Control data counts changed')
            check(trained['bindings']['protocol_sha256'] == inputs[str(OLD / 'TRAIN_PROTOCOL.json')], 'Control protocol differs')
            checkpoint = bound(trained['checkpoint'])
            check(checkpoint['sha256'] == trained['checkpoint_sha256'] == evaluated['checkpoint_sha256'], 'Control checkpoint changed')
            check(sha(train_dir / 'BATCH_TRACE.jsonl') == trained['batch_trace_sha256'], 'Control augmentation trace changed')
            check(sha(train_dir / 'history.json') == trained['history_sha256'], 'Control history changed')
            check(sha(train_dir / 'CELL_CONFIG.json') == trained['cell_config_file_sha256'], 'Control config changed')
            check(evaluated['complete'] and evaluated['PASS'] and evaluated['actual_positive_forwards'] == 319
                  and evaluated['actual_negative_forwards'] == 2689 and evaluated['baseline_candidate_copying'] is False,
                  'Control evaluation population/forward evidence differs')
            for path, digest in evaluated['source_sha256'].items():
                check(sha(path) == digest, f'Original evaluator source changed: {path}')
                inputs[path] = digest
            for path, digest in evaluated['output_sha256'].items():
                file = eval_dir / path
                check(sha(file) == digest, f'Control output changed: {file}')
                inputs[str(file)] = digest
            artifacts = {key: bound(path) for key, path in {
                'training_completion': train_dir / 'COMPLETION.json',
                'evaluation_completion': eval_dir / 'COMPLETION.json',
                'cell_config': train_dir / 'CELL_CONFIG.json',
                'history': train_dir / 'history.json',
                'batch_trace': train_dir / 'BATCH_TRACE.jsonl',
                'initial_state': train_dir / 'INITIAL_STATE.pt',
                'results': eval_dir / 'RESULTS.json',
                'predictions': eval_dir / 'PREDICTIONS.json',
            }.items()}
            for item in [checkpoint, *artifacts.values()]:
                inputs[item['path']] = item['sha256']
            rows.append(dict(arm=arm, seed=seed, name=name, run_dir=str(OLD),
                             training_dir=str(train_dir), evaluation_dir=str(eval_dir), checkpoint=checkpoint,
                             artifacts=artifacts, reused=True, new_v2_accuracy_forwards=0))
            traces.setdefault(str(seed), []).append(trained['batch_trace_sha256'])
    check(all(len(set(value)) == 1 for value in traces.values()), 'Original matched-seed traces differ')
    for filename in ('VIEW_STRATA.json', 'VIEW_DIAGNOSTIC_PROTOCOL.json'):
        inputs[str(OLD / filename)] = sha(OLD / filename)
    result = dict(schema='pallet_dht_coupling_reused_controls_v2', complete=True, PASS=True,
                  control_run_dir=str(OLD), control_protocol=bound(OLD / 'TRAIN_PROTOCOL.json'),
                  controls=rows, n_control_cells=9, input_sha256=inputs,
                  original_trace_sha256_by_seed={seed: values[0] for seed, values in traces.items()},
                  role='Previously completed same-budget controls, explicitly reused; not new v2 inference',
                  evaluation_population=dict(positive=319, negative=2689, sessions=13, role='reused DEV', held_out_final=False))
    destination = out / 'REUSED_CONTROLS.json'
    if destination.exists():
        check(read(destination) == result, 'Reused control binding changed')
    else:
        write(destination, result)
    print(f'Bound {len(rows)} exact completed controls, {len(inputs)} artifact/source hashes', flush=True)


def sources(out):
    protocol = read(OLD / 'TRAIN_PROTOCOL.json')
    check(sha(SOURCE) == protocol['source_manifest_sha256'], 'Source manifest changed')
    records = read(SOURCE)['records']
    check(len(records) == 60000, 'Source total differs')
    start = time.time()
    def verify(row):
        for field in ('image', 'label'):
            check(sha(row[field]) == row[field + '_sha256'], f'Source changed: {row[field]}')
        return row['index']
    with ThreadPoolExecutor(max_workers=4) as pool:
        for count, _ in enumerate(pool.map(verify, records), 1):
            if count % 10000 == 0:
                print(f'Actual source rehash {count}/60000 image+label pairs', flush=True)
    write(out / 'SOURCE_REVALIDATION.json', dict(complete=True, PASS=True,
          created_at_utc=datetime.now(timezone.utc).isoformat(), source_manifest=bound(SOURCE),
          checked_images=60000, checked_labels=60000, train=55980, validation=4020,
          filter_applied=False, elapsed_seconds=time.time()-start, source_sha256={str(Path(__file__)): sha(__file__)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--phase', choices=['controls', 'sources'], required=True)
    args = parser.parse_args()
    out = args.run_dir.resolve()
    check((out / 'PURPOSE.md').exists(), 'Declared result purpose required')
    {'controls': controls, 'sources': sources}[args.phase](out)
