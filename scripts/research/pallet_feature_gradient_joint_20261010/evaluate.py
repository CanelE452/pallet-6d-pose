"""One actual frozen-F call per new sealed coordinate row; frozen baselines reused.

This entrypoint may read references only AFTER the entire inference coordinate
file and its configuration/source hashes have been sealed by coordinates.py.
"""
from collections import Counter
import copy
import gzip
import json
import time

import cv2
import numpy as np
import torch

from . import common as C


def main():
    start = time.monotonic()
    seal = C.read(C.DOC / 'COORDINATES_SEAL.json')
    assert seal['status'] == 'PASS'
    assert C.sha(C.DOC / 'NEW_COORDINATES_SEALED.jsonl.gz') == seal['coordinates_sha256']
    assert C.sha(C.DOC / 'FUSION_METHOD_LOCK.json') == seal['fusion_method_lock_sha256']
    assert C.sha(C.DOC / 'INPUT_LOCK.json') == seal['input_lock_sha256']
    C.verify_bindings(seal['source_bindings'])
    assert not (C.DOC / 'PREDICTIONS.jsonl.gz').exists(), 'Preserve existing/partial evaluation'
    inputs = C.read(C.DOC / 'INPUT_LOCK.json')
    manifest = {r['id']: r for r in inputs['input_manifest']}
    baseline_path = C.BASELINE_DOC / 'PREDICTIONS.jsonl.gz'
    baseline_sha = C.sha(baseline_path)
    frozen_rows = list(C.rows(baseline_path))
    assert len(frozen_rows) == 3828
    old = {(r['seed'], r['method'], r['id']): r for r in frozen_rows}
    for r in frozen_rows:
        b = manifest[r['id']]
        assert C.digest(r['q0']) == C.digest(b['q0'])
        assert r['prediction_support'] == b['prediction_support']
        assert r['raw_hw'] == b['raw_hw']
    cv2.setNumThreads(1)
    torch.set_num_threads(4)
    # Existing adapter resolves original private input paths and authenticates
    # the immutable a22 scorer. Neither module's experiment main() is invoked.
    from scripts.research.pallet_n3_subpix_final_20261010 import common as prior
    prior.ROOT = C.SOURCE
    load_real, _, _ = prior.existing()
    E, frames, targets, _, _ = load_real()
    from scripts.research.pallet_n3_subpix_final_20261010.evaluate import score
    ids = inputs['population']['frame_ids']
    assert [f['id'] for f in frames] == ids
    frames_by_id = {f['id']: f for f in frames}
    # Evaluation-only geometric references are saved after coordinate sealing,
    # enabling independent R/t-to-error arithmetic without private RGB/weights.
    C.write(C.DOC / 'EVALUATION_POSE_REFERENCE.json', dict(
        reference_type='existing geometric proxy reconstructed from 2D and registry dimensions; not physical metrology',
        used_in_inference=False, coordinates_already_sealed_sha256=seal['coordinates_sha256'],
        source_bindings=[dict(path=relative, sha256=C.sha(C.SOURCE / relative)) for relative in (
            'data/pallet/results/paper_pose_metric_closure_v1/AXIS_REVIEW_MANIFEST.json',
            'data/pallet/results/paper_pose_metric_closure_v1/GEOMETRY_RESOLVED_POSE_GT.json')],
        records=[dict(id=f['id'], R=f['truth']['R'], t=f['truth']['t'], xyz=f['truth']['xyz'],
                      order=f['truth']['order']) for f in frames]))
    coordinates = list(C.rows(C.DOC / 'NEW_COORDINATES_SEALED.jsonl.gz'))
    assert len(coordinates) == 1914
    assert Counter(r['method'] for r in coordinates) == {m: 957 for m in C.NEW_METHODS}
    calls = 0
    pnp = Counter()
    failures = []
    new_rows = []
    actual_F_seconds = 0.
    with gzip.open(C.DOC / 'PREDICTIONS.jsonl.gz', 'wt', encoding='utf-8', compresslevel=6) as stream:
        for row in frozen_rows:
            row = copy.deepcopy(row)
            row['current_experiment_F_attempt'] = False
            row['frozen_baseline_reused'] = True
            row['frozen_baseline_source_sha256'] = baseline_sha
            stream.write(json.dumps(C.finite(row), separators=(',', ':'), allow_nan=False) + '\n')
        for i, coordinate in enumerate(coordinates):
            fid, seed, method = coordinate['id'], coordinate['seed'], coordinate['method']
            frame = frames_by_id[fid]
            bound = manifest[fid]
            support = np.asarray(bound['prediction_support'], bool)
            q0 = np.asarray(bound['q0'], float)
            points = np.asarray(coordinate['qFinal'], float)
            assert np.array_equal(frame['q'], q0, equal_nan=True)
            assert np.array_equal(points[8], q0[8], equal_nan=True)
            assert np.array_equal(points[~support], q0[~support], equal_nan=True)
            metadata = frame['captured']['captured']
            metadata_digest = C.digest({k: v for k, v in metadata.items() if k not in ('p3', 'p4')})
            before = time.monotonic()
            row = score(E, frame, points, targets[fid], method, seed)
            actual_F_seconds += time.monotonic() - before
            calls += 1
            pnp.update(row['PnP_counts'])
            assert C.digest({k: v for k, v in metadata.items() if k not in ('p3', 'p4')}) == metadata_digest
            base = old[(seed, 'N3_DIM_SYM', fid)]
            row.update(grade=base['grade'], q0=coordinate['q0'], qN=coordinate['qN'],
                qS=coordinate['qS'], qFinal=coordinate['qFinal'],
                prediction_support=support, raw_hw=coordinate['raw_hw'],
                correction=coordinate['correction'], fixed_metadata=copy.deepcopy(base['fixed_metadata']),
                coordinate_provenance='inference-only coordinates sealed before loading targets',
                coordinate_file_sha256=seal['coordinates_sha256'],
                qS_semantics='unconstrained joint solve or explicit N3 fallback; never cornerSubPix',
                current_experiment_F_attempt=True, frozen_baseline_reused=False)
            assert row['fixed_metadata']['selected_index'] == metadata['selected_index']
            assert C.digest(row['fixed_metadata']['K']) == C.digest(frame['K'])
            assert C.digest(row['fixed_metadata']['dimensions_pnp_WH_D_m']) == C.digest(frame['xyz'])
            if not row['pose']['available']:
                failures.append({'id': fid, 'seed': seed, 'method': method, 'status': 'no_pose'})
            clean = C.finite(row)
            new_rows.append(clean)
            stream.write(json.dumps(clean, separators=(',', ':'), allow_nan=False) + '\n')
            if i % 160 == 0 or i == 1913:
                stream.flush()
                C.write(C.DOC / 'EXECUTION_LEDGER.json', dict(status='RUNNING_EVALUATION', new_F_calls=calls,
                    PnP_counts=dict(pnp), new_rows=len(new_rows), elapsed_seconds=time.monotonic() - start))
                print('NEW_F', calls, '/1914', 'seconds', round(time.monotonic() - start, 2), flush=True)
    assert calls == len(new_rows) == 1914
    for seed in (1, 2, 3):
        for method in C.NEW_METHODS:
            rr = [r for r in new_rows if r['seed'] == seed and r['method'] == method]
            assert len(rr) == 319 and {r['id'] for r in rr} == set(ids)
            assert sum(len(r['corner']['errors']) for r in rr) == 2499
            assert sum(len(r['corner']['observed_errors']) for r in rr) == 2445
            assert sum(bool(r['corner']['matched']) for r in rr) == 311
    capture = C.read(C.DOC / 'POSTERIOR_PARITY.json')
    C.write(C.DOC / 'EXECUTION_LEDGER.json', dict(status='ACCURACY_COMPLETE', total_rows=5742,
        new_rows=1914, frozen_baseline_rows=3828, new_F_calls=calls, baseline_F_calls=0,
        PnP_counts=dict(pnp), N3_forward_capture=capture, inference_coordinate_seal=seal,
        new_training=0, optimizer_updates=0, new_synthetic_images=0, detector_forwards=0,
        FG_joint_cornerSubPix_calls=0, new_end_to_end_latency_benchmarks=0,
        entire319_included=True, no_pose=failures, new_F_seconds=actual_F_seconds,
        evaluation_wall_seconds=time.monotonic() - start,
        predictions_sha256=C.sha(C.DOC / 'PREDICTIONS.jsonl.gz'),
        baseline_rows_sha256=baseline_sha,
        RGB_published=False, repeated_DEV=True, physical_independent_GT=False))
    print('ACCURACY_COMPLETE', calls, 5742, 'seconds', round(time.monotonic() - start, 2), flush=True)


if __name__ == '__main__':
    main()
