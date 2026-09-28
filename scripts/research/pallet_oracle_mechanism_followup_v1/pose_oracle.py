"""CPU diagnostics on frozen D9 candidates; references enter only after freeze.

Run as a module with ``freeze``, then ``score`` and ``sanity``. Private arrays
are isolated under pose_oracle; public output contains aggregates and hashes.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import inspect
import json
from pathlib import Path
import time

import cv2
import numpy as np
from scripts.research.pallet_clean19_pose_mismatch_v1 import diagnose as D
from scripts.research.pallet_selftraining_paper_closure_v1 import common as P
from scripts.research.pallet_material_selftrain_closure_v1 import common as W

ROOT = P.ROOT
NAME = 'pallet_oracle_mechanism_followup_v1'
RAW = ROOT / 'data/pallet/results' / NAME / 'pose_oracle'
DOC = ROOT / '_docs/experiments' / NAME
ARMS = {'PLASTIC': ('R0', 'RAW_LR5', 'REF_LR5', 'TEACHER'),
        'WOOD': ('R0', 'WOOD_RAW_LR5', 'WOOD_REF_LR5', 'TEACHER')}


def read(path):
    return json.loads(Path(path).read_text())


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    path = Path(path)
    assert path.is_relative_to(RAW) or path.is_relative_to(DOC)
    encoded = value if isinstance(value, str) else json.dumps(P.clean(value), indent=2, ensure_ascii=False, allow_nan=False) + '\n'
    if path.exists():
        assert path.read_text() == encoded, ('Refusing overwrite', path)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(encoded)
    temporary.replace(path)


def bind(path):
    return P.bind(path)


def verify(binding):
    P.verify(binding)


def elapsed(start):
    return dict(wall_seconds=time.perf_counter()-start[0], cpu_seconds=time.process_time()-start[1],
                GPU_hours=0, fits=0, optimizer_updates=0)


def population(material):
    """Read memberships, inference metadata and frozen outputs, never labels."""
    if material == 'PLASTIC':
        recs = P.records()
        metadata = {r['id']: r for r in read(P.META)}
        rows = [dict(metadata[r['id']], recording=r['recording_group'], severity=r['severity']) for r in recs]
        directory, predictions_lock = P.RAW, P.DOC / 'PREDICTIONS_LOCK.json'
        paths = [P.META, P.SPLIT]
    else:
        rows = read(W.RAW / 'EVAL_METADATA.json')
        directory, predictions_lock = W.RAW, W.DOC / 'WOOD_PREDICTIONS_LOCK.json'
        paths = [W.RAW / 'EVAL_METADATA.json']
    for binding in read(predictions_lock)['files']:
        verify(binding)
    paths += [predictions_lock, directory / 'PREDICTIONS.json', directory / 'POSE_PREDICTIONS.json']
    ids = [r['id'] for r in rows]
    assert len(ids) == len(set(ids)) == (128 if material == 'PLASTIC' else 45)
    allowed = {'id', 'K', 'xyz', 'hw', 'recording', 'recording_group', 'severity', 'session', 'image', 'object_type'}
    assert all(set(r) <= allowed for r in rows)
    predictions = read(directory / 'PREDICTIONS.json')
    poses = read(directory / 'POSE_PREDICTIONS.json')
    assert all(set(predictions[a]) == set(ids) for a in ARMS[material])
    return rows, predictions, poses, paths


def candidate_record(prediction, metadata):
    """GT-free exact existing D9 candidates plus reproducible inference cues."""
    q = D.points(prediction)
    K, xyz = np.asarray(metadata['K']), np.asarray(metadata['xyz'])
    selected = D.select(q, K, xyz)
    current = D.Pose.infer(q, K, xyz, False)
    assert selected['selected_hypothesis'] == current.get('selected_hypothesis')
    hypotheses = []
    for h in selected['hypotheses']:
        pose = D.hyp_pose(h, q, K, xyz)
        if h['name'] == selected['selected_hypothesis']:
            D.close(pose, current)
        cues = None
        if pose['available']:
            X = D.Pose.cuboid(*pose['cf_extents'])
            camera = X @ np.asarray(pose['R_cf']).T + np.asarray(pose['centroid'])
            projected = camera @ K.T
            projected = projected[:, :2] / projected[:, 2, None]
            residual = np.linalg.norm(projected-q[:8], axis=1)
            cues = dict(corner8_residual_px=residual.tolist(), mean_px=float(residual.mean()),
                        median_px=float(np.median(residual)), trimmed6_mean_px=float(np.sort(residual)[:6].mean()),
                        max_px=float(residual.max()), corner8_positive_depth_fraction=float(np.mean(camera[:, 2] > 0)))
        hypotheses.append(dict(name=h['name'], pose=pose, selector_score=h['score'],
                               score_components=h['score_components'], inference_cues=cues))
    return dict(current=current, selected_name=selected['selected_hypothesis'], selector_status=selected['status'],
                hypotheses=hypotheses, reference_coordinates_read=False)


def freeze():
    start = time.perf_counter(), time.process_time()
    lock_path = RAW / 'CANDIDATES_LOCK.json'
    if lock_path.exists():
        for b in read(lock_path)['files'] + read(lock_path)['sources']:
            verify(b)
        print('CANDIDATES_ALREADY_FROZEN', flush=True)
        return
    sources, artifacts, parity = [], [], {}
    for material in ARMS:
        rows, predictions, saved_poses, paths = population(material)
        output, counts = {}, Counter()
        for arm in ARMS[material]:
            output[arm] = {}
            for row in rows:
                fid = row['id']
                record = candidate_record(predictions[arm][fid], row)
                if arm in saved_poses:
                    D.close(record['current'], saved_poses[arm][fid])
                    counts['existing_pose_numeric_parity'] += 1
                # Repeat every solve, not merely serialization, to audit determinism.
                D.close(record, candidate_record(predictions[arm][fid], row))
                counts['repeat_solver_exact_within_1e-7'] += 1
                output[arm][fid] = record
            print('CANDIDATES', material, arm, len(output[arm]), flush=True)
        path = RAW / f'{material}_CANDIDATES.json'
        save(path, dict(material=material, ids=[r['id'] for r in rows],
                        groups={r['id']: dict(recording=r['recording'], severity=r['severity']) for r in rows},
                        arms=output, GT_input=False, private_coordinates=True))
        artifacts.append(bind(path)); sources.extend(paths); parity[material] = dict(counts)
    sources += [Path(__file__), Path(D.__file__), Path(D.Pose.__file__), Path(D.Selector.__file__),
                Path(inspect.getsourcefile(D.Pose.pose_auc)), Path(inspect.getsourcefile(D.Pose.solve))]
    save(lock_path, dict(created_at=now(), files=artifacts, sources=[bind(p) for p in dict.fromkeys(sources)],
                         no_reference_coordinates_read=True, inference_only=True, parity=parity,
                         candidate_generation='Exact original D9 hypotheses, original corner8 SQPnP/LM for each; no extra GT hypotheses',
                         timing=elapsed(start)))


def oracle_choice(hypotheses):
    available = [h for h in hypotheses if h['metric']['available']]
    return min(available, key=lambda h: (h['metric']['ADDsym_normalized'], h['name'])) if available else None


def score_record(task):
    fid, row, truth = task
    cpu_start = time.process_time()
    current = D.metric(fid, row['current'], truth)
    hypotheses = [dict(name=h['name'], metric=D.metric(fid, h['pose'], truth)) for h in row['hypotheses']]
    best = oracle_choice(hypotheses)
    reversed_best = oracle_choice(list(reversed(hypotheses)))
    assert (best or {}).get('name') == (reversed_best or {}).get('name')
    oracle = best['metric'] if best else dict(id=fid, available=False)
    if current['available']:
        selected = next(h for h in hypotheses if h['name'] == row['selected_name'])
        D.close(selected['metric'], current)
        assert oracle['available'] and oracle['ADDsym_normalized'] <= current['ADDsym_normalized'] + 1e-12
    return fid, dict(current=current, hypotheses=hypotheses, oracle=oracle, oracle_name=best['name'] if best else None,
                     child_cpu_seconds=time.process_time()-cpu_start)


def aggregate(rows):
    current, oracle = D.aggregate([r['current'] for r in rows]), D.aggregate([r['oracle'] for r in rows])
    assert oracle['ADDsym_AUC'] >= current['ADDsym_AUC'] - 1e-12
    gain = oracle['ADDsym_AUC'] - current['ADDsym_AUC']
    return dict(current=current, oracle=oracle, gap=gain,
                current_coverage=current['available']/len(rows), oracle_coverage=oracle['available']/len(rows),
                no_available_candidate=len(rows)-oracle['available'],
                selection_recoverable_frames=sum(r['oracle']['available'] and (not r['current']['available'] or
                    r['oracle']['ADDsym_normalized'] < r['current']['ADDsym_normalized']-1e-12) for r in rows),
                AUC_recoverable_frames=sum(D.Pose.pose_auc([r['oracle']['ADDsym_normalized']], 1.) >
                    D.Pose.pose_auc([r['current']['ADDsym_normalized'] if r['current']['available'] else float('inf')], 1.)
                    for r in rows if r['oracle']['available']))


def score(workers=4):
    start = time.perf_counter(), time.process_time()
    dst = RAW / 'POSE_ORACLE_RESULTS.json'
    if dst.exists():
        print('POSE_ORACLE_RESULTS_ALREADY_FROZEN', flush=True); return
    lock_path = RAW / 'CANDIDATES_LOCK.json'; lock = read(lock_path)
    for b in lock['files'] + lock['sources']:
        verify(b)
    save(RAW / 'SCORING_START.json', dict(created_at=now(), candidates_lock=bind(lock_path)))
    # First evaluation reference-coordinate access, after both materials' lock.
    _, truth = D.Pose.metadata('REAL_DEV')
    result, child_cpu = {}, 0.
    for material in ARMS:
        candidates = read(RAW / f'{material}_CANDIDATES.json')
        ids, groupmeta = candidates['ids'], candidates['groups']
        groups = {'ALL': ids}
        for key in ('recording', 'severity'):
            for value in sorted({r[key] for r in groupmeta.values()}):
                groups[f'{key}:{value}'] = [i for i in ids if groupmeta[i][key] == value]
        existing = read((P.RAW if material == 'PLASTIC' else W.RAW) / 'POSE_METRICS.json')
        baseline = read((P.DOC / 'CORE_RESULTS.json') if material == 'PLASTIC' else W.DOC / 'WOOD_RESULTS.json')['groups']['ALL']
        arm_rows, parity_count = {}, 0
        for arm, rows in candidates['arms'].items():
            with ProcessPoolExecutor(max_workers=workers) as pool:
                arm_rows[arm] = dict(pool.map(score_record, [(i, rows[i], truth[i]) for i in ids], chunksize=8))
            child_cpu += sum(r['child_cpu_seconds'] for r in arm_rows[arm].values())
            if arm in existing:
                for i in ids:
                    D.close(arm_rows[arm][i]['current'], existing[arm][i]); parity_count += 1
            print('POSE_SCORED', material, arm, flush=True)
        summaries = {g: {a: aggregate([rr[i] for i in ii]) for a, rr in arm_rows.items()} for g, ii in groups.items()}
        for a in ARMS[material]:
            if a in baseline:
                assert np.isclose(summaries['ALL'][a]['current']['ADDsym_AUC'], baseline[a]['sixD']['ADDsym_AUC'], atol=1e-12)
        save(RAW / f'{material}_METRICS.json', dict(is_oracle=True, GT_DEPENDENT=True, DIAGNOSTIC_ONLY=True, arms=arm_rows))
        result[material] = dict(groups=summaries, current_all_arms_baseline={a: v['sixD'] for a, v in baseline.items()},
                                pose_numeric_parity_frames=parity_count, frames=len(ids),
                                recordings=sum(k.startswith('recording:') for k in groups),
                                reference='Legacy geometry-derived pose; not independent physical pose',
                                candidate_lock=bind(RAW / f'{material}_CANDIDATES.json'))
    timing = elapsed(start); timing['worker_cpu_seconds'] = child_cpu; timing['total_cpu_seconds'] = timing['cpu_seconds'] + child_cpu
    payload = dict(created_at=now(), materials=result, timing=timing, is_oracle=True, GT_DEPENDENT=True,
                   DIAGNOSTIC_ONLY=True, candidates_lock=bind(lock_path),
                   metric_contract=dict(error='whole-object proper yaw-group corresponding 8-corner ADD / norm(registry_xyz)',
                                        AUC_fraction=0.1, integration_points=1001, integration='trapezoidal',
                                        failure='infinite normalized error; full frame denominator retained',
                                        detection_match_gate=False, tie='minimum ADDsym_normalized then stable hypothesis name'),
                   invariance=dict(repeated_solve=True, reversed_candidate_order=True, production_in_candidate_set=True,
                                   oracle_ge_production=True),
                   reference_bindings=[bind(P.TRUTH), bind(D.Pose.E.C.POSE / 'GEOMETRY_RESOLVED_POSE_GT.json'),
                                       bind(D.Pose.E.C.POSE / 'AXIS_REVIEW_MANIFEST.json')])
    save(dst, payload)
    save(DOC / 'ORACLE_POSE_RESULTS.json', payload)


def project(X, R, t, K):
    camera = np.asarray(X) @ np.asarray(R).T + np.asarray(t)
    homogeneous = camera @ np.asarray(K).T
    return homogeneous[:, :2] / homogeneous[:, 2, None]


def sanity():
    start = time.perf_counter(), time.process_time()
    if (RAW / 'GEOMETRY_SANITY_RESULTS.json').exists():
        print('GEOMETRY_SANITY_ALREADY_FROZEN', flush=True); return
    # This entire stage is privileged; it is never an inference consumer.
    lock = read(RAW / 'CANDIDATES_LOCK.json')
    for b in lock['files'] + lock['sources']:
        verify(b)
    reference_xy = read(P.TRUTH); _, real_truth = D.Pose.metadata('REAL_DEV')
    real = {}; raw = {}
    for material in ARMS:
        rows, _, _, _ = population(material); metrics = []
        for row in rows:
            fid = row['id']; q = np.asarray(reference_xy[fid]['gt'], dtype=float)
            q[(q == -1).all(1)] = np.nan
            pose = D.Pose.infer(q, np.asarray(row['K']), np.asarray(row['xyz']), False)
            m = D.metric(fid, pose, real_truth[fid]); metrics.append(m)
        real[material] = D.aggregate(metrics); raw[material] = metrics
    table_path = ROOT / 'challenge/yolo_pose_one_model/pallet_translation_loss_v1/GEOMETRY_SIDETABLE.npz'
    table = dict(np.load(table_path)); index = {str(s): i for i, s in enumerate(table['stems'])}
    _, syn_truth = D.Pose.metadata('SYNTH_HELDOUT')
    # Uniform deterministic positions in the existing heldout membership, no metric-based selection.
    population_ids = sorted(syn_truth)
    chosen = [population_ids[i] for i in np.linspace(0, len(population_ids)-1, min(64, len(population_ids)), dtype=int)]
    synthetic = []
    for fid in chosen:
        i = index[fid]; fx, fy, cx, cy = table['K'][i]; pad = table['pad'][i]
        K = np.array([[fx, 0., cx-pad], [0., fy, cy-pad], [0., 0., 1.]])
        X = table['Xcf'][i]; points = project(np.vstack([X, np.zeros(3)]), table['R'][i], table['t'][i], K)
        pose = D.Pose.infer(points, K, table['dims'][i], True)
        metric = D.metric(fid, pose, syn_truth[fid])
        synthetic.append(metric)
    raw['SYNTHETIC_EXACT_PROJECTION'] = synthetic
    save(RAW / 'GEOMETRY_SANITY_PRIVATE.json', dict(is_oracle=True, GT_DEPENDENT=True, DIAGNOSTIC_ONLY=True, metrics=raw,
         synthetic_ids=chosen))
    payload = dict(created_at=now(), is_oracle=True, GT_DEPENDENT=True, DIAGNOSTIC_ONLY=True,
                   REAL_REFERENCE_INPUT=real, SYNTHETIC_EXACT_PROJECTION=D.aggregate(synthetic),
                   synthetic_sample_rule='64 evenly spaced sorted existing SYNTH_HELDOUT identifiers; no outcome filtering',
                   synthetic_geometry_binding=bind(table_path), timing=elapsed(start),
                   interpretation=dict(real='Circular consistency check: annotations and geometry-derived reference share source; not physical6D ceiling',
                                       synthetic='Exact renderer projection in existing Xcf convention and native camera; source=True keeps original Rx(pi) basis',
                                       verified_visible_substitution='WOOD: NA_REFERENCE_NOT_VERIFIED; PLASTIC: not required for fixed candidate oracle'))
    save(RAW / 'GEOMETRY_SANITY_RESULTS.json', payload); save(DOC / 'GEOMETRY_SANITY_RESULTS.json', payload)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage', choices=['freeze', 'score', 'sanity'])
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args(); cv2.setNumThreads(1)
    if args.stage == 'score': score(args.workers)
    else: globals()[args.stage]()


if __name__ == '__main__':
    main()
