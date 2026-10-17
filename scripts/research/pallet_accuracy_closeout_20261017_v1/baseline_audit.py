"""Cheap Phase A gate: independently remeasure sealed original poses, never solve.

Only this postprediction audit reads the frozen references. No image, neural,
PnP, optimization, candidate generation, or learned selector is imported.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import resource
import time

import numpy as np

ARMS = ('R0', 'N0', 'E0', 'POSEFIX_RAW')
PARTITIONS = {'CAL': 46, 'TEST': 49, 'REMAINDER': 143}
EXPECTED = {'R0': (66, 14.536, 14.282), 'N0': (108, 11.306, 10.631),
            'E0': (114, 14.112, 13.001), 'POSEFIX_RAW': (80, 13.836, 11.640)}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def read_journal(path):
    with gzip.open(path, 'rt') as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')


def validate_pose(pose):
    if type(pose.get('available')) is not bool:
        raise ValueError('Explicit pose availability is required')
    if not pose['available']:
        return
    R = np.asarray(pose['R_physical'], dtype=float)
    t = np.asarray(pose['centroid'], dtype=float)
    if (R.shape != (3, 3) or t.shape != (3,) or not np.isfinite(R).all()
            or not np.isfinite(t).all() or not np.allclose(R.T @ R, np.eye(3), atol=1e-7, rtol=0)
            or not math.isclose(float(np.linalg.det(R)), 1., abs_tol=1e-7)):
        raise ValueError('Finite center and proper physical rotation are required')


def metric(pose, reference):
    """Direct center error, and min angle over exactly I/Y(pi); never Y(pi/2)."""
    validate_pose(pose)
    if reference['order'] != 2:
        raise ValueError('Original physical C2 reference is required')
    if not pose['available']:
        return dict(pose_available=False, translation_cm=None, rotation_deg=None, success=False)
    R = np.asarray(pose['R_physical'], dtype=float)
    t = np.asarray(pose['centroid'], dtype=float)
    G = np.asarray(reference['R'], dtype=float)
    if G.shape != (3, 3):
        raise ValueError('Reference rotation is unresolved')
    d = float(np.linalg.norm(t - np.asarray(reference['t'], dtype=float)) * 100.)
    c, s = math.cos(math.pi), math.sin(math.pi)
    symmetries = (np.eye(3), np.array([[c, 0., s], [0., 1., 0.], [-s, 0., c]]))
    angles = [math.degrees(math.acos(max(-1., min(1., float((np.trace((G @ S).T @ R) - 1.) / 2.)))))
              for S in symmetries]
    a = min(angles)
    return dict(pose_available=True, translation_cm=d, rotation_deg=a,
                success=d <= 5. and a <= 5.)


def distribution(values, denominator):
    v = np.asarray([x for x in values if x is not None], dtype=float)
    return dict(n=len(v), full_denominator=denominator, missing=denominator - len(v),
                mean=float(np.mean(v)) if len(v) else None,
                sample_std=float(np.std(v, ddof=1)) if len(v) > 1 else None,
                sample_variance=float(np.var(v, ddof=1)) if len(v) > 1 else None,
                median=float(np.median(v)) if len(v) else None,
                P90=float(np.quantile(v, .90)) if len(v) else None,
                P95=float(np.quantile(v, .95)) if len(v) else None,
                max=float(np.max(v)) if len(v) else None,
                ddof=1, available_only=True, tail_trimming=False)


def summarize(rows):
    result = {}
    for arm in ARMS:
        m = [r['metrics'][arm] for r in rows]
        result[arm] = dict(denominator=len(rows), successes=sum(v['success'] for v in m),
                           unavailable=sum(not v['pose_available'] for v in m),
                           translation_cm=distribution([v['translation_cm'] for v in m], len(m)),
                           rotation_deg=distribution([v['rotation_deg'] for v in m], len(m)))
    return result


def transition(rows, before='N0', after='E0'):
    groups = dict(BOTH_SUCCESS=[], RECOVERY=[], DAMAGE=[], BOTH_FAILURE=[])
    for row in rows:
        a, b = row['metrics'][before]['success'], row['metrics'][after]['success']
        key = 'BOTH_SUCCESS' if a and b else 'RECOVERY' if b else 'DAMAGE' if a else 'BOTH_FAILURE'
        groups[key].append(row['id'])
    return dict(before=before, after=after, counts={k: len(v) for k, v in groups.items()}, ids=groups)


def load_bound_original(folder):
    folder = Path(folder)
    seal = read_json(folder / 'SCORE_SEAL.json')
    bindings = {}
    for name in ('BASELINE_PREDICTIONS.jsonl.gz', 'FIXED_REFERENCES.json', 'ROW_METRICS.csv'):
        digest = sha(folder / name)
        if digest != seal['artifacts'][name]:
            raise ValueError('Original sealed artifact changed: ' + name)
        bindings[name] = digest
    rows = read_journal(folder / 'BASELINE_PREDICTIONS.jsonl.gz')
    ids = [r['id'] for r in rows]
    if (len(rows) != 238 or len(set(ids)) != 238
            or Counter(r['partition'] for r in rows) != Counter(PARTITIONS)):
        raise ValueError('Original exact CAL46/TEST49/REMAINDER143 is required')
    refs = read_json(folder / 'FIXED_REFERENCES.json')['references']
    if any(set(refs[p]) != {r['id'] for r in rows if r['partition'] == p} for p in PARTITIONS):
        raise ValueError('Reference identities do not exactly match original identities')
    return rows, refs, bindings


def run(folder, output):
    started, cpu = time.perf_counter(), time.process_time()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    original, refs, bindings = load_bound_original(folder)
    rows = []
    for row in original:
        rows.append({**{k: row[k] for k in ('id', 'partition', 'session', 'cluster_id')},
                     'metrics': {a: metric(row['predictions'][a]['actual_pose'],
                                           refs[row['partition']][row['id']]) for a in ARMS}})
    # Check every original metric, rather than only matching rounded means.
    with (Path(folder) / 'ROW_METRICS.csv').open() as f:
        old = {(r['id'], r['arm']): r for r in csv.DictReader(f) if r['arm'] in ARMS}
    maximum = dict(translation_cm=0., rotation_deg=0.)
    for row in rows:
        for arm in ARMS:
            saved = old[(row['id'], arm)]
            new = row['metrics'][arm]
            if new['success'] != (saved['success_le5'] == 'True'):
                raise ValueError('Old/new success mismatch: ' + row['id'] + ':' + arm)
            for field in maximum:
                diff = abs(new[field] - float(saved[field]))
                maximum[field] = max(maximum[field], diff)
                if diff > 1e-8:
                    raise ValueError('Old/new metric mismatch: ' + row['id'] + ':' + arm + ':' + field)
    primary = [r for r in rows if r['partition'] != 'CAL']
    populations = {'PRIMARY192': summarize(primary), 'CAL46': summarize([r for r in rows if r['partition'] == 'CAL']),
                   'COMMON95': summarize([r for r in rows if r['partition'] != 'REMAINDER'])}
    for arm, (successes, t, a) in EXPECTED.items():
        value = populations['PRIMARY192'][arm]
        if (value['successes'] != successes or abs(value['translation_cm']['mean'] - t) > .000501
                or abs(value['rotation_deg']['mean'] - a) > .000501):
            raise ValueError('Phase A published hard gate failed: ' + arm)
    paired = transition(primary)
    if paired['counts'] != dict(BOTH_SUCCESS=89, RECOVERY=25, DAMAGE=19, BOTH_FAILURE=59):
        raise ValueError('E0/N0 exact paired hard gate failed')
    result = dict(schema='ACCURACY_CLOSEOUT_BASELINE_GATE_V1', gate='PASS',
                  source_bindings=bindings, score_seal_sha256=sha(Path(folder) / 'SCORE_SEAL.json'),
                  ordered_ids=[r['id'] for r in rows], primary_ordered_ids=[r['id'] for r in primary],
                  populations=populations, E0_minus_N0=paired, old_row_max_abs_error=maximum,
                  success='same pose translation_cm<=5 AND physical_C2_rotation_deg<=5',
                  rectangular_90_degree_equivalence=False, source_code_sha256=sha(__file__),
                  elapsed_seconds=time.perf_counter() - started, process_cpu_seconds=time.process_time() - cpu,
                  max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  new_model_forwards=0, new_pose_solves=0, all_failures_kept=True)
    write_json(output / 'BASELINE_REPRODUCTION.json', result)
    write_json(output / 'RECOMPUTED_BASELINE_ROWS.json', {'rows': rows})
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sealed-baseline-folder', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = run(args.sealed_baseline_folder, args.output)
    print(json.dumps({k: result[k] for k in ('gate', 'elapsed_seconds', 'old_row_max_abs_error')}))
    print(json.dumps({a: {k: result['populations']['PRIMARY192'][a][k] for k in ('successes', 'denominator')}
                      for a in ARMS}))


if __name__ == '__main__':
    main()
