"""Compute moments and paired session uncertainty from sealed scored rows only."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path
import numpy as np

from . import common as C
from ..pallet_kp_corrected_supervision_20261010_v1 import statistics as S


def distributions(rows):
    value = S.summary([dict(r, solver=r.get('solver') or {}) for r in rows])
    value['fixed_control_outputs'] = sum(r.get('output_status') == 'FRESH_FIXED_CONTROL' for r in rows)
    return value


def compute(new_rows, fixed_rows, cohort, bootstrap):
    new_rows = [dict(r, solver=r.get('solver') or {}) for r in new_rows]
    fixed_rows = [dict(r, solver=r.get('solver') or {}) for r in fixed_rows]
    groups = defaultdict(list)
    for row in fixed_rows + new_rows:
        groups[row['method']].append(row)
    methods = ('BASE', 'N3_SUBPIX') + C.METHODS
    ids = list(cohort['ids'])
    expected = set(ids)
    for method in methods:
        data = groups[method]
        C.require(len(data) == len({r['id'] for r in data}) == 245 and {r['id'] for r in data} == expected,
                  'same245 ID population missing for ' + method)
    by_method = {m: {r['id']: r for r in groups[m]} for m in methods}
    labels = {r['id']: r['label'] for r in cohort['frames']}
    sessions = bootstrap['sessions']
    draws = np.asarray(bootstrap['counts'], np.uint16)
    C.require(draws.shape == (10000, 13) and (draws.sum(1) == 13).all(), 'existing session draws invalid')
    all_inverse = np.asarray([sessions.index(by_method['N3_SUBPIX'][fid]['session']) for fid in ids])
    contrasts = [(m, 'N3_SUBPIX') for m in C.METHODS] + [
        (C.PRIMARY, 'N3_BASIN_ROBUST'),
        (C.PRIMARY, 'N3_VALIDATED_ROLE_NO_MASK'),
        ('N3_BASIN_ROBUST', 'N3_BASIN_STANDARD'),
    ]
    strata = {}
    for scope, selected in [('combined', ids), ('easy', [fid for fid in ids if labels[fid] == 'clean']),
                            ('medium', [fid for fid in ids if labels[fid] == 'moderate'])]:
        inverse = np.asarray([sessions.index(by_method['N3_SUBPIX'][fid]['session']) for fid in selected])
        summaries = {m: distributions([by_method[m][fid] for fid in selected]) for m in methods}
        pairs = {}
        for candidate, comparator in contrasts:
            pairs[candidate + '_minus_' + comparator] = {
                contract: S.paired(by_method[candidate], by_method[comparator], selected, inverse, draws, contract)
                for contract in ('common_operational', 'candidate_new_pose')
            }
        strata[scope] = dict(frames=len(selected), methods=summaries, contrasts=pairs)
    primary = strata['combined']['contrasts'][C.PRIMARY + '_minus_N3_SUBPIX']['common_operational']
    metrics = primary['metrics']
    full = primary['common_frames'] == 245
    better = full and metrics['translation_cm']['mean_delta'] < 0 and metrics['rotation_deg']['mean_delta'] < 0
    return dict(schema='boundary_refiner_saved_row_statistics_v2', complete=True,
                population=dict(frames=245, clean=153, moderate=92, severe_excluded=74,
                                all_frames_retained=True, sessions=len(set(all_inverse.tolist()))),
                primary_method=C.PRIMARY, comparator='N3_SUBPIX',
                reference='GEOMETRIC_PROXY development reference, not independent physical pose GT',
                methods=strata['combined']['methods'], contrasts=strata['combined']['contrasts'], strata=strata,
                verdict=dict(primary_full_operational_T_and_R_improved=bool(better),
                             all245_paired_outputs_available=full,
                             implementation_success_is_performance_success=False,
                             independent_generalization_established=False),
                zero_new_detector_head_PnP_ray_or_optimizer_calls=True,
                bootstrap=dict(resamples=10000, sessions=sessions, new_draws_generated=0,
                               serialized_raw_sha256=bootstrap['serialized_raw_sha256']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', default=str(C.PRIVATE))
    parser.add_argument('--output', default=str(C.DOC))
    args = parser.parse_args()
    folder = Path(args.input); output = Path(args.output)
    new = list(C.rows(folder / 'PREDICTIONS.jsonl.gz'))
    fixed = list(C.rows(folder / 'FIXED_PREDICTIONS.jsonl.gz'))
    cohort = C.read(C.CORRECTED / 'COHORT.json')
    bootstrap_path = C.REPO / '_docs/experiments/pallet_kp_difficulty_20261010_v1/BOOTSTRAP_SESSION_DRAWS.json.gz'
    with gzip.open(bootstrap_path, 'rt') as stream:
        bootstrap = json.load(stream)
    result = compute(new, fixed, cohort, bootstrap)
    result['bindings'] = {name: C.binding(path) for name, path in {
        'new_scored_rows': folder / 'PREDICTIONS.jsonl.gz',
        'fixed_scored_rows': folder / 'FIXED_PREDICTIONS.jsonl.gz',
        'cohort': C.CORRECTED / 'COHORT.json',
        'bootstrap': bootstrap_path,
        'statistics_code': Path(__file__),
        'statistics_protocol': C.DOC / 'STATISTICS_PROTOCOL.json',
    }.items()}
    C.write_new(output / 'METRICS.json', result)
    print(json.dumps(dict(verdict=result['verdict'],
                         primary=result['methods'][C.PRIMARY]['metrics']['operational']), ensure_ascii=False))


if __name__ == '__main__':
    main()
