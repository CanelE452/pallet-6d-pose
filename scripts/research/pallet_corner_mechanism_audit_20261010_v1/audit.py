"""Decompose already recorded corner errors without new inference or pose fitting.

For the two measured lines A x = b and a recorded reference p, the exact
identity is x-p = A^{-1}(b-Ap). The right side separates measured-line
displacement from its amplification at the intersection. The reference is
the existing geometric proxy, not an independently certified physical edge.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
INPUT = ROOT / '_docs/experiments/pallet_boundary_corner_refiner_20261010_v2'
DOC = ROOT / '_docs/experiments/pallet_corner_mechanism_audit_20261010_v1'
NAMES = ('OBSERVATIONS.jsonl.gz', 'POSTHOC_ROWS.jsonl.gz', 'CALIBRATION.json',
         'CALIBRATION_GEOMETRY_ROWS.jsonl.gz', 'GEOMETRY_SEAL.json',
         'DIAGNOSTICS.json', 'PROTOCOL.json')
EDGES = ((0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),
         (0,4),(1,5),(2,6),(3,7))


def binding(path):
    path = Path(path)
    return dict(path=str(path.relative_to(ROOT)), bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def read_rows(path):
    with gzip.open(path, 'rt') as stream:
        return [json.loads(line) for line in stream]


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def describe(values):
    values = np.asarray(values, dtype=np.float64)
    assert values.ndim == 1 and np.isfinite(values).all()
    return dict(n=len(values), mean=float(values.mean()) if len(values) else None,
                sample_variance=float(values.var(ddof=1)) if len(values)>1 else None,
                median=float(np.median(values)) if len(values) else None,
                P90=float(np.quantile(values,.9)) if len(values) else None,
                maximum=float(values.max()) if len(values) else None)


def freeze(output):
    value = dict(schema='frozen_corner_mechanism_audit_v1',
        purpose='explain existing errors, not choose a new predictor or acceptance threshold',
        inputs={name:binding(INPUT/name) for name in NAMES}, code=binding(Path(__file__)),
        population='existing Clean153 + Moderate92 = 245, primary N3_VALIDATED_ROLE',
        comparisons=['all computed corners', 'already admitted', 'already selected for hybrid',
                     'already displayed', 'selected improved', 'selected worsened'],
        algebra='measured unit-normal lines: x-p = inverse(A) * (b-A*p)',
        reported_quantities=['normal displacement at reference', 'intersection amplification',
            'line support residual', 'reference error / recorded radius', 'same-corner N3 error'],
        thresholds='only already published radius and 8px correctness threshold; no new deployment gate',
        controls=['orthogonal displaced lines', 'same line displacement with nearly parallel normals',
                  'zero support residual with biased corner'],
        limits=['proxy reference is not physical edge ownership evidence',
                'source CAL intersections are virtual references from certified wire queries',
                'posthoc association is not causal attribution',
                'no source/real model, pose, input, mask, decoder, threshold or checkpoint is changed'],
        actual_new_detector_calls=0, actual_new_head_calls=0, actual_new_PnP_calls=0,
        actual_new_training_updates=0, actual_new_RGB_generation=0)
    write_new(output/'PROTOCOL.json',value)
    print('FROZEN_MECHANISM_PROTOCOL', binding(output/'PROTOCOL.json')['sha256'])


def controls():
    results=[]
    for label,A,b in (
        ('orthogonal_displacement',np.eye(2),np.array([1.,1.])),
        ('nearly_parallel_displacement',np.array([[1.,0.],[np.cos(.01),np.sin(.01)]]),np.array([1.,-1.])),
        ('perfect_support_biased_corner',np.eye(2),np.array([3.,4.]))):
        x=np.linalg.solve(A,b)
        assert np.max(np.abs(A@x-b))<1e-10
        results.append(dict(name=label, line_normal_displacement_px=b.tolist(),
            intersection_error_px=float(np.linalg.norm(x)),
            support_residual_can_equal_zero=True,
            amplification=float(np.linalg.norm(x)/np.linalg.norm(b))))
    assert results[0]['intersection_error_px']==np.sqrt(2.)
    assert results[1]['intersection_error_px']>100.
    assert results[2]['intersection_error_px']==5.
    return results


def summarize(rows):
    return dict(count=len(rows), human_states=dict(Counter(r['human_state'] for r in rows)),
        error_px=describe([r['error_px'] for r in rows]),
        N3_error_px=describe([r['N3_error_px'] for r in rows]),
        delta_candidate_minus_N3_px=describe([r['error_px']-r['N3_error_px'] for r in rows]),
        normal_displacement_norm_px=describe([r['normal_displacement_norm_px'] for r in rows]),
        amplification=describe([r['amplification'] for r in rows]),
        normal_matrix_condition=describe([r['normal_matrix_condition'] for r in rows]),
        max_line_support_rms_px=describe([max(r['line_support_rms_px']) for r in rows]),
        reference_error_over_radius=describe([r['error_over_recorded_radius'] for r in rows]),
        error_within_recorded_radius=sum(r['error_px']<=r['recorded_radius_px'] for r in rows),
        error_within_existing8px=sum(r['error_px']<=8. for r in rows),
        improves_N3=sum(r['error_px']<r['N3_error_px']-1e-9 for r in rows),
        harms_N3=sum(r['error_px']>r['N3_error_px']+1e-9 for r in rows))


def run(output):
    started=time.monotonic()
    protocol=json.loads((output/'PROTOCOL.json').read_text())
    for name, bound in protocol['inputs'].items():
        assert binding(INPUT/name)==bound, name+' changed after freeze'
    assert binding(Path(__file__))==protocol['code'], 'audit code changed after freeze'
    for name in ('ROWS.jsonl.gz','RESULTS.json'):
        assert not (output/name).exists(), 'refuse output overwrite: '+name
    observations=read_rows(INPUT/'OBSERVATIONS.jsonl.gz')
    posthoc={r['id']:r for r in read_rows(INPUT/'POSTHOC_ROWS.jsonl.gz')
             if r['method']=='N3_VALIDATED_ROLE'}
    assert len(observations)==len(posthoc)==245
    assert Counter(r['difficulty_label'] for r in posthoc.values())==dict(clean=153,moderate=92)
    rows=[]; maximum_identity_error=0.; unknown=0
    for observation in observations:
        frame=posthoc[observation['id']]
        lines={r['edge']:r for r in observation['lines']}
        evidence={r['id']:r for r in frame['observations']['boundary_corner_evidence']}
        for corner in observation['corners']:
            k=int(corner['id']); ev=evidence[k]
            if not ev['reference_available'] or ev['N3_error_px'] is None:
                unknown+=1; continue
            supporting=[lines[e] for e in corner['edges']]
            assert all(k in EDGES[e] for e in corner['edges'])
            assert all(tuple(line['endpoints'])==EDGES[line['edge']] for line in supporting)
            A=np.asarray([line['normal'] for line in supporting],float)
            b=np.asarray([line['offset'] for line in supporting],float)
            assert np.allclose(np.linalg.norm(A,axis=1),1.,rtol=0,atol=1e-10)
            p=np.asarray(frame['reference_native_points_px'][k],float)
            x=np.asarray(corner['xy'],float); displacement=b-A@p
            reconstruction=np.linalg.solve(A,displacement)
            identity_error=float(np.max(np.abs(reconstruction-(x-p))))
            maximum_identity_error=max(maximum_identity_error,identity_error)
            assert identity_error<1e-7
            error=float(np.linalg.norm(x-p)); norm=float(np.linalg.norm(displacement))
            assert abs(error-ev['error_px'])<1e-9
            n3=np.asarray(frame['fixed_N3_native_points_px'][k],float)
            n3_error=float(np.linalg.norm(n3-p))
            assert abs(n3_error-ev['N3_error_px'])<1e-9
            rows.append(dict(id=frame['id'], session=frame['session'], difficulty=frame['difficulty_label'],
                corner=k, edges=corner['edges'], human_state=frame['human_states_native'][k],
                reference_xy=p.tolist(), candidate_xy=x.tolist(), native_N3_xy=n3.tolist(),
                error_px=error, N3_error_px=n3_error, normal_displacement_px=displacement.tolist(),
                normal_displacement_norm_px=norm, amplification=error/norm if norm>1e-12 else 0.,
                normal_matrix_condition=float(np.linalg.cond(A)),
                absolute_normal_determinant=float(abs(np.linalg.det(A))),
                line_support_rms_px=[line['residual_rms_px'] for line in supporting],
                line_support_length_px=[line['support_length_px'] for line in supporting],
                line_query_ids=[line['queries'] for line in supporting],
                extrapolation_ratios=corner['extrapolation_ratios'],
                recorded_radius_px=corner['radius_px'], error_over_recorded_radius=error/corner['radius_px'],
                admitted=ev['admitted'], selected_for_hybrid=ev['selected_for_hybrid'],
                displayed=ev['displayed_as_boundary_observation'], algebra_identity_error_px=identity_error,
                physical_edge_ownership_proved=False))
    scopes={'computed':rows,'admitted':[r for r in rows if r['admitted']],
        'selected_for_hybrid':[r for r in rows if r['selected_for_hybrid']],
        'displayed':[r for r in rows if r['displayed']],
        'selected_improved':[r for r in rows if r['selected_for_hybrid'] and r['error_px']<r['N3_error_px']-1e-9],
        'selected_worsened':[r for r in rows if r['selected_for_hybrid'] and r['error_px']>r['N3_error_px']+1e-9]}
    assert [len(scopes[k]) for k in scopes]==[482,447,423,395,186,237]
    calibration=json.loads((INPUT/'CALIBRATION.json').read_text())
    source=read_rows(INPUT/'CALIBRATION_GEOMETRY_ROWS.jsonl.gz')
    scale=calibration['uncertainty']['corner_scale']
    source_ratios=[]
    for item in source:
        error=float(np.linalg.norm(np.asarray(item['predicted_xy'])-item['reference_xy']))
        assert abs(error-item['error_px'])<1e-8
        source_ratios.append(error/(item['sigma_px']*scale))
    before=protocol['inputs']
    assert {name:binding(INPUT/name) for name in NAMES}==before
    with (output/'ROWS.jsonl.gz').open('xb') as raw:
        with gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0) as zipped:
            for row in rows:
                zipped.write((json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n').encode())
    result=dict(schema='actual_frozen_corner_mechanism_results_v1', complete=True,
        protocol=binding(output/'PROTOCOL.json'), rows=binding(output/'ROWS.jsonl.gz'),
        frames=245, computed_rows=len(rows), unknown_reference_corners=unknown,
        scope={name:summarize(values) for name,values in scopes.items()},
        source_virtual_corner_calibration=dict(count=len(source),
            error_over_recorded_radius=describe(source_ratios),
            within_recorded_radius=sum(value<=1. for value in source_ratios),
            physical_corner_ownership_independently_proved=False),
        analytic_controls=controls(), maximum_algebra_identity_error_px=maximum_identity_error,
        input_bindings_unchanged=True, actual_detector_calls=0, actual_head_calls=0,
        actual_PnP_calls=0, actual_new_training_updates=0, actual_RGB_generation=0,
        threshold_or_model_changes=0, elapsed_seconds=time.monotonic()-started,
        interpretation='Exact decomposition relative to existing proxy, not physical ownership or causal proof')
    write_new(output/'RESULTS.json',result)
    print(json.dumps({k:result[k] for k in ['complete','frames','computed_rows','maximum_algebra_identity_error_px','elapsed_seconds']},ensure_ascii=False))
    print(json.dumps({name:{k:v for k,v in summarize(values).items() if k not in ['human_states','normal_matrix_condition']} for name,values in scopes.items()},ensure_ascii=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['freeze','run'])
    parser.add_argument('--output',type=Path,default=DOC)
    args=parser.parse_args()
    (freeze if args.stage=='freeze' else run)(args.output)


if __name__=='__main__':
    main()
