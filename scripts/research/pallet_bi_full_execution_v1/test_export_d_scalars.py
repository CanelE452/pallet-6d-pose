"""Constructed scalar/corruption tests; no dataset or reference input opened."""
from copy import deepcopy
import json
import math
from pathlib import Path
import statistics
from tempfile import TemporaryDirectory
from . import export_d_scalars as E


def dist(values):
    ordered = sorted(values)
    # Four-value fixture, explicit interpolation at positions 2.7 and 2.85.
    return dict(n=4, mean=sum(values)/4, sample_SD=statistics.stdev(values),
        median=(ordered[1]+ordered[2])/2, P90=.3*ordered[2]+.7*ordered[3],
        P95=.15*ordered[2]+.85*ordered[3], max=ordered[-1])


def effect_fixture():
    values = {
        'translation_cm': {'total': [-3., 2., 0., 1.], 'within_before_selected_family': [-1., 0., 0., 1.],
                           'family_readout_contribution': [-2., 2., 0., 0.]},
        'rotation_deg': {'total': [2., -2., 0., 0.], 'within_before_selected_family': [1., -1., 0., 0.],
                         'family_readout_contribution': [1., -1., 0., 0.]},
        'success': {'total': [0., 1., -1., 0.], 'within_before_selected_family': [1., 0., -1., 0.],
                    'family_readout_contribution': [-1., 1., 0., 0.]}}
    rows = [dict(id='CONSTRUCTED_'+str(i), partition='TEST', cluster_id='CONSTRUCTED',
        before_family='ORIGINAL_WD', after_family='SWAPPED_WD' if i<2 else 'ORIGINAL_WD', family_changed=i<2,
        metrics={axis: {part: vector[i] for part, vector in parts.items()} for axis, parts in values.items()}) for i in range(4)]
    effect = dict(denominator=4, rows=rows, family_switches=2, recoveries=['CONSTRUCTED_1'], damages=['CONSTRUCTED_2'],
        decomposition={axis: {part: dist(vector) for part, vector in parts.items()} for axis, parts in values.items()})
    summary = dict(successes=2, translation_cm=dict(mean=10.), rotation_deg=dict(mean=5.))
    return effect, (summary, deepcopy(summary))


def expect_reject(call):
    try: call()
    except ValueError: return
    raise AssertionError('Constructed corruption was accepted')


def run():
    checks = dict(numeric_comparisons=0, max_absolute_scalar_difference=0., row_component_reassemblies=0)
    effect, summary = effect_fixture()
    E.check_effect(effect, summary, checks)
    assert checks['row_component_reassemblies'] == 12
    hand = dict(n=4, mean=0., sample_SD=math.sqrt(14/3), median=.5, P90=1.7, P95=1.85, max=2.)
    E.check_distribution(hand, [-3., 0., 1., 2.], checks)
    assert abs(E.distribution(hand, 4)['sample_variance_from_saved_SD'] - 14/3) < 1e-12
    bad = deepcopy(effect); bad['rows'][0]['metrics']['success']['family_readout_contribution'] = 0.
    expect_reject(lambda: E.check_effect(bad, summary, checks))
    bad = deepcopy(effect); bad['family_switches'] = 1
    expect_reject(lambda: E.check_effect(bad, summary, checks))
    bad = deepcopy(effect); bad['decomposition']['translation_cm']['total']['P95'] += .1
    expect_reject(lambda: E.check_effect(bad, summary, checks))
    bad = deepcopy(effect); bad['rows'][0]['id'] = bad['rows'][1]['id']
    expect_reject(lambda: E.check_effect(bad, summary, checks))
    expect_reject(lambda: E.distribution(hand | {'R_physical': [[1,0,0]]}, 4))
    with TemporaryDirectory(prefix='D-scalar-constructed-') as tmp:
        root = Path(tmp)
        # Deliberately undecodable data proves verify_seal does not JSON-decode
        # mixed pose artifact contents; only the metadata seal is decoded.
        artifact = root/'SCORED_ROWS.json.gz'; artifact.write_bytes(b'CONSTRUCTED_NON_JSON_POSE_BYTES\x00')
        seal = root/'EVALUATION_SEAL.json'
        E.write(seal, dict(complete=True, artifacts={artifact.name: E.sha(artifact)}))
        authority = E.sha(seal)
        original_read, decoded = E.read, []
        def guarded_read(path):
            decoded.append(Path(path).name)
            if Path(path) != seal: raise AssertionError('Mixed artifact decoded')
            return original_read(path)
        E.read = guarded_read
        try:
            E.verify_seal(seal, authority)
            assert decoded == ['EVALUATION_SEAL.json']
            artifact.write_bytes(b'TAMPERED')
            expect_reject(lambda: E.verify_seal(seal, authority))
            expect_reject(lambda: E.verify_seal(seal, '0'*64))
        finally: E.read = original_read
    return dict(PASS=True, constructed_only=True, actual_dataset_rows_decoded=0, raw_reference_decodes=0,
        mixed_pose_rows_decoded=0, image_decodes=0, scientific_calls=0, scalar_reassembly_checked=True,
        fixed_mean_quantile_variance_fixture_checked=True, family_switch_and_duplicate_guard_checked=True,
        artifact_tamper_and_authority_tamper_rejected=True, mixed_artifact_hash_without_decode=True,
        source_sha256=E.sha(E.__file__), test_source_sha256=E.sha(__file__))


if __name__ == '__main__':
    import argparse
    parser=argparse.ArgumentParser(); parser.add_argument('--output', type=Path, required=True)
    args=parser.parse_args(); result=run(); E.write(args.output, result); print(json.dumps(result, sort_keys=True))
