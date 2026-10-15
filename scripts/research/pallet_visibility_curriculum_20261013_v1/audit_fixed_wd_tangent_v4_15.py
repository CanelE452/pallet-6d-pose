"""Compact independent postseal metric/control audit; no fitting or inference."""
from collections import Counter, defaultdict
from pathlib import Path
from datetime import datetime, timezone
import numpy as np

from . import common as C
from . import statewise_evaluation_v4 as E
from . import evaluate_fixed_wd_archived_tangent_v4_15 as RUN
from . import score_remainder_derived_reference_v4_14 as REF
from .audit_remainder_pipeline_v4_14 import independent_metric, verify_source_closure

OUTPUT = RUN.OUTPUT.parent / 'independent_tangent_audit_v4_15'


def audit(output=OUTPUT):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('Preserve every independent review version')
    folder = RUN.OUTPUT
    protocol = C.read(folder / 'PROTOCOL.json')
    predseal = C.read(folder / 'PREDICTION_SEAL.json')
    scoreseal = C.read(folder / 'SCORE_SEAL.json')
    inputseal = C.read(folder / 'INPUT_SEAL.json')
    assert predseal['rows'] == 238 and predseal['all3_arms_complete']
    assert C.sha(folder / 'PREDICTIONS.jsonl.gz') == predseal['predictions_sha256']
    assert C.sha(folder / 'PROTOCOL.json') == predseal['protocol_sha256']
    assert C.sha(folder / 'INPUT_SEAL.json') == predseal['input_seal_sha256']
    assert C.sha(folder / 'INPUTS.jsonl.gz') == inputseal['input_sha256']
    E._verify_binding(scoreseal['prediction_seal'])
    for key in ('source_sha256', 'input_sha256', 'protected_user_sha256'):
        E._verify_sources(protocol[key])
    for path, sha in protocol['source_sha256'].items():
        assert C.sha(folder / 'code_snapshot' / path) == sha
    verify_source_closure()
    reports = {}
    refs = {}
    for partition, count in (('CAL', 46), ('TEST', 49), ('REMAINDER', 143)):
        path = folder / f'{partition}_SCORE.json'
        assert C.sha(path) == scoreseal['score_sha256'][partition]
        report = C.read(path)
        assert report['count'] == count and len(report['rows']) == count
        E._verify_binding(scoreseal['reference_bindings'][partition])
        reports[partition] = {r['id']: r for r in report['rows']}
        if partition == 'REMAINDER':
            plan = C.read(REF.OUTPUT / 'PLAN.json')
            for row in plan['annotations']:
                refs[row['id']] = REF.reference_from_document(C.read(C.ROOT / row['annotation']))
        else:
            path = C.ROOT / scoreseal['reference_bindings'][partition]['path']
            for key, row in C.read(path)['rows'].items():
                refs[key] = row['target']['truth']
    checks = 0
    maximum_diffs = dict(translation_cm=0., rotation_deg=0., ADDsym_cm=0.)
    values = defaultdict(lambda: defaultdict(list))
    alias_counts = Counter()
    input_counts = Counter()
    ids = []
    active_counts = Counter()
    for task, prediction in zip(C.rows(folder / 'INPUTS.jsonl.gz'), C.rows(folder / 'PREDICTIONS.jsonl.gz'), strict=True):
        ident = task['id']; partition = task['partition']; ids.append(ident)
        assert ident == prediction['id'] and partition == prediction['partition']
        assert E._digest(task) == prediction['input_sha256']
        assert prediction['arms'][RUN.ARMS[0]] == task['baseline']
        checks += 3; input_counts[partition] += 1
        original = {m['native_corner']: m for m in task['spec']['visible']}
        Q = np.asarray(task['family']['physical_registration_Q'])
        for arm in RUN.ARMS:
            result = prediction['arms'][arm]
            assert result['WD_family'] == task['fixed_family'] == prediction['fixed_family']
            assert result['states'] == task['spec']['states']
            for record in result['solution']['visible_observations']:
                source = original[record['native_corner']]
                for key in ('state', 'xy', 'covariance_px2', 'fit_covariance_px2', 'fit_reliability_weight'):
                    assert record[key] == source[key]
                    checks += 1
            if result['accepted']:
                assert np.allclose(np.asarray(result['actual_pose']['R_cf']) @ Q,
                                   result['actual_pose']['R_physical'], atol=1e-9, rtol=0)
                checks += 1
            if arm != RUN.ARMS[0]:
                control = prediction['tangent_control_audit'][arm]
                if control['exact_baseline_alias']:
                    assert result == task['baseline']
                    alias_counts[arm] += 1; checks += 1
                else:
                    ta = result['solution']['tangent_prediction_factor_audit']
                    assert ta['weight'] == 1 and ta['variance_inflation'] == 4
                    assert ta['active_native_ids'] == control['active_tangent_ids']
                    assert len(ta['active_native_ids']) == len(set(ta['active_native_ids']))
                    assert all(task['spec']['states'][k] in ('B', 'I') for k in ta['active_native_ids'])
                    assert ta['S_terms'] == ta['U_terms'] == 0
                    checks += 5; active_counts[arm] += 1
                    for row in ta['rows']:
                        assert row['assumed_variance_px2'] == 4 * row['original_directional_variance_px2']
                        assert row['fixed_CAL_reliability'] == original[row['native_corner']]['fit_reliability_weight']
                        checks += 2
            metric = independent_metric(result['actual_pose'], refs[ident])
            saved = reports[partition][ident]['metrics'][arm]
            for key in maximum_diffs:
                assert (metric[key] is None) == (saved[key] is None)
                if metric[key] is not None:
                    diff = abs(metric[key] - saved[key])
                    maximum_diffs[key] = max(maximum_diffs[key], diff)
                    assert diff < 1e-8
                checks += 1
            assert metric['success_5cm_5deg'] == saved['success_5cm_5deg']
            checks += 1
            values[partition][arm].append(dict(id=ident, **metric, accepted=result['accepted']))
    assert ids == protocol['ids'] and len(ids) == 238
    assert input_counts == Counter(CAL=46, TEST=49, REMAINDER=143)
    summaries = {}
    for partition, arms in values.items():
        summaries[partition] = {}
        for name, rows in arms.items():
            translations = np.array([r['translation_cm'] for r in rows])
            rotations = [r['rotation_deg'] for r in rows if r['rotation_deg'] is not None]
            summary = dict(rows=len(rows), mean_translation_cm=float(translations.mean()),
                maximum_translation_cm=float(translations.max()),
                maximum_translation_id=rows[int(translations.argmax())]['id'],
                mean_rotation_deg=float(np.mean(rotations)) if rotations else None,
                translation_le5=sum(translations <= 5), successes_5cm_5deg=sum(bool(r['success_5cm_5deg']) for r in rows),
                fallback_count=sum(not r['accepted'] for r in rows))
            saved = C.read(folder / f'{partition}_SCORE.json')['summary'][name]
            assert abs(summary['mean_translation_cm'] - saved['translation_cm']['mean']) < 1e-8
            assert abs(summary['maximum_translation_cm'] - saved['translation_cm']['max']) < 1e-8
            assert summary['translation_le5'] == saved['translation_le5']
            assert summary['successes_5cm_5deg'] == saved['successes_5cm_5deg']
            assert summary['fallback_count'] == saved['fallback_count']
            checks += 5; summaries[partition][name] = summary
    for key in ('source_sha256', 'input_sha256', 'protected_user_sha256'):
        E._verify_sources(protocol[key])
    output.mkdir(parents=True)
    C.write(output / 'AUDIT.json', dict(schema='INDEPENDENT_FIXED_WD_TANGENT_POSTSEAL_AUDIT_V4_15',
        status='PASS', reviewed_at_UTC=datetime.now(timezone.utc).isoformat(), checks=checks, failures=0,
        auditor_source=E._binding(__file__), prediction_seal=E._binding(folder / 'PREDICTION_SEAL.json'),
        score_seal=E._binding(folder / 'SCORE_SEAL.json'), source_sha256=protocol['source_sha256'],
        protected_user_sha256=protocol['protected_user_sha256'], population=dict(input_counts),
        maximum_metric_difference=maximum_diffs, exact_baseline_alias_counts=dict(alias_counts),
        active_tangent_control_counts=dict(active_counts), summaries=summaries,
        reference_policy_unchanged=True, all_failures_native_flags_and_unconfirmed_axes_kept=True,
        independent_metric='Euclidean centroid, SciPy SO3 C2 angle, explicit C2 transformed body vertex ADD',
        no_fits_or_inference=True, no_default_adoption=True,
        finding='Both tangent controls break achieved TEST49 maximum5cm objective; frozen e baseline remains preserved. Remaining143 mean improves but remains far from goal.',
        new_parameters_or_runtime_GT_routing=False))
    C.write(output / 'AUDIT_SEAL.json', dict(audit=E._binding(output / 'AUDIT.json'),
        auditor_source=E._binding(__file__), prediction_seal=E._binding(folder / 'PREDICTION_SEAL.json'),
        source_inputs_and_protected_verified_pre_and_post=True))
    print(dict(status='PASS', checks=checks, failures=0, maximum_metric_difference=maximum_diffs,
               receipt=E._binding(output / 'AUDIT.json')), flush=True)


if __name__ == '__main__':
    audit()
