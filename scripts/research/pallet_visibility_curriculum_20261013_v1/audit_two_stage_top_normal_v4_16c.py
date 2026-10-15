"""Independent postseal whole238 audit, with no pose fit or image forward."""
from collections import Counter, defaultdict
from itertools import product
from pathlib import Path
from datetime import datetime, timezone
import argparse
import numpy as np

from . import common as C
from . import statewise_evaluation_v4 as E
from . import evaluate_two_stage_top_normal_v4_16c as RUN
from . import statewise_top_normal_outline_solver_v4_16 as N
from . import statewise_archived_BI_tangent_solver_v4_15 as T
from . import score_remainder_derived_reference_v4_14 as REF
from .audit_remainder_pipeline_v4_14 import independent_metric, verify_source_closure

OUTPUT = RUN.OUTPUT.parent / 'independent_top_normal_audit_v4_16c'


def _proper(R):
    R = np.asarray(R, float)
    return (R.shape == (3, 3) and np.isfinite(R).all()
            and np.allclose(R.T @ R, np.eye(3), atol=1e-7, rtol=0)
            and abs(np.linalg.det(R) - 1.) < 1e-7)


def _joint_check(solution, task, family, boundary):
    """Re-evaluate scalar/Jacobian callbacks only; never run an optimizer."""
    spec = task['spec']
    xyz, K, states, visible, domains, hw, cfg, records, regions, Q, prior, _ = N._prepare(
        family['native_xyz'], spec['K'], spec['states'], spec['visible'], spec['domains'],
        spec['image_hw'], family['coarse_pose'], spec['config'], boundary, task['u_regions'])
    if not solution['accepted']:
        assert solution['R'] is None and solution['t'] is None
        assert solution['projected_native'] is None and solution['objective'] is None
        assert solution['objective_components'] == {}
        assert not solution['OT_solved'] and not solution['U_constraints_satisfied']
        assert 'tangent_prediction_factor_audit' in solution
        for record in records:
            assert solution['reported_native'][record['native_corner']] is None
        return dict(accepted=False, constraint_callbacks=0, minimum_joint_slack=None)
    R = np.asarray(solution['R']); t = np.asarray(solution['t'])
    assert _proper(R) and _proper(Q)
    assert np.allclose(R @ Q, solution['R_physical'], atol=1e-9, rtol=0)
    scale = max(float(np.linalg.norm(np.ptp(xyz, axis=0))), 1e-3)
    p = N.V._pose_vector(R, t, scale)
    q, depth = N.V.project_native(xyz, R, t, K)
    assert np.allclose(q, solution['projected_native'], atol=1e-7, rtol=0)
    assert depth.min() >= cfg['positive_depth_m'] - 1e-10
    choices = [(list(d['outside_canvas_side']) if isinstance(d['outside_canvas_side'], tuple)
                else [d['outside_canvas_side']] if d['outside_canvas_side'] else list(N.V._SIDES))
               if d['state'] == 'T' else [None] for d in domains]
    feasible = []
    for sides in product(*choices):
        slack = N.P._joint_constraint_pair(p, xyz, K, visible, domains, sides, hw,
                                          scale, cfg, records, regions)[0]
        if np.isfinite(slack).all() and slack.min() >= -cfg['feasibility_tolerance']:
            feasible.append(float(slack.min()))
    assert feasible, 'Accepted pose violates an individual frozen joint guard'
    center = np.asarray(family['coarse_pose']['R']) @ N.NORMAL.TOP_NORMAL_CF
    assert np.allclose(center, solution['top_plane_normal_prior_audit']['coarse_normal_center'], atol=1e-12, rtol=0)
    _, _, normal = N.NORMAL.normal_log_and_jacobian(R, prior['R'])
    assert np.isfinite(normal['angle_rad'])
    assert solution['rank']['prior_not_counted_as_image_information']
    assert not solution['ambiguity']['global_uniqueness_proved']
    assert not solution['OT_unique_solution_claimed']
    assert all(states[r['native_corner']] in ('B', 'I') for r in records)
    assert all(states[r['native_corner']] == 'U' for r in regions)
    assert all(d['state'] in ('O', 'T') for d in domains)
    partial_ids = {r['native_corner'] for r in records}
    for m in visible:
        if m['native_corner'] not in partial_ids:
            assert solution['reported_native'][m['native_corner']] == m['xy'].tolist()
    for k in partial_ids:
        assert solution['reported_native'][k] is None
    return dict(accepted=True, constraint_callbacks=len(list(product(*choices))),
                minimum_joint_slack=max(feasible), top_normal_angle_deg=normal['angle_deg'])


def audit(output=OUTPUT, *, run_folder=RUN.OUTPUT):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('Preserve every independent audit version')
    folder = Path(run_folder).resolve(); protocol = C.read(folder / 'PROTOCOL.json')
    predseal = C.read(folder / 'PREDICTION_SEAL.json')
    scoreseal = C.read(folder / 'SCORE_SEAL.json')
    assert predseal['rows'] == 238 and predseal['all3_arms_complete']
    assert C.sha(folder / 'PREDICTIONS.jsonl.gz') == predseal['predictions_sha256']
    assert C.sha(folder / 'PROTOCOL.json') == predseal['protocol_sha256']
    assert C.sha(folder / 'INPUT_SEAL.json') == predseal['input_seal_sha256']
    E._verify_binding(scoreseal['prediction_seal'])
    for key in ('source_sha256', 'input_sha256', 'protected_user_sha256'):
        E._verify_sources(protocol[key])
    for path, sha in protocol['source_sha256'].items():
        assert C.sha(folder / 'code_snapshot' / path) == sha
    verify_source_closure()
    failed_runs = []
    for name in ('two_stage_top_normal_eval_v4_16b', 'two_stage_top_normal_eval_v4_16b_cpu8'):
        failed = folder.parent / name
        assert failed.exists() and not (failed / 'PREDICTION_SEAL.json').exists()
        assert not (failed / 'SCORE_SEAL.json').exists()
        old_protocol = C.read(failed / 'PROTOCOL.json')
        assert old_protocol['ids'] == protocol['ids']
        assert old_protocol['normal_profiles'] == protocol['normal_profiles']
        assert old_protocol['immutable_prepared_archive'] == protocol['immutable_prepared_archive']
        receipts = [E._binding(path) for path in sorted(failed.glob('*RECEIPT*.json'))]
        failed_runs.append(dict(output=str(failed.relative_to(C.ROOT)), incomplete_unscored=True,
            protocol=E._binding(failed / 'PROTOCOL.json'), input_seal=E._binding(failed / 'INPUT_SEAL.json'),
            immutable_math_inputs_equal=True, receipts=receipts))
    parentseal = C.read(folder / 'INPUT_SEAL.json')['parent_input_seal']
    E._verify_binding(parentseal)
    reports = {}; refs = {}
    for partition, count in (('CAL', 46), ('TEST', 49), ('REMAINDER', 143)):
        path = folder / f'{partition}_SCORE.json'
        assert C.sha(path) == scoreseal['score_sha256'][partition]
        data = C.read(path)
        assert data['count'] == count and len(data['rows']) == count
        reports[partition] = {r['id']: r for r in data['rows']}
        E._verify_binding(scoreseal['reference_bindings'][partition])
        if partition == 'REMAINDER':
            for row in C.read(REF.OUTPUT / 'PLAN.json')['annotations']:
                refs[row['id']] = REF.reference_from_document(C.read(C.ROOT / row['annotation']))
        else:
            path = C.ROOT / scoreseal['reference_bindings'][partition]['path']
            refs.update({key: row['target']['truth'] for key, row in C.read(path)['rows'].items()})
    checks = 0; counters = Counter(); ids = []
    differences = dict(translation_cm=0., rotation_deg=0., ADDsym_cm=0.)
    values = defaultdict(lambda: defaultdict(list))
    for task, row in zip(C.rows(RUN.ARCHIVE.OUTPUT / 'INPUTS.jsonl.gz'), C.rows(folder / 'PREDICTIONS.jsonl.gz'), strict=True):
        ident = task['id']; partition = task['partition']; ids.append(ident)
        assert ident == row['id'] and partition == row['partition']
        assert E._digest(task) == row['input_sha256']
        assert row['arms'][RUN.ARMS[0]] == task['baseline']
        checks += 3; counters[partition] += 1
        bank = task['stage1_family_seed_bank']; original = bank['ORIGINAL_WD']['family']
        center = np.asarray(original['coarse_pose']['R']) @ N.NORMAL.TOP_NORMAL_CF
        for name in ('ORIGINAL_WD', 'SWAPPED_WD'):
            family = bank[name]['family']
            Q = np.asarray(family['physical_registration_Q'])
            assert _proper(Q) and _proper(family['coarse_pose']['R'])
            assert np.allclose(np.asarray(family['coarse_pose']['R']) @ N.NORMAL.TOP_NORMAL_CF, center, atol=1e-12, rtol=0)
            checks += 2
        legacy = T.archive_legacy_B_tangent_predictions(task['boundary_original_legacy'],
            task['spec']['visible'], task['spec']['states'], task['spec']['config'])
        for name in RUN.ARMS[1:]:
            arm = row['arms'][name]; aa = row['normal_two_stage_audit'][name]
            assert aa['normal_parameters'] == protocol['normal_profiles'][name]
            assert aa['stage1_selected_family'] == arm['WD_family'] == row['fixed_family'][name]
            assert aa['stage2_family_frozen'] and aa['same_initial_top_normal_center_both_stages']
            assert not aa['new_full_P1_P3P_global_search']
            checks += 4
            accepted = 0
            for family in aa['stage1_all_families']:
                expected = bank[family['name']]['family']
                for key in ('native_xyz', 'cf_extents', 'physical_registration_Q'):
                    assert np.allclose(family[key], expected[key], atol=1e-12, rtol=0)
                    checks += 1
                result = _joint_check(family['solution'], task, expected, legacy)
                accepted += int(result['accepted']); counters['joint_pose_callbacks'] += result['constraint_callbacks']
                counters['stage1_accepted' if result['accepted'] else 'stage1_rejected'] += 1
            hierarchy = arm['hierarchical_correspondence_audit']
            assert hierarchy['initial_joint_feasible_family_count'] == accepted
            assert hierarchy['initial_correspondence_unresolved'] == (accepted == 0)
            if accepted == 0:
                assert arm['WD_family'] == 'ORIGINAL_WD'
                assert arm['ambiguity_audit']['material_ambiguity']
                assert not arm['solution']['OT_solved'] and not arm['keypoint_readout']['OT_solved']
                counters['stage1_unresolved_cases'] += 1
            checks += 2
            result = _joint_check(arm['solution'], task, bank[arm['WD_family']]['family'], task['boundary_all'])
            counters['joint_pose_callbacks'] += result['constraint_callbacks']
            counters['stage2_accepted' if result['accepted'] else 'stage2_rejected'] += 1
            assert arm['states'] == task['spec']['states']
            assert hierarchy['fixed_family'] == aa['stage1_selected_family']
            checks += 2
        for name in RUN.ARMS:
            arm = row['arms'][name]; metric = independent_metric(arm['actual_pose'], refs[ident])
            stored = reports[partition][ident]['metrics'][name]
            for key in differences:
                assert (metric[key] is None) == (stored[key] is None)
                if metric[key] is not None:
                    diff = abs(metric[key] - stored[key]); differences[key] = max(differences[key], diff)
                    assert diff < 1e-8
                checks += 1
            assert metric['success_5cm_5deg'] == stored['success_5cm_5deg']
            checks += 1
            values[partition][name].append(dict(id=ident, **metric, accepted=arm['accepted']))
    assert ids == protocol['ids'] and len(ids) == 238
    assert all(counters[p] == n for p, n in (('CAL', 46), ('TEST', 49), ('REMAINDER', 143)))
    summaries = {}
    for partition, arms in values.items():
        summaries[partition] = {}
        for name, rows in arms.items():
            t = np.array([r['translation_cm'] for r in rows]); r = [v['rotation_deg'] for v in rows if v['rotation_deg'] is not None]
            summary = dict(rows=len(rows), mean_translation_cm=float(t.mean()), maximum_translation_cm=float(t.max()),
                maximum_translation_id=rows[int(t.argmax())]['id'], mean_rotation_deg=float(np.mean(r)) if r else None,
                translation_le5=int(np.sum(t <= 5)), successes_5cm_5deg=sum(bool(v['success_5cm_5deg']) for v in rows),
                fallback_count=sum(not v['accepted'] for v in rows))
            stored = C.read(folder / f'{partition}_SCORE.json')['summary'][name]
            assert abs(summary['mean_translation_cm'] - stored['translation_cm']['mean']) < 1e-8
            assert summary['translation_le5'] == stored['translation_le5']
            assert summary['successes_5cm_5deg'] == stored['successes_5cm_5deg']
            checks += 3; summaries[partition][name] = summary
    for key in ('source_sha256', 'input_sha256', 'protected_user_sha256'):
        E._verify_sources(protocol[key])
    output.mkdir(parents=True)
    C.write(output / 'AUDIT.json', dict(schema='INDEPENDENT_TWO_STAGE_NORMAL_ALL238_AUDIT_V4_16C', status='PASS',
        reviewed_at_UTC=datetime.now(timezone.utc).isoformat(), checks=checks, failures=0, counters=dict(counters),
        summaries=summaries, maximum_metric_difference=differences, auditor_source=E._binding(__file__),
        independent_metric_helper_source=E._binding(independent_metric.__code__.co_filename),
        prediction_seal=E._binding(folder / 'PREDICTION_SEAL.json'), score_seal=E._binding(folder / 'SCORE_SEAL.json'),
        source_sha256=protocol['source_sha256'], protected_user_sha256=protocol['protected_user_sha256'],
        preserved_failed_runs=failed_runs,
        scalar_joint_constraint_check='every accepted stage1 family and selected stage2 pose; no optimizer, image forward or pose fit',
        S_U_image_terms_added=False, original_BI_covariance_and_reliability_preserved_by_bound_inputs=True,
        prior_excluded_from_image_information_rank=True, no_global_uniqueness_or_independent_metrology_claim=True,
        all238_fallbacks_native_flags_unconfirmed_reference_axes_preserved=True,
        metric_recomputation_uses_exact_frozen_reference_policy=True, parameter_or_default_adoption_by_auditor=False))
    C.write(output / 'AUDIT_SEAL.json', dict(audit=E._binding(output / 'AUDIT.json'), auditor_source=E._binding(__file__),
        prediction_seal=E._binding(folder / 'PREDICTION_SEAL.json'), source_inputs_protected_verified_pre_post=True))
    print(dict(status='PASS', checks=checks, failures=0, differences=differences,
               receipt=E._binding(output / 'AUDIT.json')), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-folder', type=Path, default=RUN.OUTPUT)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    audit(args.output, run_folder=args.run_folder)
