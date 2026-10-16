"""Strict prediction-only replay parity against the original e journals.

Compares all stored endpoint diagnostics, all feasible samples, both stages
and registered families, selected/fallback numeric poses and final source.
Historical journal compaction is reproduced only for repeated audit lists.
No reference, metric, optimizer, image, or inferred pose is opened/computed.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import gzip
import json
import math
from pathlib import Path

from .candidates import canonical, pose_record, sha
from .typed_replay import actual_pose

ATOL = 1e-9
POSE_CONTRACT = ('available', 'R_cf', 'centroid', 'R_physical', 'physical_registration_Q',
                 'cf_extents', 'source', 'joint_feasible', 'fallback', 'fallback_pose_does_not_solve_OT')


def rows(path):
    with gzip.open(path, 'rt') as stream:
        for line in stream:
            yield json.loads(line)


def read(path):
    return json.loads(Path(path).read_text())


def compare(expected, actual, path='', issues=None):
    """Exact structure/types/strings; all floats have fixed absolute tolerance."""
    issues = [] if issues is None else issues
    if isinstance(expected, bool) or expected is None or isinstance(expected, str):
        if type(actual) is not type(expected) or actual != expected:
            issues.append(dict(path=path, expected=expected, actual=actual, kind='exact'))
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        numeric = isinstance(actual, (int, float)) and not isinstance(actual, bool)
        finite = numeric and math.isfinite(expected) and math.isfinite(actual)
        if not finite or abs(actual-expected) > ATOL:
            issues.append(dict(path=path, expected=expected, actual=actual, kind='numeric',
                               absolute_difference=None if not finite else abs(actual-expected)))
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(expected) != len(actual):
            issues.append(dict(path=path, expected_count=len(expected),
                               actual_count=len(actual) if isinstance(actual, list) else None, kind='list_shape'))
        else:
            for index, (e, a) in enumerate(zip(expected, actual)):
                compare(e, a, path+'/'+str(index), issues)
    elif isinstance(expected, dict):
        if not isinstance(actual, dict) or set(expected) != set(actual):
            issues.append(dict(path=path, expected_keys=sorted(expected),
                               actual_keys=sorted(actual) if isinstance(actual, dict) else None, kind='dict_shape'))
        if isinstance(actual, dict):
            for key in expected.keys() & actual.keys():
                compare(expected[key], actual[key], path+'/'+str(key), issues)
    else:
        raise TypeError('Unsupported archived JSON value at '+path)
    return issues


def _compact_outcomes(records):
    result = deepcopy(records)
    for record in result:
        for name in ('visible_measurement_audit', 'domain_audit'):
            audit = record.pop(name, None)
            if audit is not None:
                record[name+'_count'] = len(audit)
    return result


def _pose_contract(value):
    return {key: value[key] for key in POSE_CONTRACT}


def stage_view(stage, expected):
    """Project a replay result into every numerical archived comparison field."""
    families = []
    for family, old in zip(stage['families'], expected['families']):
        solution = family['solution']
        families.append(dict(name=family['name'], native_xyz=family['native_xyz'],
            cf_extents=family['cf_extents'], physical_registration_Q=family['physical_registration_Q'],
            coarse_pose=family['coarse_pose'], accepted=bool(solution['accepted']),
            solver_schema=solution['schema'], solver_status=solution.get('status'),
            original_outcomes=_compact_outcomes(solution.get('candidates', [])),
            feasible_samples=solution.get('feasible_pose_samples', []),
            selected_solver_pose=pose_record(solution, family),
            published_actual_pose=_pose_contract(actual_pose(solution, family)),
            original_objective=solution.get('objective'),
            original_objective_components=solution.get('objective_components'),
            original_search=solution.get('search'), original_parent_search=solution.get('parent_search_audit'),
            original_rank=solution.get('rank'), original_ambiguity=solution.get('ambiguity')))
    return dict(selected_family=stage['selected_family'], family_ambiguity=stage['family_ambiguity'], families=families)


def expected_stage_view(stage):
    fields = ('name', 'native_xyz', 'cf_extents', 'physical_registration_Q', 'coarse_pose',
              'accepted', 'solver_schema', 'solver_status', 'selected_solver_pose', 'published_actual_pose',
              'original_objective', 'original_objective_components', 'original_search',
              'original_parent_search', 'original_rank', 'original_ambiguity')
    families = []
    for family in stage['families']:
        result = {key: family[key] for key in fields}
        result['published_actual_pose'] = _pose_contract(result['published_actual_pose'])
        result['original_outcomes'] = _compact_outcomes([r['original_record'] for r in family['original_outcomes']])
        result['feasible_samples'] = [r['original_record'] for r in family['feasible_samples']]
        families.append(result)
    return dict(selected_family=stage['selected_family'], family_ambiguity=stage['family_ambiguity'], families=families)


def compare_frame(original, replay):
    if replay.get('technical_failure'):
        return dict(id=original['id'], passed=False, issues=[dict(kind='technical_failure',
                    error=replay.get('error'), traceback=replay.get('traceback'))])
    issues = compare(original['id'], replay['id'], 'id')
    compare(original['original_e']['frozen_family'], replay['selected_family'], 'selected_family', issues)
    compare(original['original_e']['accepted'], replay['accepted'], 'accepted', issues)
    compare(not original['original_e']['accepted'], replay['fallback'], 'fallback', issues)
    compare(_pose_contract(original['original_e']['actual_pose']), _pose_contract(replay['actual_pose']), 'actual_pose', issues)
    for name in ('G10', 'G13'):
        expected = expected_stage_view(original['stages'][name])
        actual = stage_view(replay['stage_results'][name], original['stages'][name])
        compare(expected, actual, name, issues)
    return dict(id=original['id'], passed=not issues, issues=issues,
                auxiliary_pose_metadata_omitted=sorted(set(original['original_e']['actual_pose'])-set(replay['actual_pose'])),
                original_row_sha256=canonical(original), replay_row_sha256=canonical(replay))


def check(*, output, candidate_folder, phase, receipt):
    output, candidate_folder, receipt = Path(output), Path(candidate_folder), Path(receipt)
    protocol_path = output/'REPLAY_PROTOCOL.json'; protocol = read(protocol_path)
    ids = protocol['pilot_ids'] if phase == 'pilot' else protocol['ordered_ids']
    wanted, seen, results, slot_bindings = set(ids), set(), [], {}
    original_path = candidate_folder/'ORIGINAL_E_CANDIDATE_BANK.jsonl.gz'
    manifest_path = candidate_folder/'CANDIDATE_MANIFEST.json'; manifest = read(manifest_path)
    if sha(original_path) != manifest['artifacts'][original_path.name]:
        raise ValueError('Original candidate bank seal differs')
    index = {ident: i for i, ident in enumerate(protocol['ordered_ids'])}
    for original in rows(original_path):
        if original['id'] not in wanted:
            continue
        ident = original['id']
        slot_path = output/'slots'/f'{index[ident]:03d}_TYPED_MULTI.json'
        if not slot_path.exists():
            results.append(dict(id=ident, passed=False, issues=[dict(kind='MISSING_TYPED_MULTI_SLOT')]))
        else:
            slot = read(slot_path)
            if slot['id'] != ident:
                raise ValueError('Exact slot ID required')
            slot_bindings[str(slot_path.relative_to(output))] = sha(slot_path)
            results.append(compare_frame(original, slot['result']))
        seen.add(ident)
    if seen != wanted:
        raise ValueError('All declared original parity rows required')
    order = {ident: i for i, ident in enumerate(ids)}
    results.sort(key=lambda r: order[r['id']])
    value = dict(schema='STRICT_ORIGINAL_E_TYPED_REPLAY_PARITY_V1', phase=phase, rows=len(results),
        ordered_ids=ids, all_passed=all(r['passed'] for r in results),
        passed_rows=sum(r['passed'] for r in results), atol=ATOL, rtol=0.,
        original_bank_sha256=sha(original_path), original_manifest_sha256=sha(manifest_path),
        protocol_sha256=sha(protocol_path), input_slot_sha256=slot_bindings,
        comparison_fields='All original stage optimizer outcome records and feasible samples, selected solver/actual poses, objective/components, search/rank/ambiguity; exact family/native model/source/accept/fallback; original repeated-audit compaction only',
        actual_pose_contract=list(POSE_CONTRACT),
        omitted_auxiliary_metadata_is_reported_not_numerical_or_selection_parity=True,
        GT_or_metrics_or_optimizer_calls=0, result=results)
    with receipt.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')
    return value


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--candidate-folder', required=True)
    parser.add_argument('--phase', choices=('pilot', 'full'), required=True)
    parser.add_argument('--receipt', required=True)
    args = parser.parse_args()
    result = check(**vars(args))
    print(json.dumps({k: result[k] for k in ('rows', 'all_passed', 'passed_rows', 'atol', 'rtol')}))
