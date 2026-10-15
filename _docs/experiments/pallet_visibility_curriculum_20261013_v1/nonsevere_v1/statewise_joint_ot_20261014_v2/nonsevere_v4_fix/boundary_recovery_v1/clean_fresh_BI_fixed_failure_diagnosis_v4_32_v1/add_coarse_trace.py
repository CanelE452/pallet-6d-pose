"""Append-only trace of stored candidates; no objective or pose generation."""
from pathlib import Path
import json
import hashlib
import gzip
import numpy as np

ROOT = Path('/home/minjae/Documents/github/pallet-pose')
OUT = Path(__file__).parent
RUN = OUT.parent/'clean_fresh_BI_pose_eval_v4_32_r2'
CASES = ('wood_night_01:031344', 'wood_night_01:031426', 'eval_noapril:1775201428535961600')
OWN = 'ORIGINAL_WD/OWN_INDEPENDENT_INITIAL_COARSE'
EXPECTED_PRED_SEAL = 'eff80efdb1cb9c17f7eb13df50b953467f82bbe3415c8385b9aa146456948b48'


def sha(path):
    value = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def equal_pose(record, pose):
    return (np.allclose(record['R'], pose['R'], atol=2e-12, rtol=0.)
            and np.allclose(record['t'], pose['t'], atol=2e-12, rtol=0.))


def matched(records, pose):
    return [record for record in records if equal_pose(record, pose)]


def main():
    destination = OUT/'COARSE_TRACE_V1.json'
    if destination.exists():
        raise FileExistsError('Preserve the original diagnostic attempt')
    plan = read(OUT/'COARSE_TRACE_PLAN_V1.json')
    if sha(__file__) != plan['source_sha256'] or sha(OUT/'EVIDENCE.json') != plan['evidence_sha256']:
        raise ValueError('Predeclared source/evidence binding changed')
    sealpath = RUN/'PREDICTION_SEAL.json'
    if sha(sealpath) != EXPECTED_PRED_SEAL:
        raise ValueError('Exact closed prediction seal required')
    seal = read(sealpath)
    predpath = RUN/'PREDICTIONS.jsonl.gz'
    if sha(predpath) != seal['predictions_sha256']:
        raise ValueError('Sealed prediction bytes changed')
    evidence = {row['id']: row for row in read(OUT/'EVIDENCE.json')['cases']}
    selected = {}
    with gzip.open(predpath, 'rt', encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            if row['id'] in CASES:
                if row['id'] in selected:
                    raise ValueError('Duplicate fixed case')
                selected[row['id']] = row
    if set(selected) != set(CASES):
        raise ValueError('All fixed cases must be retained')
    rows = []
    for identifier in CASES:
        result = selected[identifier]
        arms = {}
        for arm, value in result['arms'].items():
            if arm == 'SAME_FROZEN_E':
                continue
            candidates = value['candidate_selection']['candidates']
            own = [item for item in candidates if item['name'] == OWN]
            original = value['family_refinements']['ORIGINAL_WD']['candidates']
            coarse_matches = matched(original, evidence[identifier]['original_coarse_native_pose'])
            if len(own) != 1 or not coarse_matches or not all(item['accepted'] for item in coarse_matches):
                raise ValueError('Stored original independent coarse must be feasible and retained')
            selected_pose = dict(R=value['R'], t=value['t'])
            family = value['WD_family']
            selected_matches = matched(value['family_refinements'][family]['candidates'], selected_pose)
            record = dict(
                original_independent_coarse_alias=OWN,
                original_independent_coarse_feasible_selector_records=own,
                original_independent_coarse_saved_refinement_pose_matches=coarse_matches,
                original_independent_coarse_guard_scope='saved accepted/minimum_joint_slack only; per-guard coarse arrays not stored here and not reconstructed',
                original_initial_legacy_token_present=any('ORIGINAL_INITIAL' in item['name'] for item in candidates),
                all_selector_initial_coarse_aliases=[item for item in candidates if 'COARSE' in item['name']],
                selected_name=value['candidate_selection']['selected_name'],
                selected_family=family,
                selected_endpoint_origin=value['selected_endpoint_origin'],
                selected_pose_matching_saved_refinement_records=selected_matches,
                start_selection_by_family={name: item['start_selection'] for name, item in value['family_refinements'].items()},
                best_stored_feasible_by_family={name: min((item for item in candidates if item['family'] == name), key=lambda item: (item['total_energy'], item['index'])) for name in ('ORIGINAL_WD', 'SWAPPED_WD') if any(item['family'] == name for item in candidates)},
                selected_candidate=next(item for item in candidates if item['name'] == value['candidate_selection']['selected_name']),
                selected_components=evidence[identifier]['arms'][arm]['independent_selected_components'])
            if arm.endswith('_CONTAINMENT'):
                native_arm = arm.replace('_CONTAINMENT', '_NATIVE')
                native = result['arms'][native_arm]
                native_family = native['WD_family']
                native_rows = matched(value['family_refinements'][native_family]['candidates'], dict(R=native['R'], t=native['t']))
                retained = [item for item in candidates if item['family'] == native_family and 'SAME_POLICY_NATIVE_' in item['name']
                    and abs(item['native_energy']-native['solution']['native_observation_objective']) <= 2e-12
                    and abs(item['regularizer_energy']-native['solution']['regularizer_objective']) <= 2e-12]
                if not retained:
                    raise ValueError('Current C bank must retain the native winner at its recomputed C cost')
                record['corresponding_native_selected_endpoint'] = dict(
                    arm=native_arm, family=native_family, original_native_selected_name=native['candidate_selection']['selected_name'],
                    retained_C_selector_records=retained,
                    native_pose_matching_C_search_records=native_rows,
                    retained_native_J_minus_selected_C_J=min(item['total_energy'] for item in retained)-record['selected_candidate']['total_energy'],
                    matching_scope='stored same-policy native endpoint token and independent scalar identity; no missing bank poses regenerated')
            arms[arm] = record
        rows.append(dict(id=identifier, arms=arms))
    output = dict(schema='FIXED_THREE_STORED_COARSE_AND_NATIVE_RETENTION_TRACE_V1', cases=rows,
        source_sha256={str(Path(__file__).relative_to(ROOT)):sha(__file__)},
        input_sha256={str(path.relative_to(ROOT)):sha(path) for path in (OUT/'EVIDENCE.json', OUT/'COARSE_TRACE_PLAN_V1.json', sealpath, predpath)},
        scope=dict(new_reference_GT_or_RGB_decodes=0, new_objective_guard_or_candidate_generation=0,
                   coarse_is_not_truth=True, no_global_optimality_or_unseen_bank_certificate=True))
    destination.write_text(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(cases=3, arms=12, original_coarse_feasible_and_retained=12,
                         native_winner_retained_in_C=6, new_reference_decodes=0), ensure_ascii=False))


if __name__ == '__main__':
    main()
