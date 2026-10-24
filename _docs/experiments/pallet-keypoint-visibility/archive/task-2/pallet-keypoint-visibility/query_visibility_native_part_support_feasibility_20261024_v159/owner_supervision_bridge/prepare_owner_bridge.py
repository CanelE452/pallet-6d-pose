import datetime
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

BASE = Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility')
ROOT = BASE / 'query_visibility_native_part_support_feasibility_20261024_v159'
OUT = ROOT / 'owner_supervision_bridge'
FACTS = BASE / 'query_visibility_mr6d_full_public_source_20261024_v153/full230_native_DINO_v1/FEATURE_INPUT_FACTS1840_INPUT_ONLY.jsonl'
OWNERS = BASE / 'query_visibility_train_query_owner_witness_20261024_v158/joined_owner_witness_v1/ALL390_SEPARATE_QUERY_VISIBLE_OWNER_WITNESS.jsonl'
SCOPE = ROOT / 'ROOT_SCOPE_BEFORE_SOURCE_NATIVE_SUPPORT_BODIES.json'
RULES = OUT / 'PRE_JOIN_RULES.json'
MAPPING = {'TARGET_VISIBLE_SUPPORTED': 0, 'FOREIGN_VISIBLE_SUPPORTED': 1, 'BACKGROUND_SUPPORTED': 2, 'UNKNOWN': -1}
IDENTITY = ['global_index', 'frame_index', 'frame_id', 'source_id', 'split', 'native_slot', 'query_id', 'query_xy', 'L']
WRITTEN = []

def binding(path):
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}

def write_json(name, value):
    path = OUT / name
    assert path.resolve().parent == OUT.resolve()
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    WRITTEN.append(path)
    return binding(path)

def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]

def digest_identity(items):
    value = [{key: row[key] for key in IDENTITY} for row in items]
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

def coverage(items, key):
    groups = defaultdict(list)
    for row in items:
        groups[str(key(row))].append(row)
    result = {}
    for group, members in sorted(groups.items()):
        result[group] = {
            'slots': len(members),
            'prior_owner_rows': sum(row['prior_owner_observation_present'] for row in members),
            'without_prior_owner_observation': sum(not row['prior_owner_observation_present'] for row in members),
            'categorical_weak_owner_target_available': sum(row['owner_target_available'] for row in members),
            'owner_counts': dict(Counter(row['owner_state'] for row in members)),
            'owner_aux_target_counts': dict(Counter(str(row['owner_aux_target']) for row in members)),
            'observation_scope_counts': dict(Counter(row['observation_scope'] for row in members)),
            'center4_valid_counts': dict(Counter('NOT_ASSESSED' if row['center4_valid'] is None else str(row['center4_valid']).lower() for row in members)),
        }
    return result

def main():
    input_bindings = {'scope': binding(SCOPE), 'facts1840': binding(FACTS), 'prior_owner390': binding(OWNERS), 'pre_join_rules': binding(RULES)}
    assert input_bindings['scope']['sha256'] == '49ca7c44ac80611c8be988b9c30f020528d4090ee40a31f8825a405690c25be3'
    assert input_bindings['facts1840']['sha256'] == 'd856400906dec6b4b78ba01c76b5c58fe5b4799685a3ced6292439e0e9c1ed7b'
    assert input_bindings['prior_owner390']['sha256'] == '7bb9d041505a770231ac44ea44e690e46d4afc198d404556fc1272c8d3c4c798'
    pre_join = write_json('PRE_JOIN_INPUT_BINDINGS.json', {'schema': 'V159_OWNER_BRIDGE_PRE_JOIN_INPUT_SEAL', 'UTC': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'inputs': input_bindings, 'rule_receipt_written_before_join': True, 'training_admitted': False})
    facts = rows(FACTS)
    observations = rows(OWNERS)
    assert len(facts) == 1840 and len(observations) == 390
    assert [row['global_index'] for row in facts] == list(range(1840))
    assert len({row['query_id'] for row in facts}) == 1840
    assert len({row['global_index'] for row in observations}) == 390
    assert len({row['query_id'] for row in observations}) == 390
    by_key = {(row['global_index'], row['query_id']): row for row in observations}
    assert set(by_key) <= {(row['global_index'], row['query_id']) for row in facts}
    assert all(row['split'] == 'SOURCE_DEVELOPMENT' for row in observations)
    assert all(row['source_only_weak_observation'] and not row['human_gold'] and not row['physical_native_gold'] and not row['pixel_coordinate_gold'] and not row['training_admitted'] for row in observations)
    result = []
    for fact in facts:
        observed = by_key.get((fact['global_index'], fact['query_id']))
        row = {key: fact[key] for key in IDENTITY}
        row['input_local_central4_available'] = fact['local_central4_available']
        if observed is not None:
            for key in ['global_index', 'query_id', 'source_id', 'frame_index', 'native_slot', 'split']:
                assert fact[key] == observed[key], (fact['global_index'], key)
            assert fact['query_xy'] == observed['original_query_xy']
            assert fact['L'] == observed['original_L']
            state = observed['final_visible_owner_support']
            assert state in MAPPING
            if not observed['center4_valid']:
                assert state == 'UNKNOWN'
            row.update(
                owner_state=state,
                owner_aux_target=MAPPING[state],
                owner_target_available=state != 'UNKNOWN',
                prior_owner_observation_present=True,
                observation_scope=observed['observation_scope'],
                center4_valid=observed['center4_valid'],
                center4_validity_provenance='SEALED_V158_PRIOR_SOURCE_OBSERVATION',
                reviewer_provenance={
                    'prior_join_binding': input_bindings['prior_owner390'],
                    'primary_owner': observed['primary_owner'],
                    'primary_reason_KO': observed['primary_reason_KO'],
                    'cross_review': observed['cross_review'],
                    'panel': observed['panel'],
                    'new_visual_review_in_this_bridge': False,
                },
                owner_unknown_reason='PRIOR_WEAK_OWNER_UNKNOWN' if state == 'UNKNOWN' else None,
            )
        else:
            row.update(
                owner_state='UNKNOWN', owner_aux_target=-1, owner_target_available=False,
                prior_owner_observation_present=False, observation_scope='NO_PRIOR_OWNER_OBSERVATION',
                center4_valid=None, center4_validity_provenance='NOT_ASSESSED_NO_RGB_OR_VALIDMASK_BODY_READ',
                reviewer_provenance=None,
                owner_unknown_reason='SOURCE_HELD_OWNER_REVIEW_PROHIBITED' if fact['split'] != 'SOURCE_DEVELOPMENT' else 'NO_PRIOR_SOURCE_OWNER_OBSERVATION',
            )
        row.update(
            supervision_domain='SAVED_RGB96_CENTER2X2_CATEGORICAL_VISIBLE_OWNER_WEAK',
            native_visibility_target_defined=False,
            point_quality_target_defined=False,
            source_only_weak_observation=True,
            human_gold=False, physical_native_gold=False, pixel_coordinate_gold=False,
            training_admitted=False,
        )
        result.append(row)
    assert digest_identity(result) == digest_identity(facts)
    assert sum(row['prior_owner_observation_present'] for row in result) == 390
    assert sum(not row['prior_owner_observation_present'] for row in result) == 1450
    assert Counter(row['owner_state'] for row in result) == Counter({'TARGET_VISIBLE_SUPPORTED': 53, 'FOREIGN_VISIBLE_SUPPORTED': 52, 'BACKGROUND_SUPPORTED': 148, 'UNKNOWN': 1587})
    assert Counter(row['observation_scope'] for row in result) == Counter({'SINGLE_REVIEW_WEAK': 350, 'CROSS40_THREE_AGREE_WEAK': 30, 'CROSS40_CONSERVATIVE_UNKNOWN': 10, 'NO_PRIOR_OWNER_OBSERVATION': 1450})
    assert Counter(row['native_slot'] for row in result) == Counter({i: 230 for i in range(8)})
    assert Counter(row['split'] for row in result) == Counter({'SOURCE_DEVELOPMENT': 1480, 'SOURCE_VALIDATION': 360})
    assert all(row['owner_aux_target'] == -1 and row['owner_state'] == 'UNKNOWN' and not row['prior_owner_observation_present'] and row['center4_valid'] is None for row in result if row['split'] != 'SOURCE_DEVELOPMENT')
    assert all(row['owner_aux_target'] == -1 and row['owner_state'] == 'UNKNOWN' and row['center4_valid'] is None for row in result if not row['prior_owner_observation_present'])
    assert all(row['owner_aux_target'] == MAPPING[row['owner_state']] and row['owner_target_available'] == (row['owner_aux_target'] >= 0) for row in result)
    invalid = [row for row in result if row['center4_valid'] is False]
    assert len(invalid) == 1 and invalid[0]['global_index'] == 1043 and invalid[0]['owner_aux_target'] == -1
    assert all(not row['training_admitted'] and not row['native_visibility_target_defined'] and not row['point_quality_target_defined'] for row in result)
    forbidden_field_names = {'original_role_state', 'original_role_target', 'original_entity_target', 'point_quality_target', 'role_external_label_changed'}
    assert all(not (set(row) & forbidden_field_names) for row in result)
    target_path = OUT / 'ALL1840_SEPARATE_OWNER_AUXILIARY_SUPERVISION.jsonl'
    assert target_path.resolve().parent == OUT.resolve()
    target_path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in result))
    WRITTEN.append(target_path)
    coverage_binding = write_json('COVERAGE_BY_SOURCE_NATIVE.json', {
        'schema': 'V159_OWNER_BRIDGE_ALL1840_COVERAGE',
        'all': coverage(result, lambda row: 'ALL'),
        'by_split': coverage(result, lambda row: row['split']),
        'by_source_sequence': coverage(result, lambda row: row['source_id'].split(':')[0]),
        'by_native': coverage(result, lambda row: row['native_slot']),
        'by_source_sequence_and_native': coverage(result, lambda row: row['source_id'].split(':')[0] + ':native' + str(row['native_slot'])),
        'by_original_source_id': coverage(result, lambda row: row['source_id']),
        'training_admitted': False,
    })
    input_bindings_after = {'scope': binding(SCOPE), 'facts1840': binding(FACTS), 'prior_owner390': binding(OWNERS), 'pre_join_rules': binding(RULES)}
    assert input_bindings_after == input_bindings
    guard_binding = write_json('GUARD_CHECKS.json', {
        'PASS': True, 'all1840_retained_once_in_original_order': True,
        'all_geometry_identity_split_equal_input': True,
        'input_and_output_identity_sha256': digest_identity(facts),
        'known390_exact_key_and_geometry_join': True, 'other1450_UNKNOWN_minus1': True,
        'all360_source_held_UNKNOWN_minus1': True,
        'native8_counts': {str(i): 230 for i in range(8)},
        'source_split_counts_unchanged': {'SOURCE_DEVELOPMENT': 1480, 'SOURCE_VALIDATION': 360},
        'owner_is_distinct_from_native_visibility_and_quality': True,
        'no_original_visibility_or_quality_targets_copied_or_modified': True,
        'reviewer_scopes_preserved': {'single_weak': 350, 'cross_three_agree_weak': 30, 'cross_conservative_UNKNOWN': 10},
        'invalid_Q1043_preserved_UNKNOWN_minus1_center4_false': True,
        'center4_mask_counts': {'true': 389, 'false': 1, 'NOT_ASSESSED': 1450},
        'inputs_byte_identical_before_after': True,
        'script_writes_restricted_to_owner_bridge_directory': True,
        'RGB_or_held_label_or_feature_NPZ_model_fit_forward_private_current_V112_PnP_qL_or_selection_calls': 0,
        'training_admitted': False,
    })
    complete = {
        'schema': 'V159_PUBLIC_MR_OWNER_AUXILIARY_SUPERVISION_BRIDGE_COMPLETE',
        'completed': True, 'PASS': True, 'UTC': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'scope': 'Concrete data preparation only: existing public MR6D wood factual1840 slots with sealed prior SOURCE_DEVELOPMENT390 owner weak observations; no new judgments.',
        'input_rows': 1840, 'prior_owner_rows': 390, 'rows_without_prior_owner_observation': 1450,
        'source_held_rows_all_UNKNOWN': 360,
        'owner_counts': dict(Counter(row['owner_state'] for row in result)),
        'owner_auxiliary_target_map': MAPPING,
        'owner_target_available_rows': sum(row['owner_target_available'] for row in result),
        'observation_scope_counts': dict(Counter(row['observation_scope'] for row in result)),
        'center4_mask_counts': {'true': 389, 'false': 1, 'NOT_ASSESSED': 1450},
        'inputs': input_bindings, 'pre_join_input_seal': pre_join,
        'code': binding(Path(__file__)),
        'outputs': [binding(target_path), coverage_binding, guard_binding],
        'original_qL_native_split_order_visibility_quality_untouched': True,
        'human_pixel_native_coordinate_physical_gold_created': 0,
        'training_admitted': False, 'model_or_experiment_protocol_selected': None,
        'objective': 'Full WOOD/SQUARE/native8 objective retained. Public MR6D wood slots only; all1840 slots retained without filtering.',
        'future_use_limit': 'Any later experiment must separately freeze source-only protocol and training admission. Categorical owner targets are not native D/E or point-quality targets.',
    }
    write_json('COMPLETE.json', complete)
    print(json.dumps({'PASS': True, 'rows': len(result), 'owner_counts': complete['owner_counts'], 'available_categorical_weak_targets': complete['owner_target_available_rows'], 'complete': binding(OUT / 'COMPLETE.json')}, ensure_ascii=False))

if __name__ == '__main__':
    main()
