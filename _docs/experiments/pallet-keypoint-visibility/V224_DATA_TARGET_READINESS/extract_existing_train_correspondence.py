"""Extract existing weak labels at immutable TRAIN original queries; no fitting.

The mixed ancestor is read as binary lines. Excluded line bytes are traversed,
but only whitelisted original global indices are JSON-decoded or interpreted.
This does not claim zero excluded-byte access or create new native gold.
"""
import collections
import hashlib
import json
from pathlib import Path

P = Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility')
OUT = P / 'publication_root_preparation/V224_ROOT_READINESS'
META = P / 'query_visibility_expected_part_weak_paired_source_training_20261025_v219/source_fit_rows_v1/PUBLIC_TRAIN1480_FILTERED_METADATA.jsonl'
WHITE = P / 'publication_root_preparation/V224_YCB_TEACHER_INTERFACE_A/ORIGIN_QUERY_TRAIN_WHITELIST.json'
MIXED = P / 'query_visibility_mr6d_source_supervision_bridge_20261024_v154/ROOT_ALL1840_SEPARATE_QUERY_SUPERVISION.jsonl'


def guard(path):
    assert '_v112' not in str(path).lower()
    assert str(path).startswith(str(P) + '/')
    return path


def digest(path):
    return hashlib.sha256(guard(path).read_bytes()).hexdigest()


def save(name, value):
    with guard(OUT / name).open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def main():
    for path in (OUT, META, WHITE, MIXED):
        guard(path)
    OUT.mkdir(exist_ok=True)
    assert digest(META) == '231a7f0532c189984daf53a9e184305b3b223dc3181a9695c0e03acd9c7d892a'
    assert digest(WHITE) == 'd8a20aa2b6c87868afa05c097cd30ae17eaed9f0048f6dfdf6ed56753b8c3400'
    white = json.loads(WHITE.read_text())
    meta = [json.loads(line) for line in META.read_text().splitlines() if line]
    assert len(meta) == len(white['row_id_whitelist']) == 1208
    assert {r['row_id'] for r in meta} == set(white['row_id_whitelist'])
    by_global = {r['original_global_index']: r for r in meta}
    assert len(by_global) == 1208
    assert all(r['component_role'] == 'TRAIN' and r['nominal_fit_eligible']
               and r['lineage_bound'] and r['context_origin'] == 'ORIGINAL_INITIALIZER'
               for r in meta)
    save('SCOPE_BEFORE_SELECTED_LABEL_BODY.json', {
        'purpose': 'Retrieve already-existing original-query weak correspondence at approved TRAIN indices; no new labels, training or model design.',
        'allowed_metadata': {'path': str(META), 'sha256': digest(META)},
        'whitelist': {'path': str(WHITE), 'sha256': digest(WHITE)},
        'mixed_ancestor': {'path': str(MIXED), 'previously_declared_sha256_NOT_rehashed': 'e5c37048b35fc4194b0ec22acb1c045e514a09c8355385d839e7edd6c96baf68'},
        'selector_before_JSON_decode': 'original global line ordinal from separately verified TRAIN-only metadata; ancestor producer requires line index == global_index',
        'allowed_decoded_rows': 1208,
        'excluded_line_bytes_traversed': True,
        'excluded_rows_JSON_decoded_or_labels_interpreted': False,
        'excluded_rows_NOT_zero_byte_access_claim': True,
        'current223_RGB_GT_features_scores_or_models': False,
        'V112_any_filesystem': False,
        'GT_nearest_matching_permutation_query_crop_or_U_change': False,
        'new_training_or_forward_or_PnP': False,
        'new_training_admission': False,
    })
    selected = []
    skipped = 0
    total = 0
    with guard(MIXED).open('rb') as f:
        for ordinal, raw in enumerate(f):
            total += 1
            if ordinal not in by_global:
                skipped += 1
                continue
            m = by_global[ordinal]
            r = json.loads(raw)
            assert r['global_index'] == ordinal
            assert r['frame_index'] == ordinal // 8 == m['original_frame_index']
            assert r['native_slot'] == ordinal % 8 == m['native_slot']
            assert r['query_id'] == m['row_id']
            assert r['source_id'] == m['source_id']
            assert r['split'] == 'SOURCE_DEVELOPMENT'
            assert r['query_xy'] == m['query_xy'] and r['L'] == m['L']
            assert r['original_input_valid'] == m['original_query_available']
            assert r['role_target'] in (-1, 0, 1)
            assert not r['human_gold'] and not r['exact_real_point_gold']
            assert not r['source_visibility_scores_used'] and not r['coordinates_or_native_order_changed']
            if r['role_target'] >= 0:
                assert r['entity_target'] == 1 and r['role_correspondence_supported']
                assert r['selected_gt_instance_index'] is not None
                assert r['role_state'] == ('D' if r['role_target'] == 0 else 'E')
            selected.append({
                'row_id': m['row_id'], 'original_global_index': ordinal,
                'component_id': m['component_id'], 'component_role': 'TRAIN',
                'context_origin': 'ORIGINAL_INITIALIZER',
                'original_q_L_native_availability_equal': True,
                'authority': 'EXISTING_ASSISTANT_WEAK_ORIGINAL_QUERY_ROLE; NOT human/native physical gold',
                'new_training_admission': False,
                'ancestor_record_unchanged': r,
            })
    assert total == 1840 and skipped == 632 and len(selected) == 1208
    assert {r['row_id'] for r in selected} == set(white['row_id_whitelist'])
    target = OUT / 'EXISTING_ORIGINAL_QUERY_TRAIN_WEAK_CORRESPONDENCE1208.jsonl'
    with guard(target).open('x') as f:
        for r in selected:
            f.write(json.dumps(r, ensure_ascii=False, allow_nan=False) + '\n')
    counts = collections.Counter()
    eligible = []
    by_native = collections.defaultdict(collections.Counter)
    for item in selected:
        r = item['ancestor_record_unchanged']
        counts['rows'] += 1
        counts['role_' + r['role_state']] += 1
        counts['entity_' + str(r['entity_target'])] += 1
        counts['available'] += int(r['original_input_valid'])
        counts['role_correspondence_supported'] += int(r['role_correspondence_supported'])
        by_native[str(r['native_slot'])]['role_' + r['role_state']] += 1
        if r['original_input_valid'] and r['role_target'] >= 0:
            eligible.append(item)
            by_native[str(r['native_slot'])]['available_' + r['role_state']] += 1
    summary = {
        'completed': True, 'PASS': True,
        'selected_labels': {'path': str(target), 'sha256': digest(target)},
        'counts': dict(counts), 'by_native': {k: dict(v) for k, v in sorted(by_native.items())},
        'eligible_existing_weak_D_E': len(eligible),
        'eligible_D_E_counts': dict(collections.Counter(r['ancestor_record_unchanged']['role_state'] for r in eligible)),
        'eligible_frames': len({r['ancestor_record_unchanged']['frame_index'] for r in eligible}),
        'eligible_declared_components': len({r['component_id'] for r in eligible}),
        'physical_independent_scene_count': 'UNKNOWN; components/sequences do not certify physical independence',
        'weak_correspondence_found_in_separate_ancestor': True,
        'absence_in_input_metadata_is_NOT_absence_of_existing_labels': True,
        'selected_native_quality_labels_all_unknown': all(r['ancestor_record_unchanged']['point_quality_label'] == -1 for r in selected),
        'all_q_L_native_input_availability_unchanged': True,
        'new_native_gold_created': False, 'new_training_admission': False,
        'mixed_ancestor_lines': total, 'excluded_records_unparsed': skipped,
        'excluded_records_raw_bytes_traversed': True,
        'excluded_record_label_values_used': False,
        'GT_images_depth_masks_array_model_forward_training_PnP': 0,
        'scope_caveat': 'Mixed ancestor bytes were traversed; excluded record bodies were not JSON-decoded or used. Not a zero-excluded-byte-access claim.',
        'conclusion_KO': '실제 초기query에 연결된 기존 assistant 약한 instance/native-role 감독은 존재한다. 입력metadata에 필드가 없는 것과 정답자료가 없는 것은 다르다. 다만 이것은 새 사람gold/물리점gold/좌표품질정답/독립검증을 제공하지 않고, 기존 학습에 미사용한 신규자료라고 주장하지 않는다.',
    }
    save('EXISTING_ORIGINAL_QUERY_TRAIN_WEAK_CORRESPONDENCE_SUMMARY.json', summary)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
