"""Retrieve only existing reference rows at approved original TRAIN identities.

Excluded binary line bytes are traversed and identity headers are read for
selection; excluded geometry and labels are never JSON-decoded/interpreted.
"""
from pathlib import Path
import hashlib
import json
import re

P = Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility')
R = P / 'publication_root_preparation/V226_ROOT_ALIGNMENT'
SOURCE = P / 'publication_root_preparation/V224_ROOT_READINESS/EXISTING_ORIGINAL_QUERY_TRAIN_WEAK_CORRESPONDENCE1208.jsonl'
REFERENCE = P / 'query_visibility_mr6d_full_public_source_20261024_v153/native_reference_preparation/provided_reference_v1/ALL2424_DYNAMIC_NATIVE_REFERENCE_GT_ONLY.jsonl'


def guard(path):
    assert '_v112' not in str(path).lower()
    assert str(path).startswith(str(P) + '/')
    return path


def save(name, obj):
    with guard(R / name).open('x') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def main():
    for p in (R, SOURCE, REFERENCE):
        guard(p)
    raw = SOURCE.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == '581a5b2f71f28c7cdbd7a5b313d0f8ded586ed6e498e484344beb18376d26451'
    rows = [json.loads(s) for s in raw.splitlines()]
    requested = {}
    for item in rows:
        a = item['ancestor_record_unchanged']
        assert item['component_role'] == 'TRAIN'
        if not (a['original_input_valid'] and a['role_state'] in ('D', 'E')):
            continue
        assert a['entity_target'] == 1 and a['role_correspondence_supported']
        assert a['selected_gt_instance_index'] is not None
        key = (a['frame_index'], a['selected_gt_instance_index'], a['native_slot'])
        assert key not in requested
        requested[key] = item
    assert len(requested) == 368
    scope = {
        'purpose': 'Existing provided-reference/original-query/crop alignment diagnostic; not native physical point quality or new labels.',
        'source_TRAIN_only': {'path': str(SOURCE), 'sha256': hashlib.sha256(raw).hexdigest()},
        'reference': {'path': str(REFERENCE), 'previously_declared_SHA_NOT_rehashed': 'd86eede90279dbbee5dbc3e87ffbb5cfcb2a7005fb429f9a6edbf5f91e7e7879'},
        'selector_before_geometry_JSON_interpretation': 'Exact(frame_index, selected_gt_instance_index, native_slot) already supported in approved TRAIN-only weak metadata.',
        'requested_reference_rows': 368,
        'requested_identities': [{'frame_index': k[0], 'gt_instance_index': k[1], 'native_slot': k[2], 'row_id': v['row_id']} for k, v in sorted(requested.items())],
        'excluded_raw_line_bytes_and_three_identity_headers_traversed': True,
        'excluded_geometry_or_label_JSON_interpreted': False,
        'whole_mixed_file_hash_or_reference_pointer_follow': False,
        'no_GT_nearest_or_permutation_search_or_query_crop_U_change': True,
        'no_current223_image_GT_feature_score_model_or_V112_filesystem': True,
        'new_training_forward_PnP': 0,
        'new_training_admission': False,
    }
    save('SCOPE_BEFORE_SELECTED_REFERENCE_BODY.json', scope)
    header_keys = ('frame_index', 'gt_instance_index', 'native_slot')
    patterns = [re.compile(rb'"' + k.encode() + rb'"\s*:\s*(-?\d+)') for k in header_keys]
    selected = []
    seen = set()
    total = 0
    excluded = 0
    with REFERENCE.open('rb') as f:
        for rawline in f:
            total += 1
            headers = [pattern.search(rawline) for pattern in patterns]
            assert all(x is not None for x in headers)
            key = tuple(int(x.group(1)) for x in headers)
            if key not in requested:
                excluded += 1
                continue
            ref = json.loads(rawline)
            assert tuple(ref[k] for k in header_keys) == key and key not in seen
            item = requested[key]
            seen.add(key)
            selected.append({'row_id': item['row_id'], 'component_id': item['component_id'],
                             'component_role': 'TRAIN', 'original_query_weak_record_unchanged': item['ancestor_record_unchanged'],
                             'provided_reference_record_unchanged': ref,
                             'authority': 'Provided annotation reference and assistant weak role correspondence; not exact physical native gold.',
                             'new_training_admission': False})
    assert total == 2424 and excluded == 2056 and seen == set(requested)
    dest = R / 'SELECTED_TRAIN_REFERENCE368.jsonl'
    with dest.open('x') as f:
        for row in sorted(selected, key=lambda x: x['original_query_weak_record_unchanged']['global_index']):
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
    summary = {'completed': True, 'PASS': True,
               'selected': {'path': str(dest), 'sha256': hashlib.sha256(dest.read_bytes()).hexdigest()},
               'reference_rows_streamed': total, 'selected_reference_bodies_JSON_decoded': len(selected),
               'excluded_reference_bodies_not_JSON_decoded': excluded,
               'excluded_raw_bytes_and_identity_headers_traversed': True,
               'selected_reference_keys': sorted(selected[0]['provided_reference_record_unchanged']),
               'GT_nearest_permutation_or_query_correction': False,
               'new_training_forward_PnP': 0, 'new_training_admission': False}
    save('REFERENCE_SELECTION_SUMMARY.json', summary)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
