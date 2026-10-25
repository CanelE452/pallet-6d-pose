#!/usr/bin/env python3
"""V229 exact CURRENT input-only metadata closure; never opens image pointers."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat

R = Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility/publication_root_preparation')
H = R / 'V229_REAL_QA_B'
ENTRY = R / 'V229_SELECTED_FIXED_CONTEXT_REAL/INPUT_METADATA_ENTRY.json'
ENTRY_SHA = '6aeaec1e67c127c21579f47dd530ddda64b6ebeedfaaea3a180dec436e7f234f'
ROUTE = R / 'V229_REAL_INPUT_ROUTE_C/INPUT_ROUTE.json'
ROUTE_SHA = 'e0295176597d3e7271e98429b3364587fd16b72b46671d34596741f25c480fda'
PROTOCOL = H / 'INPUT_METADATA_PREPARATION_PROTOCOL.json'
PROTOCOL_SHA = '7a1727f2549b93bb8bd4d6d10746d74a76cc98e3926ecb743cbfe1bed9b2b39d'
OUT = H / 'current223_input_metadata_v1'
CURRENT = b'CURRENT223_1784'
HASH = re.compile(r'^[0-9a-f]{64}$')


def lexical(path):
    value = os.fspath(path)
    assert os.path.isabs(value), ('absolute path required', value)
    assert 'v112' not in value.lower(), 'V112 lexical rejection before FS'
    assert '..' not in Path(value).parts, 'parent traversal forbidden'
    return Path(value)


def no_symlinks(path):
    path = lexical(path)
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        info = os.lstat(current)
        assert not stat.S_ISLNK(info.st_mode), ('symlink', str(current))
    return path


def read_binding(binding):
    assert set(binding) <= {'path', 'sha256', 'size_bytes'}
    assert {'path', 'sha256'} <= set(binding)
    assert HASH.fullmatch(binding['sha256'])
    path = no_symlinks(binding['path'])
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, 'rb') as handle:
        info = os.fstat(handle.fileno())
        assert stat.S_ISREG(info.st_mode)
        data = handle.read()
    assert hashlib.sha256(data).hexdigest() == binding['sha256'], str(path)
    if 'size_bytes' in binding:
        assert type(binding['size_bytes']) is int and binding['size_bytes'] >= 0
        assert len(data) == binding['size_bytes']
    return data


def string_end(raw, start):
    assert raw[start] == 34
    i = start + 1
    while i < len(raw):
        if raw[i] == 92:
            i += 2
        elif raw[i] == 34:
            return i + 1
        else:
            i += 1
    raise AssertionError('unterminated JSON string')


def top_level_domain_bytes(raw):
    """Read only top-level domain marker; no json.loads or public value decoding."""
    depth = 0
    found = []
    i = 0
    while i < len(raw):
        c = raw[i]
        if c == 34:
            end = string_end(raw, i)
            if depth == 1:
                after = end
                while after < len(raw) and raw[after] in b' \t\r\n':
                    after += 1
                if after < len(raw) and raw[after] == 58 and raw[i:end] == b'"domain"':
                    start = after + 1
                    while start < len(raw) and raw[start] in b' \t\r\n':
                        start += 1
                    assert raw[start] == 34, 'domain must be string'
                    stop = string_end(raw, start)
                    marker = raw[start + 1:stop - 1]
                    assert b'\\' not in marker, 'escaped domain unsupported'
                    found.append(marker)
            i = end
            continue
        if c in (123, 91):
            depth += 1
        elif c in (125, 93):
            depth -= 1
            assert depth >= 0
        i += 1
    assert depth == 0 and len(found) == 1, 'exactly one top-level domain required'
    return found[0]


def same(a, b):
    if isinstance(a, (float, int)) and not isinstance(a, bool) and isinstance(b, (float, int)) and not isinstance(b, bool):
        return (math.isnan(a) and math.isnan(b)) or a == b
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(same(a[k], b[k]) for k in a)
    return type(a) is type(b) and a == b


def inert_image_binding(binding):
    assert isinstance(binding, dict)
    assert set(binding) <= {'path', 'sha256', 'size_bytes'}
    assert {'path', 'sha256'} <= set(binding)
    lexical(binding['path'])  # Deliberately no stat/open/hash of this path.
    assert HASH.fullmatch(binding['sha256'])
    if 'size_bytes' in binding:
        assert type(binding['size_bytes']) is int and binding['size_bytes'] >= 0


def validate_current(rows, groups):
    assert len(rows) == 1784 and len(groups) == 223
    assert len({r['row_id'] for r in rows}) == 1784
    assert len({g['group_id'] for g in groups}) == 223
    inner_checks = {}
    for fi, group in enumerate(groups):
        assert group['domain'] == CURRENT.decode()
        assert group['frame_index'] == fi
        assert group['group_id'] == 'CURRENT_FRAME_%04d' % fi
        assert group['native_slot'] == list(range(8))
        assert len(group['original_query_xy8']) == 8
        assert len(group['original_L8']) == 8
        assert len(group['original_query_available8']) == 8
        assert len(group['original_input_metadata8']) == 8
        assert len(group['raw_hw']) == 2
        assert all(type(v) is int and v > 0 for v in group['raw_hw'])
        assert HASH.fullmatch(group['decoded_RGB_sha256'])
        full = group['source_fullRGB']
        assert full['kind'] == 'ORIGINAL_ENCODED_RGB_SAFE_ALIAS'
        assert 'decoder' in full
        inert_image_binding(full['binding'])
        for k in range(8):
            index = fi * 8 + k
            row = rows[index]
            assert row['domain'] == CURRENT.decode()
            assert row['original_global_index'] == index
            assert row['frame_index'] == fi and row['native_slot'] == k
            assert row['id'] == group['id'] and row['group_id'] == group['group_id']
            assert row['row_id'] == str(group['id']) + ':native' + str(k)
            assert same(row['query_xy'], group['original_query_xy8'][k])
            assert same(row['L'], group['original_L8'][k])
            assert type(row['original_query_available']) is bool
            assert same(row['original_query_available'], group['original_query_available8'][k])
            assert same(row['actual_crop_grid_query_xy'], row['query_xy'])
            assert same(row['actual_crop_grid_L'], row['L'])
            if row['original_query_available']:
                assert len(row['query_xy']) == 2 and all(math.isfinite(v) for v in row['query_xy'])
                assert math.isfinite(row['L']) and row['L'] > 0
            original = group['original_input_metadata8'][k]
            assert isinstance(original, dict)
            for key in ('domain', 'group_id', 'frame_index', 'id', 'row_id', 'original_global_index', 'native_slot', 'query_xy', 'L', 'original_query_available'):
                if key in original:
                    assert same(original[key], row[key]), ('original metadata parity', fi, k, key)
                    inner_checks[key] = inner_checks.get(key, 0) + 1
    return inner_checks


def self_test():
    current = b'{"x":{"domain":"PUBLIC"},"domain":"CURRENT223_1784","x2":"escaped\\\"domain"}'
    excluded = b'{"domain":"PUBLIC45_360","opaque":"do not JSON parse values","nested":{"x":[1,2]}}'
    assert top_level_domain_bytes(current) == CURRENT
    assert top_level_domain_bytes(excluded) != CURRENT
    assert same([float('nan'), 1.25, None], [float('nan'), 1.25, None])
    assert not same([1.0, 2.0], [1.0, 2.00000001])
    try:
        lexical('/not-opened/V112/payload.npz')
        raise AssertionError('forbidden path accepted')
    except AssertionError as error:
        assert 'V112 lexical' in str(error)
    print(json.dumps({'invented_fixture_groups': 3, 'PASS': True}))


def write_json(path, obj):
    data = (json.dumps(obj, ensure_ascii=False, indent=2) + '\n').encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644)
    with os.fdopen(fd, 'wb') as handle:
        handle.write(data)
    return {'path': str(path), 'sha256': hashlib.sha256(data).hexdigest(), 'size_bytes': len(data)}


def run(entry_path):
    assert lexical(entry_path) == ENTRY
    entry_binding = {'path': str(ENTRY), 'sha256': ENTRY_SHA}
    entry = json.loads(read_binding(entry_binding))
    assert entry['authorized_now'] is True and entry['new_candidate_changes'] is False
    assert entry['candidate_protocol_sha256'] == 'be3a9d24cee1b18458d3214bbd20833a37786af55d0c9bd2f16ed379129f6c41'
    assert entry['input_route_sha256'] == ROUTE_SHA
    protocol_binding = {'path': str(PROTOCOL), 'sha256': PROTOCOL_SHA}
    read_binding(protocol_binding)
    route_binding = {'path': str(ROUTE), 'sha256': ROUTE_SHA}
    route = json.loads(read_binding(route_binding))
    artifacts = route['current_input_artifacts_for_future_exact_entry']
    allowed = entry['allowed_metadata']
    expected = [{key: artifacts[name][key] for key in ('path', 'sha256')} for name in ('rows1784', 'frame_groups')]
    assert allowed == expected and len(allowed) == 2
    raw_rows, raw_groups = [read_binding(binding) for binding in allowed]
    row_lines = raw_rows.splitlines()
    group_lines = raw_groups.splitlines()
    assert len(row_lines) == 1784 and len(group_lines) == 268
    rows = []
    for line in row_lines:
        assert top_level_domain_bytes(line) == CURRENT
        rows.append(json.loads(line))
    groups = []
    excluded_markers = {}
    excluded_bytes = 0
    for line in group_lines:
        marker = top_level_domain_bytes(line)
        if marker == CURRENT:
            groups.append(json.loads(line))
        else:
            excluded_markers[marker.decode('ascii')] = excluded_markers.get(marker.decode('ascii'), 0) + 1
            excluded_bytes += len(line)
    assert sum(excluded_markers.values()) == 45
    inner_checks = validate_current(rows, groups)
    available = sum(row['original_query_available'] for row in rows)
    source_paths = {group['source_fullRGB']['binding']['path'] for group in groups}
    source_digests = {group['source_fullRGB']['binding']['sha256'] for group in groups}
    manifest = {
        'completed': True, 'PASS': True, 'phase': 'V229', 'domain': CURRENT.decode(),
        'entry': entry_binding, 'protocol': protocol_binding, 'input_route': route_binding,
        'source_metadata': allowed, 'frames223': groups, 'rows1784': rows,
        'original_input_order_q_L_native_availability_preserved': True,
        'image_bindings_are_inert_recorded_metadata_only': True,
        'GT_or_score_features_added': False, 'candidate_or_threshold_changed': False,
        'input_counts': {'frames': 223, 'rows': 1784, 'available': available, 'original_U': 1784 - available,
                         'unique_source_image_paths': len(source_paths), 'unique_recorded_source_SHA': len(source_digests)},
    }
    no_symlinks(H)
    os.mkdir(OUT)  # Fresh only, after validation; never retry or overwrite.
    manifest_binding = write_json(OUT / 'INPUT_MANIFEST_CURRENT223_ONLY.json', manifest)
    qa = {
        'completed': True, 'PASS': True, 'manifest': manifest_binding, 'entry': entry_binding,
        'protocol': protocol_binding, 'input_route': route_binding, 'source_metadata': allowed,
        'counts': manifest['input_counts'], 'all_native8_per_frame_in_original_order': True,
        'rows_original_global_indices_exact_0_through_1783': True,
        'row_id_id_native_group_join_exact': True, 'original_q8_L8_availability8_and_crop_grid_exact': True,
        'group_original_input_metadata8_matching_keys_checked': inner_checks,
        'fullRGB_binding_decoder_raw_hw_decoded_SHA_schema_validated_not_payload': True,
        'mixed_metadata_access': {'raw_lines_traversed_for_SHA': 268, 'current_lines_JSON_parsed': 223,
                                  'excluded_domain_marker_counts': excluded_markers,
                                  'excluded_values_JSON_parsed': 0, 'excluded_nonnewline_bytes_traversed': excluded_bytes,
                                  'excluded_bytes_read_zero_claim': False},
        'access': {'approved_input_metadata_files': 2, 'image_pointer_stat_hash_or_body': 0,
                   'GT_annotations_reference_score_body': 0, 'NPZ_model_weights_body': 0,
                   'fit_forward_PnP_threshold_or_model_selection': 0, 'broad_scan': 0,
                   'V112_filesystem': 0, 'original_files_modified': False},
        'limitations': ['Recorded RGB opaque/decoded hashes and dimensions are metadata-only, not fresh source image verification.',
                        'Original input U remains input availability; no trained confidence, new D/E target, or truth-based recovery.',
                        'PUBLIC45 raw bytes were traversed for the mixed-container SHA and domain markers; their JSON values were not parsed.'],
    }
    qa_binding = write_json(OUT / 'INPUT_METADATA_QA.json', qa)
    print(json.dumps({'completed': True, 'PASS': True, 'manifest': manifest_binding,
                      'QA': qa_binding, 'counts': manifest['input_counts']}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--entry')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        assert args.entry, '--entry required'
        run(args.entry)
