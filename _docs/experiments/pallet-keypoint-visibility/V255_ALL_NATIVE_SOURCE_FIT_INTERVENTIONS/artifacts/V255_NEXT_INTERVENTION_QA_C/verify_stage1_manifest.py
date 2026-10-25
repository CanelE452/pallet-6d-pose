"""Independent fixed V255 Stage1 metadata replay; never import the producer."""
import collections
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import struct

R = Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility/publication_root_preparation')
H = R / 'V255_NEXT_INTERVENTION_QA_C'
COMPLETE = {'path': str(R / 'V255_ROOT_NATIVE_COVERAGE/source_fit_parent_manifest_v1/COMPLETE.json'), 'sha256': '95b61a1c4c11ce36d68971bbab0e15a61757201e16fb306057ee3c3e529e1724', 'size_bytes': 4914}
PROTOCOL = {'path': str(R / 'V255_ROOT_NATIVE_COVERAGE/PROTOCOL_BEFORE_SOURCE_FIT_CONFIG_SELECTION.json'), 'sha256': '56c8afad8c72129526bbbfe357a8c6033a39abe499ff70f87a0a32ec338b4349', 'size_bytes': 5534}
CONFIG_ORIGINAL = '/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility/query_visibility_registered_native_diversity_20261023_v129/render_preparation/BASE_CONFIGS192_BEFORE_RGB.json'


def guarded(path):
    p = Path(path)
    assert p.is_absolute() and '..' not in p.parts
    assert not any(t in str(p).casefold() for t in ('v112', 'cache', '.pyc'))
    for a in [*reversed(p.parents), p]:
        assert not stat.S_ISLNK(os.lstat(a).st_mode), str(a)
    return p


def read(binding):
    p = guarded(binding['path'])
    with os.fdopen(os.open(p, os.O_RDONLY | os.O_NOFOLLOW), 'rb') as f:
        b = f.read()
    assert len(b) == binding['size_bytes'] and hashlib.sha256(b).hexdigest() == binding['sha256']
    return b


def ws(raw, i):
    while i < len(raw) and raw[i] in b' \r\n\t':
        i += 1
    return i


def quoted_end(raw, i):
    assert raw[i] == 34
    i += 1
    escaped = False
    while i < len(raw):
        c = raw[i]
        if escaped:
            escaped = False
        elif c == 92:
            escaped = True
        elif c == 34:
            return i + 1
        i += 1
    raise AssertionError('Unclosed string')


def opaque_end(raw, i):
    """Boundary scan only; excluded strings/numbers are never decoded."""
    i = ws(raw, i)
    if raw[i] == 34:
        return quoted_end(raw, i)
    if raw[i] not in (91, 123):
        j = i
        while j < len(raw) and raw[j] not in b',]} \r\n\t':
            j += 1
        assert j > i
        return j
    stack = [93 if raw[i] == 91 else 125]
    j = i + 1
    while stack:
        c = raw[j]
        if c == 34:
            j = quoted_end(raw, j)
            continue
        if c in (91, 123):
            stack.append(93 if c == 91 else 125)
        elif c in (93, 125):
            assert c == stack.pop()
        j += 1
    return j


def independent_rows_spans(raw):
    i = ws(raw, 0)
    assert raw[i] == 123
    i = ws(raw, i + 1)
    result = None
    while raw[i] != 125:
        e = quoted_end(raw, i)
        literal_key = raw[i:e]
        i = ws(raw, e)
        assert raw[i] == 58
        i = ws(raw, i + 1)
        end = opaque_end(raw, i)
        if literal_key == b'"rows"':
            assert result is None and raw[i] == 91
            result = []
            j = ws(raw, i + 1)
            while raw[j] != 93:
                k = opaque_end(raw, j)
                result.append((j, k))
                j = ws(raw, k)
                if raw[j] == 44:
                    j = ws(raw, j + 1)
                else:
                    assert raw[j] == 93
            assert j + 1 == end
        i = ws(raw, end)
        if raw[i] == 44:
            i = ws(raw, i + 1)
        else:
            assert raw[i] == 125
    assert result is not None and ws(raw, i + 1) == len(raw)
    return result


def fp(x):
    if x is None:
        return b'N'
    if isinstance(x, bool):
        return b'B' + bytes([x])
    if isinstance(x, list):
        return b'L' + struct.pack('<Q', len(x)) + b''.join(map(fp, x))
    assert isinstance(x, (int, float))
    return b'F' + struct.pack('<d', x)


def bv(x):
    assert x in ('True', 'False')
    return x == 'True'


def run():
    complete = json.loads(read(COMPLETE))
    protocol = json.loads(read(PROTOCOL))
    assert complete['completed'] is True and complete['PASS'] is True
    assert complete['inputs'] == protocol['inputs']
    expected_outputs = {'CENSUS.json', 'SELECTED_CONFIG_ROW_SPANS.json', 'SELECTED_FIT_CONFIGS.json', 'SELECTION_FAILURES.json', 'SOURCE_FIT_PARENT_MANIFEST.json'}
    assert set(complete['outputs']) == expected_outputs
    actual_dir = Path(COMPLETE['path']).parent
    for name, binding in complete['outputs'].items():
        assert Path(binding['path']) == actual_dir / name
    outputs = {k: json.loads(read(v)) for k, v in complete['outputs'].items()}
    raw = {k: read(v) for k, v in complete['inputs'].items()}
    assert set(raw) == {'syn7040_rows', 'groups', 'scenes', 'configs192_opaque'}
    rows = list(csv.DictReader(io.StringIO(raw['syn7040_rows'].decode())))
    assert len(rows) == 7040 and len({r['row_id'] for r in rows}) == len(rows)
    roles, selected = collections.defaultdict(set), collections.defaultdict(list)
    role_counts = collections.Counter()
    for row in rows:
        assert row['sample_family'] == '0'
        fit, base = bv(row['fit']), int(row['base_index'])
        roles[base].add(fit)
        role_counts['FIT' if fit else 'DEV'] += 1
        if fit and row['augmentation_family'] == 'BEFORE_ANCHOR4' and row['variant'] == 'BEFORE':
            selected[base].append(row)
    groups = [json.loads(line) for line in raw['groups'].splitlines() if line]
    group_map = {g['group_index']: g for g in groups}
    assert len(group_map) == len(groups)
    scene_spans = independent_rows_spans(raw['scenes'])
    config_spans = independent_rows_spans(raw['configs192_opaque'])
    assert len(config_spans) == 192
    selected_scene_ids = sorted({int(r['scene_index']) for v in selected.values() for r in v})
    scene = {i: json.loads(raw['scenes'][slice(*scene_spans[i])]) for i in selected_scene_ids}
    config_ids = sorted({s['source_base_config']['row_index'] for s in scene.values()})
    # This is the sole decode of records in the original mixed config container.
    config = {i: json.loads(raw['configs192_opaque'][slice(*config_spans[i])]) for i in config_ids}
    spans_out = outputs['SELECTED_CONFIG_ROW_SPANS.json']
    def span_record(i, pair, source):
        a, b = pair
        return {'row_index': i, 'byte_start': a, 'byte_end_exclusive': b,
                'size_bytes': b-a, 'row_sha256': hashlib.sha256(source[a:b]).hexdigest()}
    assert spans_out['rows'] == [span_record(i, config_spans[i], raw['configs192_opaque']) for i in config_ids]
    assert spans_out['selected_scene_spans'] == [span_record(i, scene_spans[i], raw['scenes']) for i in selected_scene_ids]
    assert spans_out['source_sha256'] == complete['inputs']['configs192_opaque']['sha256']
    assert spans_out['source_size_bytes'] == len(raw['configs192_opaque']) and spans_out['excluded_record_values_emitted'] is False
    parent_doc = outputs['SOURCE_FIT_PARENT_MANIFEST.json']
    parents = {p['base_index']: p for p in parent_doc['parents']}
    assert len(parents) == len(selected) == len(config_ids)
    assert set(parents) == set(selected)
    selected_cfg_rows = outputs['SELECTED_FIT_CONFIGS.json']['rows']
    assert selected_cfg_rows == [config[parents[b]['config_row_index']] for b in sorted(selected)]
    unavailable = masked = null_q_slots = 0
    for base, members in sorted(selected.items()):
        p = parents[base]
        assert roles[base] == {True}
        assert collections.Counter(int(r['native_slot']) for r in members) == {k: 1 for k in range(8)}
        assert p['identity_PASS'] is True and p['failure_reasons'] == [] and p['source_base_role'] == 'FIT'
        s = scene[4*base]
        assert s['scene_index'] == 4*base and s['base_index'] == base
        assert s['base_id'] == 'V129_BASE_%03d' % base and s['variant'] == 'BEFORE' and s['split'] == 'TRAIN'
        decl = s['source_base_config']
        assert decl['file']['path'] == CONFIG_ORIGINAL
        assert decl['file']['sha256'] == complete['inputs']['configs192_opaque']['sha256']
        cfg = config[decl['row_index']]
        assert cfg['base_index'] == base and cfg['base_id'] == s['base_id'] and cfg['split'] == 'TRAIN'
        assert sorted(cfg['permCF_to_canonical']) == list(range(8))
        assert p['config_row_index'] == decl['row_index']
        assert p['selected_config_span'] == span_record(decl['row_index'], config_spans[decl['row_index']], raw['configs192_opaque'])
        g0 = group_map[int(members[0]['group_index'])]
        keys = ('q8', 'L8', 'original_available8', 'box_xyxy', 'raw_hw')
        for key in keys:
            assert fp(p['before_input'][key]) == fp(g0[key])
        assert p['before_input']['group_id'] == g0['group_id']
        assert p['before_input']['group_index'] == g0['group_index']
        assert p['before_input']['image'] == g0['image']
        assert p['before_input']['expected_BEFORE_encoded_RGB_binding'] == s['full_RGB']
        pixel_sha = s['decoded_RGB_sha256']
        assert len(pixel_sha) == 64 and all(c in '0123456789abcdef' for c in pixel_sha)
        assert p['before_input']['expected_BEFORE_decoded_RGB_sha256'] == pixel_sha
        assert p['before_input']['render_raw_hw_declared_separately'] == s['raw_hw']
        assert p['before_input']['originalpermCF_to_canonical'] == cfg['permCF_to_canonical']
        expected_members = []
        for row in members:
            k = int(row['native_slot'])
            g = group_map[int(row['group_index'])]
            assert all(fp(g[key]) == fp(g0[key]) for key in keys)
            assert g['group_id'] == row['group_id'] and g['family_code'] == 0
            assert g['image']['sha256'] == row['image_sha256'] == s['full_RGB']['sha256']
            assert g['image']['path'] == s['full_RGB']['path'] and g['image']['member'] == s['full_RGB_key']
            assert int(row['scene_index']) == int(row['actual_image_scene_index']) == int(row['anchor_scene_index']) == 4*base
            assert int(row['declared_CF_to_canonical']) == cfg['permCF_to_canonical'][k]
            assert bv(row['original_query_available']) == g['original_available8'][k]
            expected_members.append({'row_id': row['row_id'], 'original_source_index': int(row['source_index']),
                'original_sample_index': None if row['sample_index'] == '' else int(row['sample_index']),
                'native_slot': k, 'declared_CF_to_canonical': cfg['permCF_to_canonical'][k],
                'original_query_available': bv(row['original_query_available']), 'original_crop_available': bv(row['crop_available']),
                'group_id': row['group_id'], 'group_index': int(row['group_index']), 'original_known_or_masked_retained': True})
        assert p['native_members'] == sorted(expected_members, key=lambda m: (m['native_slot'], m['row_id']))
        u = sum(not bv(m['original_query_available']) for m in members)
        m = sum(m['sample_index'] == '' for m in members)
        assert p['all_original_parent_rows'] == 8 and p['unavailable_native_members_retained'] == u and p['masked_native_members_retained'] == m
        unavailable += u
        masked += m
        null_q_slots += sum(q is None for q in g0['q8'])
    assert outputs['SELECTION_FAILURES.json']['rows'] == [] and parent_doc['all_parent_identity_PASS'] is True
    counts = {'all_original_SYN_rows': len(rows), 'original_base_count': len(roles),
        'selected_original_FIT_base_count': len(selected), 'selected_original_FIT_beforeanchor_BEFORE_members': sum(map(len, selected.values())),
        'role_counts': dict(role_counts), 'masked_original_sample_index_absent': sum(r['sample_index'] == '' for r in rows),
        'all_original_groups_input_metadata_JSON_decoded': len(groups), 'all_config_records_lexically_traversed': len(config_spans),
        'selected_config_records_JSON_decoded': len(config), 'excluded_config_records_lexically_skipped': len(config_spans)-len(config),
        'all_scene_records_lexically_traversed': len(scene_spans), 'selected_scene_records_JSON_decoded': len(scene),
        'excluded_config_records_JSON_decoded': 0, 'excluded_config_records_numeric_decoded': 0, 'nonselected_scene_records_JSON_decoded': 0,
        'parent_failure_count': 0, 'all_parent_identity_PASS': True}
    for k, v in counts.items():
        assert outputs['CENSUS.json'][k] == complete['census'][k] == v, k
    assert complete['nonselected_config_JSON_or_numeric_decode'] == 0
    script = guarded(__file__).read_bytes()
    review = {'completed': True, 'PASS': True, 'stage1_complete': COMPLETE, 'root_protocol': PROTOCOL,
        'verifier': {'path': str(Path(__file__)), 'sha256': hashlib.sha256(script).hexdigest(), 'size_bytes': len(script)},
        'inputs': complete['inputs'], 'outputs_verified': complete['outputs'], 'independent_counts': counts,
        'selected_native_members_unavailable_retained': unavailable, 'selected_native_members_masked_retained': masked,
        'selected_before_q8_null_slots': null_q_slots, 'input_identity_nullable_FP64_bool_comparison': 'exact bytes, no tolerance',
        'all_selected_role_before_q_native_mapping_declared_config_span_expected_pixel_hash_checks': True,
        'SOURCE_DEV_CSV_strings_and_input_only_group_metadata_read': True,
        'SOURCE_DEV_labels_predictions_scores_used_or_numeric_converted': False,
        'mixed_original_config_and_scene_containers_opaque_hashed_and_lexically_scanned': True,
        'nonselected_original_config_or_scene_records_JSON_or_numeric_decoded': 0,
        'selected_FIT_procedural_config_metadata_decoded': len(config),
        'raw_mesh_RGB_GT_archives_NPZ_arrays_weights_models_current_or_held_payload_reads': 0,
        'embedded_pointer_FS_follow': False, 'producer_import_or_call': False, 'new_selection_or_tuning': False,
        'scope': 'Stage1 metadata identity only; expected decoded RGB SHA binding checked, no rendered pixel parity assertion.',
        'execution_order_evidence': 'Frozen CODE preflight plus saved COMPLETE/outputs; not retrospective proof of process temporal behavior.'}
    out = H / 'actual_review_v1'
    assert not any(t in str(out).casefold() for t in ('v112', 'cache', '.pyc'))
    guarded(out.parent)
    out.mkdir()
    data = (json.dumps(review, ensure_ascii=False, sort_keys=True, indent=2) + '\n').encode()
    with os.fdopen(os.open(out / 'REVIEW.json', os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644), 'wb') as f:
        f.write(data)
    print(json.dumps({'completed': True, 'PASS': True, 'review': {'path': str(out / 'REVIEW.json'), 'sha256': hashlib.sha256(data).hexdigest(), 'size_bytes': len(data)}, 'counts': counts, 'selected_unavailable': unavailable, 'selected_masked': masked, 'q8_null_slots': null_q_slots}, sort_keys=True))


if __name__ == '__main__':
    run()
