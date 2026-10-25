"""Select immutable original SOURCE FIT parent configs; never render or load a model."""
import argparse
import collections
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re
import struct

PROTOCOL_SHA = '56c8afad8c72129526bbbfe357a8c6033a39abe499ff70f87a0a32ec338b4349'
INPUT_KEYS = ('syn7040_rows', 'groups', 'scenes', 'configs192_opaque')
ORIGINAL_CONFIG_SOURCE_PATH = '/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility/query_visibility_registered_native_diversity_20261023_v129/render_preparation/BASE_CONFIGS192_BEFORE_RGB.json'
NUMBER = re.compile(rb'-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?')
WS = b' \t\r\n'


def guard(path, fresh=False):
    p = Path(path)
    if not p.is_absolute() or any(s == '..' or any(t in s.lower() for t in ('v112', 'cache', '.pyc')) for s in p.parts):
        raise ValueError('Forbidden artifact path')
    for a in reversed((p,) + tuple(p.parents)):
        if a.is_symlink():
            raise ValueError('Symlink artifact/ancestor')
    if fresh and p.exists():
        raise ValueError('Output must be fresh')
    return p


def sha(data):
    return hashlib.sha256(data).hexdigest()


def pin(path):
    p = guard(path)
    b = p.read_bytes()
    return {'path': str(p), 'sha256': sha(b), 'size_bytes': len(b)}


def checked(binding):
    b = guard(binding['path']).read_bytes()
    assert len(b) == binding['size_bytes'] and sha(b) == binding['sha256']
    return b


def skip_ws(raw, i):
    while i < len(raw) and raw[i] in WS:
        i += 1
    return i


def string_end(raw, i):
    if i >= len(raw) or raw[i] != 34:
        raise ValueError('Expected JSON string')
    i += 1
    while i < len(raw):
        c = raw[i]
        if c == 34:
            return i + 1
        if c < 32:
            raise ValueError('Control byte in string')
        if c == 92:
            i += 1
            if i >= len(raw):
                raise ValueError('Truncated escape')
            if raw[i] == 117:
                if i + 4 >= len(raw) or any(c not in b'0123456789abcdefABCDEF' for c in raw[i + 1:i + 5]):
                    raise ValueError('Invalid Unicode escape')
                i += 4
            elif raw[i] not in b'"\\/bfnrt':
                raise ValueError('Invalid JSON escape')
        i += 1
    raise ValueError('Unterminated string')


def value_end(raw, i, depth=0):
    """Lexical JSON traversal only: no float/int/string/object value decoding."""
    if depth > 96:
        raise ValueError('Excessive JSON nesting')
    i = skip_ws(raw, i)
    if i >= len(raw):
        raise ValueError('Missing value')
    c = raw[i]
    if c == 34:
        return string_end(raw, i)
    if c in (123, 91):
        close = 125 if c == 123 else 93
        i = skip_ws(raw, i + 1)
        if i < len(raw) and raw[i] == close:
            return i + 1
        while True:
            if c == 123:
                i = skip_ws(raw, string_end(raw, i))
                if i >= len(raw) or raw[i] != 58:
                    raise ValueError('Missing object colon')
                i += 1
            i = skip_ws(raw, value_end(raw, i, depth + 1))
            if i >= len(raw):
                raise ValueError('Unclosed container')
            if raw[i] == close:
                return i + 1
            if raw[i] != 44:
                raise ValueError('Missing comma')
            i = skip_ws(raw, i + 1)
    for literal in (b'true', b'false', b'null'):
        if raw.startswith(literal, i):
            return i + len(literal)
    match = NUMBER.match(raw, i)
    if match is None:
        raise ValueError('Invalid JSON literal')
    return match.end()


def array_spans(raw, i):
    if i >= len(raw) or raw[i] != 91:
        raise ValueError('rows must be an array')
    result = []
    i = skip_ws(raw, i + 1)
    if raw[i] == 93:
        return result, i + 1
    while True:
        start = i
        end = value_end(raw, i)
        result.append((start, end))
        i = skip_ws(raw, end)
        if i >= len(raw):
            raise ValueError('Truncated rows array')
        if raw[i] == 93:
            return result, i + 1
        if raw[i] != 44:
            raise ValueError('Missing rows comma')
        i = skip_ws(raw, i + 1)


def row_spans(raw):
    """Only the pinned writer's literal top-level key \"rows\" is admitted."""
    i = skip_ws(raw, 0)
    if i >= len(raw) or raw[i] != 123:
        raise ValueError('Expected top-level object')
    i = skip_ws(raw, i + 1)
    spans = None
    seen_keys = set()
    while i < len(raw) and raw[i] != 125:
        end = string_end(raw, i)
        key = raw[i:end]
        if key in seen_keys:
            raise ValueError('Duplicate top-level key')
        seen_keys.add(key)
        i = skip_ws(raw, end)
        if i >= len(raw) or raw[i] != 58:
            raise ValueError('Missing top-level colon')
        i = skip_ws(raw, i + 1)
        if key == b'"rows"':
            spans, i = array_spans(raw, i)
        else:
            i = value_end(raw, i)
        i = skip_ws(raw, i)
        if i >= len(raw):
            raise ValueError('Truncated top-level object')
        if raw[i] == 125:
            break
        if raw[i] != 44:
            raise ValueError('Missing top-level comma')
        i = skip_ws(raw, i + 1)
        if i >= len(raw) or raw[i] == 125:
            raise ValueError('Trailing top-level comma')
    if i >= len(raw) or raw[i] != 125 or skip_ws(raw, i + 1) != len(raw) or spans is None:
        raise ValueError('Invalid rows wrapper/trailing bytes')
    return spans


def strict_loads(data):
    def pairs(items):
        out = {}
        for k, v in items:
            if k in out:
                raise ValueError('Duplicate selected-record key')
            out[k] = v
        return out
    def reject(s):
        raise ValueError('Nonstandard JSON constant')
    value = json.loads(data, object_pairs_hook=pairs, parse_constant=reject)
    def finite(v):
        if isinstance(v, float) and not math.isfinite(v):
            raise ValueError('Nonfinite selected JSON number')
        if isinstance(v, dict):
            for x in v.values(): finite(x)
        elif isinstance(v, list):
            for x in v: finite(x)
    finite(value)
    return value


def select_json_rows(raw, indices, decoder=strict_loads):
    """The sole record decoder is called once per selected span, never elsewhere."""
    ids = list(indices)
    if any(type(i) is not int or i < 0 for i in ids) or len(set(ids)) != len(ids):
        raise ValueError('Unique nonnegative row indices required')
    spans = row_spans(raw)
    if any(i >= len(spans) for i in ids):
        raise ValueError('Selected row index out of range')
    selected, provenance = {}, []
    for i in sorted(ids):
        start, end = spans[i]
        record = raw[start:end]
        selected[i] = decoder(record)
        provenance.append({'row_index': i, 'byte_start': start, 'byte_end_exclusive': end,
                           'size_bytes': end - start, 'row_sha256': sha(record)})
    return selected, provenance, len(spans)


def bval(value):
    if value == 'True': return True
    if value == 'False': return False
    raise ValueError('Exact source CSV boolean namespace')


def fp(value):
    if value is None: return b'N'
    if isinstance(value, list): return b'L' + struct.pack('<I', len(value)) + b''.join(fp(v) for v in value)
    if isinstance(value, bool): return b'B' + bytes([value])
    if not isinstance(value, (float, int)): raise ValueError('Numeric/boolean/null input only')
    return b'F' + struct.pack('<d', float(value))


def select_parent_rows(rows):
    assert len(rows) == 7040 and len({r['row_id'] for r in rows}) == 7040
    role = collections.Counter()
    role_by_base = collections.defaultdict(set)
    selected = collections.defaultdict(list)
    for r in rows:
        assert r['sample_family'] == '0'
        fit = bval(r['fit'])
        role['FIT' if fit else 'DEV'] += 1
        base = int(r['base_index'])
        role_by_base[base].add(fit)
        if fit and r['augmentation_family'] == 'BEFORE_ANCHOR4' and r['variant'] == 'BEFORE':
            selected[base].append(r)
    return selected, role_by_base, {
        'all_original_SYN_rows': len(rows), 'role_counts': dict(sorted(role.items())),
        'masked_original_sample_index_absent': sum(r['sample_index'] == '' for r in rows),
        'original_base_count': len(role_by_base), 'selected_original_FIT_base_count': len(selected),
        'selected_original_FIT_beforeanchor_BEFORE_members': sum(map(len, selected.values())),
        'selection_uses_only_original_fit_and_fixed_family_variant': True,
        'prediction_Elogit_reference_or_label_fields_used_or_numeric_converted': 0,
        'SOURCE_DEV_CSV_strings_encountered_for_container_role_census': True,
        'SOURCE_DEV_predictions_labels_scores_used': False,
    }


def make_manifest(rows, groups, scene_records, configs, spans, role_by_base, selected, config_binding):
    group_map = {g['group_index']: g for g in groups}
    assert len(group_map) == len(groups)
    scenes = {s['scene_index']: s for s in scene_records.values()}
    assert len(scenes) == len(scene_records)
    span_map = {s['row_index']: s for s in spans}
    parents, failures = [], []
    config_rows = []
    for base, members in sorted(selected.items()):
        reasons = []
        slots = collections.Counter(int(r['native_slot']) for r in members)
        if slots != {k: 1 for k in range(8)}: reasons.append('MISSING_OR_DUPLICATE_NATIVE_MEMBERSHIP')
        if role_by_base[base] != {True}: reasons.append('BASE_CROSSES_ORIGINAL_FIT_DEV_ROLE')
        declarations = []
        old_groups = []
        member_out = []
        for r in members:
            k = int(r['native_slot'])
            g = group_map.get(int(r['group_index']))
            s = scenes.get(int(r['scene_index']))
            if g is None or s is None:
                reasons.append('MISSING_DECLARED_GROUP_OR_SCENE')
                continue
            if not 0 <= k < 8 or int(r['scene_index']) != 4 * base or int(r['actual_image_scene_index']) != 4 * base or int(r['anchor_scene_index']) != 4 * base:
                reasons.append('BEFORE_ACTUAL_ANCHOR_SCENE_IDENTITY')
            if g['group_id'] != r['group_id'] or g['family_code'] != 0 or g['image']['sha256'] != r['image_sha256']:
                reasons.append('ORIGINAL_GROUP_IDENTITY')
            full = s.get('full_RGB', {})
            if s.get('base_index') != base or s.get('base_id') != 'V129_BASE_%03d' % base or s.get('variant') != 'BEFORE' or s.get('split') != 'TRAIN':
                reasons.append('DECLARED_BEFORE_SCENE_IDENTITY_OR_ROLE')
            if g['image'].get('path') != full.get('path') or g['image'].get('sha256') != full.get('sha256') or g['image'].get('member') != s.get('full_RGB_key'):
                reasons.append('GROUP_SCENE_BEFORE_RGB_BINDING')
            decl = s.get('source_base_config')
            if not isinstance(decl, dict) or not isinstance(decl.get('file'), dict) or decl['file'].get('sha256') != config_binding['sha256'] or decl['file'].get('path') != ORIGINAL_CONFIG_SOURCE_PATH:
                reasons.append('CONFIG_SOURCE_BINDING')
            else:
                declarations.append(decl['row_index'])
            if bval(r['original_query_available']) != g['original_available8'][k]:
                reasons.append('ORIGINAL_INPUT_U_IDENTITY')
            old_groups.append(g)
            canonical = None if r['declared_CF_to_canonical'] == '' else int(r['declared_CF_to_canonical'])
            member_out.append({'row_id': r['row_id'], 'original_source_index': int(r['source_index']),
                               'original_sample_index': None if r['sample_index'] == '' else int(r['sample_index']),
                               'native_slot': k, 'declared_CF_to_canonical': canonical,
                               'original_query_available': bval(r['original_query_available']),
                               'original_crop_available': bval(r['crop_available']),
                               'group_id': r['group_id'], 'group_index': int(r['group_index']),
                               'original_known_or_masked_retained': True})
        declarations = sorted(set(declarations))
        if len(declarations) != 1: reasons.append('MISSING_OR_INCONSISTENT_CONFIG_ROW_INDEX')
        ci = declarations[0] if len(declarations) == 1 else None
        cfg = configs.get(ci)
        if not isinstance(cfg, dict): reasons.append('SELECTED_CONFIG_UNAVAILABLE')
        else:
            if cfg.get('base_index') != base or cfg.get('base_id') != 'V129_BASE_%03d' % base or cfg.get('split') != 'TRAIN':
                reasons.append('SELECTED_CONFIG_BASE_IDENTITY_OR_ROLE')
            perm = cfg.get('permCF_to_canonical')
            if not isinstance(perm, list) or sorted(perm) != list(range(8)):
                reasons.append('MISSING_OR_INVALID_NATIVE_PERMUTATION')
            elif any(m['declared_CF_to_canonical'] != perm[m['native_slot']] for m in member_out if 0 <= m['native_slot'] < 8):
                reasons.append('ORIGINAL_NATIVE_PERMUTATION_MISMATCH')
        if old_groups:
            keys = ('q8', 'L8', 'original_available8', 'box_xyxy', 'raw_hw')
            if any(len({fp(g[key]) for g in old_groups}) != 1 for key in keys):
                reasons.append('ORIGINAL_BEFORE_INPUTS_DIFFER_ACROSS_NATIVE_ROWS')
            g = old_groups[0]
            s = scenes.get(4 * base, {})
            before = {key: g[key] for key in keys}
            before.update({'group_id': g['group_id'], 'group_index': g['group_index'],
                           'image': g['image'], 'expected_BEFORE_decoded_RGB_sha256': s.get('decoded_RGB_sha256'),
                           'expected_BEFORE_encoded_RGB_binding': s.get('full_RGB'),
                           'render_raw_hw_declared_separately': s.get('raw_hw'),
                           'originalpermCF_to_canonical': None if not isinstance(cfg, dict) else cfg.get('permCF_to_canonical')})
            if not isinstance(before['expected_BEFORE_decoded_RGB_sha256'], str): reasons.append('MISSING_EXPECTED_BEFORE_RGB_HASH')
        else:
            before = None
        reasons = sorted(set(reasons))
        parent = {'base_index': base, 'config_row_index': ci, 'identity_PASS': not reasons,
                  'failure_reasons': reasons, 'source_base_role': 'FIT',
                  'before_input': before, 'native_members': sorted(member_out, key=lambda r: (r['native_slot'], r['row_id'])),
                  'selected_config_span': span_map.get(ci),
                  'all_original_parent_rows': len(members),
                  'unavailable_native_members_retained': sum(not bval(r['original_query_available']) for r in members),
                  'masked_native_members_retained': sum(r['sample_index'] == '' for r in members)}
        parents.append(parent)
        if reasons: failures.append({'base_index': base, 'failure_reasons': reasons})
        if isinstance(cfg, dict): config_rows.append(cfg)
    return parents, config_rows, failures


def run(inputs_bindings, output_dir):
    """Four explicitly relocated bindings; never follow embedded source pointers."""
    assert set(inputs_bindings) == set(INPUT_KEYS)
    raw = {key: checked(inputs_bindings[key]) for key in INPUT_KEYS}
    rows = list(csv.DictReader(io.StringIO(raw['syn7040_rows'].decode('utf-8'))))
    selected, role_by_base, census = select_parent_rows(rows)
    groups = [strict_loads(line) for line in raw['groups'].splitlines() if line]
    scene_indices = sorted({int(r['scene_index']) for rr in selected.values() for r in rr})
    scenes, scene_spans, scene_count = select_json_rows(raw['scenes'], scene_indices)
    if any(not isinstance(s, dict) or s.get('scene_index') != i for i, s in scenes.items()):
        raise ValueError('Selected scene row-index identity; no search/fallback')
    declared = []
    for s in scenes.values():
        if not isinstance(s, dict) or not isinstance(s.get('source_base_config'), dict):
            raise ValueError('Missing selected scene config declaration; no guessed index')
        declared.append(s['source_base_config']['row_index'])
    config_indices = sorted(set(declared))
    configs, spans, config_count = select_json_rows(raw['configs192_opaque'], config_indices)
    assert config_count == 192
    parents, config_rows, failures = make_manifest(rows, groups, scenes, configs, spans, role_by_base, selected, inputs_bindings['configs192_opaque'])
    census.update({'all_config_records_lexically_traversed': config_count,
                   'selected_config_records_JSON_decoded': len(configs),
                   'excluded_config_records_JSON_decoded': 0,
                   'excluded_config_records_numeric_decoded': 0,
                   'excluded_config_records_lexically_skipped': config_count - len(configs),
                   'selected_scene_records_JSON_decoded': len(scenes),
                   'all_scene_records_lexically_traversed': scene_count,
                   'nonselected_scene_records_JSON_decoded': 0,
                   'all_original_groups_input_metadata_JSON_decoded': len(groups),
                   'all_parent_identity_PASS': not failures, 'parent_failure_count': len(failures),
                   'same_before_native_input_byte_comparison': 'nullable decoded LE-FP64/bool exact, no tolerance',
                   'old_V254_strict_response_subset_used': False,
                   'new_rendering_or_model_or_training': False})
    result = {
        'SELECTED_FIT_CONFIGS.json': {'completed': True, 'rows': config_rows, 'selected_from_original_SOURCE_FIT_only': True, 'new_generation_admitted_by_this_file': False},
        'SOURCE_FIT_PARENT_MANIFEST.json': {'completed': True, 'all_parent_identity_PASS': not failures, 'parents': parents,
            'q8_is_original_initializer_not_native8_GT_projection': True, 'teacher_permutation_not_forward_feature': True,
            'native_coverage_loop_is_not_all8_D_or_physical_gold': True, 'new_generation_admitted_by_this_file': False},
        'SELECTED_CONFIG_ROW_SPANS.json': {'completed': True, 'source_sha256': inputs_bindings['configs192_opaque']['sha256'],
            'source_size_bytes': inputs_bindings['configs192_opaque']['size_bytes'], 'array_key': 'rows', 'rows': spans,
            'selected_scene_spans': scene_spans, 'excluded_record_values_emitted': False},
        'CENSUS.json': census,
        'SELECTION_FAILURES.json': {'completed': True, 'rows': failures, 'failed_parents_retained_in_parent_manifest': True},
    }
    out = guard(output_dir, fresh=True)
    out.mkdir(parents=True)
    outputs = {}
    for name, obj in result.items():
        (out / name).write_text(json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + '\n')
        outputs[name] = pin(out / name)
    complete = {'completed': True, 'PASS': not failures, 'inputs': inputs_bindings, 'outputs': outputs,
                'census': census, 'protocol_sha256': PROTOCOL_SHA,
                'new_generation_training_or_model': False, 'current_or_held': 0,
                'mixed_config_whole_bytes_hash_and_lexical_traversal': True,
                'nonselected_config_JSON_or_numeric_decode': 0,
                'SOURCE_DEV_input_metadata_CSV_strings_encountered': True,
                'SOURCE_DEV_visibility_label_prediction_or_score_values_used': False,
                'RGB_mesh_GT_archive_weights_arrays_payload_reads': 0}
    (out / 'COMPLETE.json').write_text(json.dumps(complete, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + '\n')
    return complete


def authorize(entry_path):
    e = strict_loads(guard(entry_path).read_bytes())
    if e.get('execution_authorized') is not True:
        raise PermissionError('Exact Root true entry required before SOURCE input reads')
    assert set(e) == {'execution_authorized', 'code', 'root_protocol', 'inputs', 'output_dir'}
    assert checked(e['code']) == guard(__file__).read_bytes() and Path(e['code']['path']) == Path(__file__)
    pb = checked(e['root_protocol'])
    assert sha(pb) == PROTOCOL_SHA
    protocol = strict_loads(pb)
    assert e['inputs'] == protocol['inputs'] and e['output_dir'] == protocol['output_dir']
    guard(e['output_dir'], fresh=True)
    return e


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--entry', required=True)
    e = authorize(ap.parse_args().entry)
    result = run(e['inputs'], e['output_dir'])
    print(json.dumps({'completed': result['completed'], 'PASS': result['PASS'], 'outputs': result['outputs']}, sort_keys=True))


if __name__ == '__main__':
    main()
