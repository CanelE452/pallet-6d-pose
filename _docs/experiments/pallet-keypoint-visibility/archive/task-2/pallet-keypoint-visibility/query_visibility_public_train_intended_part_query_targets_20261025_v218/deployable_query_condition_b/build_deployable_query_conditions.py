#!/usr/bin/env python3
"""Target-free C61-declared condition vectors; metadata-only SOURCE audit.

No image/model loader or target-to-feature conversion exists in this module.
Only initializer q8, active original L, original availability, native ID enter
the pure builder. Adjacent directions below are diagnostic re-expressions.
"""
import argparse
import collections
import copy
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import stat
import struct

API_KEYS = frozenset(('q8_xy', 'native_slot', 'L', 'query_available8'))
# Stored dynamic native order. These are topology names, not GT permutations.
NATIVE_NAMES = ('front_top_left', 'front_top_right', 'front_bottom_right',
                'front_bottom_left', 'rear_top_left', 'rear_top_right',
                'rear_bottom_right', 'rear_bottom_left')
AXIS_PARTNERS = {
    'width': (1, 0, 3, 2, 5, 4, 7, 6),
    'deck_to_bottom': (3, 2, 1, 0, 7, 6, 5, 4),
    'front_to_rear': (4, 5, 6, 7, 0, 1, 2, 3),
}


def finite_number(x):
    return type(x) in (int, float) and math.isfinite(x)


def finite_xy(q):
    return isinstance(q, (list, tuple)) and len(q) == 2 and all(finite_number(x) for x in q)


def f32(x):
    return struct.unpack('<f', struct.pack('<f', float(x)))[0]


def validate_deployment_payload(p):
    assert isinstance(p, dict) and set(p) == API_KEYS, 'Only exact deployment keys permitted'
    k = p['native_slot']
    assert type(k) is int and 0 <= k < 8, 'Native ID must be integer 0..7'
    q8, available = p['q8_xy'], p['query_available8']
    assert isinstance(q8, (list, tuple)) and len(q8) == 8
    assert isinstance(available, (list, tuple)) and len(available) == 8
    assert all(type(x) is bool for x in available)
    assert all(not available[j] or finite_xy(q8[j]) for j in range(8)), 'Available query requires finite original xy'
    if available[k]:
        assert finite_number(p['L']) and p['L'] > 0, 'Available active query requires positive finite original L'
    return k, q8, available


def build_deployment_condition(p):
    """Return 32 finite float32 scalars, with original U gate and self-zero."""
    k, q8, available = validate_deployment_payload(p)
    native = [f32(int(j == k)) for j in range(8)]
    mask, xy = [0.0] * 8, [0.0] * 16
    if available[k]:
        L = float(p['L'])
        for j in range(8):
            if j != k and available[j]:
                mask[j] = 1.0
                for axis in range(2):
                    d = (float(q8[j][axis]) - float(q8[k][axis])) / L
                    xy[2*j + axis] = f32(max(-8.0, min(8.0, d)))
    return {'native_onehot8': native, 'other_mask8': mask, 'other_dxdy16': xy,
            'condition32': native + mask + xy, 'original_inputU': not available[k]}


def initializer_geometry_diagnostics(p):
    """No new observed information. Never a target or extra input channel."""
    k, q8, available = validate_deployment_payload(p)
    out = {'native_role_name': NATIVE_NAMES[k], 'active_query_available': available[k],
           'available_initializer_slots': [j for j in range(8) if available[j]],
           'adjacent_directions': [], 'available_initializer_bbox_xyxy': None,
           'initializer_span_over_active_L': None,
           'not_an_observed_instance_or_part_certificate': True}
    for axis, mapping in AXIS_PARTNERS.items():
        j = mapping[k]
        row = {'axis_role': axis, 'other_native_slot': j, 'available': False,
               'unit_dxdy': None, 'length_over_active_L': None}
        if available[k] and available[j]:
            dx, dy = (float(q8[j][i]) - float(q8[k][i]) for i in range(2))
            length = math.hypot(dx, dy)
            row.update(available=True, length_over_active_L=length / float(p['L']))
            if length > 0:
                row['unit_dxdy'] = [dx / length, dy / length]
        out['adjacent_directions'].append(row)
    if available[k]:
        qs = [q8[j] for j in range(8) if available[j]]
        bounds = [min(q[0] for q in qs), min(q[1] for q in qs),
                  max(q[0] for q in qs), max(q[1] for q in qs)]
        out['available_initializer_bbox_xyxy'] = bounds
        out['initializer_span_over_active_L'] = [(bounds[2]-bounds[0])/p['L'],
                                                 (bounds[3]-bounds[1])/p['L']]
    return out


def canonical_json(x):
    return json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def unique_original_rows(candidate_rows):
    """Read only original_input. No best instance/nearest projection choice."""
    unique = {}
    for candidate in candidate_rows:
        r = candidate['original_input']
        row_id = r['row_id']
        if row_id in unique:
            assert canonical_json(r) == canonical_json(unique[row_id]), 'Duplicate candidate changed original identity'
        else:
            unique[row_id] = copy.deepcopy(r)
    out = sorted(unique.values(), key=lambda r: (r['frame_position'], r['native_slot']))
    assert len({(r['frame_key'], r['native_slot']) for r in out}) == len(out)
    return out


def make_frame_payload(frame_rows, k):
    assert len(frame_rows) == 8 and {r['native_slot'] for r in frame_rows} == set(range(8))
    rr = sorted(frame_rows, key=lambda r: r['native_slot'])
    assert len({r['frame_key'] for r in rr}) == 1
    assert len({r['component_id'] for r in rr}) == 1
    assert len({r['internal_source_role'] for r in rr}) == 1
    assert all(r['original_role'] == 'TRAIN' for r in rr)
    return {'q8_xy': [r['query_xy'] for r in rr], 'native_slot': k,
            'L': rr[k]['L'], 'query_available8': [r['original_query_available'] for r in rr]}


def nofollow_path(p, final_missing=False):
    p = Path(p)
    assert p.is_absolute() and 'v112' not in str(p).casefold(), 'Forbidden or relative path'
    cur = Path(p.anchor)
    for i, name in enumerate(p.parts[1:]):
        cur = cur / name
        try:
            mode = os.lstat(cur).st_mode
        except FileNotFoundError:
            assert final_missing and i == len(p.parts)-2
            return p
        assert not stat.S_ISLNK(mode), 'Symlink forbidden'
    return p


def read_bytes(p):
    p = nofollow_path(p)
    with os.fdopen(os.open(p, os.O_RDONLY | os.O_NOFOLLOW), 'rb') as f:
        return f.read()


def digest(b):
    return hashlib.sha256(b).hexdigest()


def binding(p):
    return {'path': str(p), 'sha256': digest(read_bytes(p))}


def checked(b):
    assert set(b) == {'path', 'sha256'}
    raw = read_bytes(b['path'])
    assert digest(raw) == b['sha256']
    return json.loads(raw)


def write_new(p, b):
    p = nofollow_path(p, True)
    with os.fdopen(os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644), 'wb') as f:
        f.write(b)


def write_json(p, x):
    write_new(p, (json.dumps(x, ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode())


def generate(entry_path):
    e = json.loads(read_bytes(entry_path))
    assert e['approved_metadata_only_generation'] is True
    assert binding(Path(__file__)) == e['code']
    protocol = checked(e['protocol'])
    scope = checked(protocol['scope'])
    assert scope['phase'] == 'V218'
    permitted = {x['path']: x['sha256'] for x in scope['allowed_saved_metadata']}
    assert all(permitted[x['path']] == x['sha256'] for x in protocol['exact_inputs'])
    by_name = {Path(x['path']).name: x for x in protocol['exact_inputs']}
    doc = checked(by_name['ALL120_CANDIDATE_NATIVE_QUERY_DIAGNOSTICS.json'])
    originals = unique_original_rows(doc['rows120'])
    assert len(doc['rows120']) == 120 and len(originals) == 96
    groups = collections.defaultdict(list)
    for r in originals:
        groups[r['frame_position']].append(r)
    assert set(groups) == set(range(12))
    records, diagnostic = [], []
    for pos, rr in sorted(groups.items()):
        for r in sorted(rr, key=lambda x: x['native_slot']):
            payload = make_frame_payload(rr, r['native_slot'])
            features = build_deployment_condition(payload)
            records.append({'row_id': r['row_id'], 'frame_position': pos,
                            'native_slot': r['native_slot'], 'features': features})
            diagnostic.append({'row_id': r['row_id'], **initializer_geometry_diagnostics(payload)})
    assert all(len(r['features']['condition32']) == 32 and
               all(math.isfinite(x) for x in r['features']['condition32']) for r in records)
    assert sum(r['features']['original_inputU'] for r in records) == sum(not r['original_query_available'] for r in originals) == 11
    roles = collections.Counter(r['internal_source_role'] for r in originals)
    assert roles == {'SOURCE_FIT': 64, 'SOURCE_DEV': 32}
    assert all(r['features']['original_inputU'] == (not o['original_query_available'])
               for r, o in zip(records, originals))
    out = nofollow_path(e['output_directory'], True)
    assert out.parent == Path(__file__).parent
    os.mkdir(out)
    write_json(out/'ALL96_ORIGINAL_IDENTITIES_ADMIN_ONLY.json',
               {'completed': True, 'rows96': originals, 'not_model_features': True})
    write_new(out/'ALL96_DEPLOYABLE_CONDITION32.jsonl',
              ''.join(canonical_json(r)+'\n' for r in records).encode())
    write_json(out/'ALL96_INITIALIZER_GEOMETRY_DIAGNOSTICS_NOT_EXTRA_FEATURES.json',
               {'completed': True, 'rows96': diagnostic, 'new_observed_information': False})
    rows = []
    for r, o in zip(records, originals):
        rows.append({'row_id': o['row_id'], 'frame_position': o['frame_position'],
                     'frame_key': o['frame_key'], 'component_id': o['component_id'],
                     'internal_source_role': o['internal_source_role'], 'native_slot': o['native_slot'],
                     'q_x': o['query_xy'][0] if o['query_xy'] else None,
                     'q_y': o['query_xy'][1] if o['query_xy'] else None, 'L': o['L'],
                     'original_inputU': not o['original_query_available'],
                     'other_context_slots': int(sum(r['features']['other_mask8']))})
    s = io.StringIO(newline=''); w = csv.DictWriter(s, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    write_new(out/'ALL96_CONDITION_IDENTITY_AND_AVAILABILITY.csv', s.getvalue().encode())
    summary = {'completed': True, 'PASS': True, 'frames': len(groups), 'components': len({r['component_id'] for r in originals}),
               'candidate_rows_read': len(doc['rows120']), 'unique_query_rows': len(originals), 'original_inputU': 11,
               'original_roles': dict(roles), 'native_counts': dict(collections.Counter(r['native_slot'] for r in originals)),
               'original_inputU_by_native': dict(collections.Counter(r['native_slot'] for r in originals if not r['original_query_available'])),
               'U_by_role': dict(collections.Counter(r['internal_source_role'] for r in originals if not r['original_query_available'])),
               'finite_float32_condition_scalars': len(records)*32, 'condition_dimension': 32,
               'GT_candidate_features': 0, 'new_pixel_context_samples': 0, 'new_architecture': False,
               'same_declared_information_as_C61': True, 'C61_code_or_weight_parity_claimed': False,
               'native_visibility_labels_created': 0, 'training_admission': False, 'fit_or_forward': 0,
               'component_role_preserved': True, 'diagnostic_adjacent_features_appended': False}
    write_json(out/'SUMMARY.json', summary)
    outputs = {name: binding(out/name) for name in ('ALL96_ORIGINAL_IDENTITIES_ADMIN_ONLY.json',
               'ALL96_DEPLOYABLE_CONDITION32.jsonl', 'ALL96_INITIALIZER_GEOMETRY_DIAGNOSTICS_NOT_EXTRA_FEATURES.json',
               'ALL96_CONDITION_IDENTITY_AND_AVAILABILITY.csv', 'SUMMARY.json')}
    write_json(out/'COMPLETE.json', {'completed': True, 'PASS': True, 'entry': binding(entry_path),
                                   'code': binding(Path(__file__)), 'protocol': e['protocol'],
                                   'inputs': protocol['exact_inputs'], 'outputs': outputs,
                                   'summary': summary, 'metadata_only': True, 'training_admission': False})
    print(json.dumps({'COMPLETE': binding(out/'COMPLETE.json'), 'summary': summary}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--entry', required=True)
    args = parser.parse_args()
    generate(args.entry)
