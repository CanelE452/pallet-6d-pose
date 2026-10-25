"""Two separately authorized SOURCE TRAIN stages; no raw-image/GT reader."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys

sys.dont_write_bytecode = True
P = Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility')
R = P / 'publication_root_preparation'
V66 = P / 'query_visibility_controlled_real_20261022_v66'
TRAIN_COMPLETE = {'path': str(R / 'V246_ROOT_PALLOX_TRAIN/train_input_metadata_v1/COMPLETE.json'),
                  'sha256': '8db146bc2a5b7acfd1f59539d9f4e3943f9ea550a734721e30ff795ea8fc807f'}
MIXED = {'path': str(V66 / 'public_pallet/SYMMETRIC_ALL50_V2/PAIR_EVIDENCE_MANIFEST.jsonl'),
         'sha256': 'f74b2d532ad09a7759bfdea839a5c48c5f30cd14437201e209fd96d24da63f70'}
HELPER = {'path': str(V66 / 'contract/paired_truth.py'),
          'sha256': '649c93b98085289bb1aa797b3beba141423f5ffbeb055b5c7f2f571e208f0c14'}
PAIRED_PROTOCOL_SHA = '554178a520287030a5aef5162272ee45c3307d49c8e68c1e20449cb8c57ebbfb'
OUT = {'select': R / 'V246_ROOT_PALLOX_TRAIN/train_pair_certificate_metadata_v2',
       'probe': R / 'V246_ROOT_PALLOX_TRAIN/same_q_conditional_probe_v2'}
FRAME_NAME = 'TRAIN44_FRAMES_INPUT_ONLY.jsonl'
QUERY_NAME = 'TRAIN352_ORIGINAL_NATIVE_QUERIES_INPUT_ONLY.jsonl'
CERT_KEYS = ('visible_positive', 'visible_negative', 'full_positive', 'full_negative')
QUERY_FIELDS = {'query_id', 'frame_index', 'frame_input_id', 'pair_id', 'view', 'split',
                'native_slot', 'query_xy', 'roi_side', 'original_query_continuous_valid',
                'query_nearest_pixel_valid', 'query_rgb96_centre_domain_valid', 'reason',
                'native_semantic_state', 'human_gold'}
# Unretained values, including counts/statistics, are skipped lexically, never json.loads'd.
TREE = {'pair_id': None, 'session': None, 'split': None, 'A_frame': None, 'B_frame': None,
        'files': {v: {'image': {'path': None, 'sha256': None, 'bytes': None}}
                  for v in ('A', 'B')},
        'evidence': {'path': None, 'sha256': None}, 'protocol_sha256': None,
        'tier': None, 'supplied_pose_and_K_identical': None,
        'pixel_RGB_depth_alignment_certified': None,
        'native_depth_is_assumed_RGB_grid': None}


def lex(path):
    raw = str(path)
    if any(x in raw.lower() for x in ('v112', '__pycache__', '.pyc', 'current',
                                       'private', 'held', 'holdout', 'oldval', 'source_val')):
        raise ValueError('Forbidden path')
    p = Path(raw)
    if not p.is_absolute() or '..' in p.parts:
        raise ValueError('Absolute traversal-free path required')
    return p


def guard(path):
    p = lex(path)  # lexical rejection precedes every filesystem action
    for ancestor in reversed((p, *p.parents)):
        try:
            info = os.lstat(ancestor)
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode):
            raise ValueError('Symlink forbidden')
    return p


def identity(b):
    lex(b['path'])
    if re.fullmatch('[0-9a-f]{64}', b['sha256']) is None:
        raise ValueError('SHA256 required')
    return b['path'], b['sha256']


def same(a, b):
    return identity(a) == identity(b)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def binding(path):
    p = guard(path)
    raw = p.read_bytes()
    return {'path': str(p), 'sha256': digest(raw), 'size_bytes': len(raw)}


def admitted(b, e, certificate=False):
    white = e['certificate_whitelist'] if certificate else e['metadata_whitelist']
    if identity(b) not in {identity(x) for x in white}:
        raise ValueError('Undeclared payload')
    p = guard(b['path'])
    raw = p.read_bytes()
    if digest(raw) != b['sha256']:
        raise ValueError('SHA mismatch')
    for k in ('bytes', 'size_bytes'):
        if k in b and b[k] != len(raw):
            raise ValueError('Size mismatch')
    return p, raw


def metadata(b, e):
    return json.loads(admitted(b, e)[1])


def ws(s, i):
    while i < len(s) and s[i].isspace():
        i += 1
    return i


def string_end(s, i):
    if s[i] != '"':
        raise ValueError('String expected')
    i += 1
    while i < len(s):
        if s[i] == '\\':
            i += 2
        elif s[i] == '"':
            return i + 1
        else:
            i += 1
    raise ValueError('Unterminated string')


def value_end(s, i):
    i = ws(s, i)
    if s[i] == '"':
        return string_end(s, i)
    if s[i] in '[{':
        stack = [s[i]]
        i += 1
        while i < len(s) and stack:
            if s[i] == '"':
                i = string_end(s, i)
                continue
            if s[i] in '[{':
                stack.append(s[i])
            elif s[i] in ']}':
                if (stack.pop(), s[i]) not in (('[', ']'), ('{', '}')):
                    raise ValueError('Unbalanced skipped value')
            i += 1
        if stack:
            raise ValueError('Unbalanced skipped value')
        return i
    start = i
    while i < len(s) and s[i] not in ',}':
        i += 1
    if not s[start:i].strip():
        raise ValueError('Empty value')
    return i


def selected_object(s, tree, i=0):
    i = ws(s, i)
    if s[i] != '{':
        raise ValueError('Object required')
    i += 1
    out, seen = {}, set()
    while True:
        i = ws(s, i)
        if s[i] == '}':
            return out, i + 1
        end = string_end(s, i)
        key = json.loads(s[i:end])
        if key in seen:
            raise ValueError('Duplicate object key')
        seen.add(key)
        i = ws(s, end)
        if s[i] != ':':
            raise ValueError('Colon required')
        i = ws(s, i + 1)
        if key not in tree:
            i = value_end(s, i)
        elif tree[key] is None:
            end = value_end(s, i)
            out[key] = json.loads(s[i:end])
            i = end
        else:
            out[key], i = selected_object(s, tree[key], i)
        i = ws(s, i)
        if s[i] == '}':
            return out, i + 1
        if s[i] != ',':
            raise ValueError('Comma required')
        i += 1


def first_pair_id(raw):
    # Pinned producer writes pair_id as the first key. Excluded suffix stays bytes.
    match = re.match(rb'\s*\{\s*"pair_id"\s*:\s*("(?:[^"\\]|\\.)*")\s*[,}]', raw)
    if match is None:
        raise ValueError('First identity field differs from pinned producer')
    return json.loads(match.group(1))


def train_inputs(e):
    if not same(e['train_complete'], TRAIN_COMPLETE):
        raise ValueError('TRAIN input authority changed')
    complete = metadata(e['train_complete'], e)
    if not (complete['completed'] is True and complete['PASS'] is True):
        raise ValueError('TRAIN extraction incomplete')
    inputs = complete['outputs']
    frames = [json.loads(s) for s in admitted(inputs[FRAME_NAME], e)[1].splitlines() if s]
    queries = [json.loads(s) for s in admitted(inputs[QUERY_NAME], e)[1].splitlines() if s]
    if len(frames) != 44 or len(queries) != 352:
        raise ValueError('44/352 denominator changed')
    fm = {f['frame_input_id']: f for f in frames}
    pairs = {}
    if len(fm) != 44:
        raise ValueError('Duplicate frame')
    for f in frames:
        if f['split'] != 'TRAIN_LAB' or not f['session'].endswith('_lab') or f['raw_hw'] != [480, 640]:
            raise ValueError('Non-TRAIN or changed native grid')
        if f['view'] not in ('before', 'after'):
            raise ValueError('Unknown view')
        pair = pairs.setdefault(f['pair_id'], {})
        if f['view'] in pair:
            raise ValueError('Duplicate pair view')
        pair[f['view']] = f
    if len(pairs) != 22 or any(set(v) != {'before', 'after'} for v in pairs.values()):
        raise ValueError('22 paired siblings required')
    seen, slots = set(), Counter()
    for q in queries:
        if set(q) != QUERY_FIELDS or q['query_id'] in seen:
            raise ValueError('Query schema/ID changed')
        seen.add(q['query_id'])
        f = fm[q['frame_input_id']]
        if q['split'] != 'TRAIN_LAB' or any(q[k] != f[k] for k in ('pair_id', 'view', 'split')):
            raise ValueError('Query/frame identity changed')
        if q['native_slot'] not in range(8) or q['query_id'] != f['frame_input_id'] + ':native' + str(q['native_slot']):
            raise ValueError('Native/row ID changed')
        if q['native_semantic_state'] != 'UNKNOWN' or q['human_gold'] is not False:
            raise ValueError('Native authority changed')
        slots[(q['frame_input_id'], q['native_slot'])] += 1
    if len(slots) != 352 or set(slots.values()) != {1}:
        raise ValueError('All original slots required')
    if sum(q['original_query_continuous_valid'] for q in queries) != 297:
        raise ValueError('Original input support changed')
    return complete, frames, queries, pairs


def canonicalize_evidence(r):
    # Pure transport of the one producer-declared relative alias; never search.
    name = r['pair_id'].replace(':', '__') + '.npz'
    relative = 'SYMMETRIC_ALL50_V2/' + name
    absolute = str(V66 / 'public_pallet/SYMMETRIC_ALL50_V2' / name)
    evidence = r['evidence']
    if set(evidence) != {'path', 'sha256'}:
        raise ValueError('Unexpected original evidence binding schema')
    original = evidence['path']
    if original == absolute:
        return r
    if original != relative:
        raise ValueError('Only exact declared relative evidence alias is allowed')
    r['evidence'] = {'path': absolute, 'sha256': evidence['sha256'],
                     'original_relative_path': original}
    return r


def validate_pair(r, pair_id, pair):
    if set(r) != set(TREE) or r['pair_id'] != pair_id or r['split'] != 'TRAIN_LAB':
        raise ValueError('Selected metadata fields/role changed')
    if not r['session'].endswith('_lab') or r['protocol_sha256'] != PAIRED_PROTOCOL_SHA:
        raise ValueError('Session/protocol changed')
    if r['tier'] != 'CONDITIONAL_TEACHER_NOT_GOLD' or type(r['supplied_pose_and_K_identical']) is not bool:
        raise ValueError('Certificate authority mismatch')
    if r['pixel_RGB_depth_alignment_certified'] is not False or r['native_depth_is_assumed_RGB_grid'] is not True:
        raise ValueError('Assumption flags changed')
    for view, side in (('before', 'A'), ('after', 'B')):
        f = pair[view]
        im = r['files'][side]['image']
        identity(im)
        if r['session'] != f['session'] or im['path'] != f['image'] or im['sha256'] != f['image_sha256']:
            raise ValueError('Image identity mismatch')
        if f['frame_input_id'] != 'PALLOX50:' + pair_id + ':' + view:
            raise ValueError('Original frame ID mismatch')
        if Path(im['path']).stem != r[side + '_frame']:
            raise ValueError('A/B frame correspondence mismatch')
    expected = V66 / 'public_pallet/SYMMETRIC_ALL50_V2' / (pair_id.replace(':', '__') + '.npz')
    evidence = r['evidence']
    if set(evidence) not in ({'path', 'sha256'}, {'path', 'sha256', 'original_relative_path'}):
        raise ValueError('Unexpected canonical evidence binding schema')
    if 'original_relative_path' in evidence and evidence['original_relative_path'] != 'SYMMETRIC_ALL50_V2/' + expected.name:
        raise ValueError('Original relative alias changed')
    if identity(r['evidence'])[0] != str(expected):
        raise ValueError('Unexpected per-pair certificate path')


def write(path, value, jsonl=False):
    p = guard(path)
    if p.exists():
        raise ValueError('Preserve existing output')
    text = ''.join(json.dumps(x, ensure_ascii=False, allow_nan=False) + '\n' for x in value) if jsonl else json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    p.write_text(text)
    return binding(p)


def select(e, out, pairs):
    if not same(e['mixed_manifest'], MIXED) or e['certificate_whitelist']:
        raise ValueError('Stage1 metadata-only authority mismatch')
    _, raw = admitted(e['mixed_manifest'], e)
    selected, skipped = {}, 0
    for line in raw.splitlines():
        if not line.strip():
            continue
        pair_id = first_pair_id(line)
        if pair_id not in pairs:
            skipped += 1
            continue
        if pair_id in selected:
            raise ValueError('Duplicate selected pair')
        text = line.decode('utf-8')
        row, end = selected_object(text, TREE)
        if text[end:].strip():
            raise ValueError('Trailing record data')
        row = canonicalize_evidence(row)
        validate_pair(row, pair_id, pairs[pair_id])
        selected[pair_id] = row
    if set(selected) != set(pairs) or skipped != 28:
        raise ValueError('Selected22/excluded28 closure mismatch')
    rows = [selected[k] for k in pairs]  # original TRAIN first-frame order
    rb = write(out / 'TRAIN22_PAIR_CERTIFICATE_METADATA.jsonl', rows, True)
    white = write(out / 'TRAIN22_CERTIFICATE_PAYLOAD_WHITELIST.json',
                  {'TRAIN_only': True, 'pair_count': 22, 'files': [r['evidence'] for r in rows]})
    return {'selected_pairs': 22, 'original_frames': 44, 'original_queries': 352,
            'metadata': rb, 'certificate_payload_whitelist': white,
            'SOURCE_VAL_transport': {'mixed_file_bytes_opaque_SHA_and_line_transport': True,
                                     'excluded_pair_identity_strings_read': skipped,
                                     'excluded_remaining_values_JSON_interpreted': 0,
                                     'excluded_labels_stats_or_identifiers_output': 0},
            'TRAIN_nonallowlisted_stats_values_interpreted': 0,
            'certificate_payload_stat_hash_or_read': 0}


def probe(e, out, queries, pairs):
    import numpy as np
    selected_complete = metadata(e['selected_metadata_complete'], e)
    if not (selected_complete['completed'] is True and selected_complete['PASS'] is True and selected_complete['stage'] == 'select'):
        raise ValueError('Certificate selection not complete')
    if not same(selected_complete['train_complete'], e['train_complete']):
        raise ValueError('Selection TRAIN binding mismatch')
    if not same(selected_complete['code'], e['code']) or not same(selected_complete['protocol'], e['protocol']):
        raise ValueError('Selection producer binding mismatch')
    rows = [json.loads(x) for x in admitted(selected_complete['metadata'], e)[1].splitlines() if x]
    white = metadata(selected_complete['certificate_payload_whitelist'], e)
    if not white['TRAIN_only'] or white['pair_count'] != 22 or len(rows) != 22:
        raise ValueError('TRAIN certificate denominator mismatch')
    if {identity(x) for x in white['files']} != {identity(x) for x in e['certificate_whitelist']} or len(e['certificate_whitelist']) != 22:
        raise ValueError('Exact22 certificate whitelist required')
    if not same(e['paired_truth_code'], HELPER):
        raise ValueError('Pinned scalar helper changed')
    namespace = {'__name__': 'v246_pinned_pure_paired_truth'}
    helper_source = admitted(e['paired_truth_code'], e)[1]
    exec(compile(helper_source, HELPER['path'], 'exec'), namespace)  # pure NumPy helper, no file reader/imported producer
    bypair = {}
    for r in rows:
        pid = r['pair_id']
        if pid in bypair or pid not in pairs:
            raise ValueError('Selected pair duplicates/unexpected IDs')
        validate_pair(r, pid, pairs[pid])
        bypair[pid] = r
    if set(bypair) != set(pairs):
        raise ValueError('Selected pair missing')
    # Group original rows without changing original output order.
    answers, qa = {}, []
    for pid, pair in pairs.items():
        record = bypair[pid]
        path, _ = admitted(record['evidence'], e, certificate=True)
        with np.load(path, allow_pickle=False) as z:
            for view, side in (('before', 'A'), ('after', 'B')):
                fields = [z[k + '_' + side] for k in CERT_KEYS]
                if any(x.dtype != np.bool_ or x.shape != (480, 640) for x in fields):
                    raise ValueError('Native Boolean480x640 certificate required')
                vp, vn, fp, fn = fields
                if fn.any() or (vp & vn).any() or (fp & fn).any() or (vp & ~fp).any():
                    raise ValueError('Certificate conflict/full-negative forbidden')
                empty = not any(x.any() for x in fields)
                if not record['supplied_pose_and_K_identical'] and not empty:
                    raise ValueError('Ineligible pair has contradictory nonempty support')
                cert = namespace['construct_certainty'](*fields)
                for q in queries:
                    if q['pair_id'] == pid and q['view'] == view:
                        truth = namespace['query_truth'](cert, q['query_xy'])
                        if truth['exact_state'] == 2 or truth['radius2_state'] == 2 or truth['semantic_self'] is not None or truth['corner_semantic_truth'] is not None:
                            raise ValueError('Unexpected O/SELF/native authority')
                        if not q['original_query_continuous_valid'] and (truth['valid'] or truth['exact_state'] != -1 or truth['radius2_state'] != -1):
                            raise ValueError('Original unavailable query must remain U')
                        reason = ('PAIR_METADATA_INELIGIBLE_UNKNOWN' if not record['supplied_pose_and_K_identical']
                                  else 'EMPTY_CERTIFICATE_UNKNOWN' if empty
                                  else 'INPUT_OR_ROUNDED_QUERY_UNSUPPORTED_U' if not truth['valid']
                                  else 'PARTIAL_OR_NO_CERTIFIED_SUPPORT_UNKNOWN' if truth['exact_state'] == -1
                                  else 'EXPLICIT_CONDITIONAL_SAME_PIXEL_SUPPORT')
                        answers[q['query_id']] = {'original_input_unchanged': q,
                            'certificate_binding': record['evidence'], 'certificate_view': side,
                            'conditional_pixel_truth': truth, 'conditional_reason': reason,
                            'native_visibility_label': None,
                            'native_expected_part_correspondence': 'UNKNOWN', 'training_admission': False}
                qa.append({'pair_id': pid, 'view': side, 'native_boolean_fields': 4,
                           'shape': [480, 640], 'full_negative_pixels': 0, 'conflicts': 0,
                           'empty_certificate': empty,
                           'pair_metadata_eligible': record['supplied_pose_and_K_identical']})
    if len(answers) != 352:
        raise ValueError('Original352 rows must be retained')
    ordered = [answers[q['query_id']] for q in queries]
    rb = write(out / 'TRAIN352_SAME_Q_CONDITIONAL_PIXEL_TRUTH.jsonl', ordered, True)
    summary = {'rows': 352, 'frames': 44, 'pairs': 22,
               'original_continuous_available': 297, 'original_input_U_retained': 55,
               'exact1pixel_counts': dict(Counter(str(x['conditional_pixel_truth']['exact_state']) for x in ordered)),
               'radius2_5x5_counts': dict(Counter(str(x['conditional_pixel_truth']['radius2_state']) for x in ordered)),
               'helper_valid': sum(x['conditional_pixel_truth']['valid'] for x in ordered),
               'radius2_strict_E_core': sum(x['conditional_pixel_truth'].get('radius2_strict_E_core', False) for x in ordered),
               'radius2_clipped_by_frame': sum(x['conditional_pixel_truth'].get('radius2_clipped_by_frame', False) for x in ordered),
               'canonical_state_encoding': {'-1': 'UNKNOWN', '0': 'CONDITIONAL_VISIBLE_SUPPORT', '1': 'CONDITIONAL_EXTERNAL_SUPPORT'},
               'semantic_SELF': None, 'native_labels_created': 0, 'training_admission': False,
               'scope': 'Conditional same-pixel/window support; no native-part/corner gold or literal source mask absence negative.'}
    sb = write(out / 'SUMMARY.json', summary)
    qb = write(out / 'CERTIFICATE_FIELD_QA.json', {'completed': True, 'PASS': True, 'pairs': 22, 'views': 44, 'rows': qa,
               'only_four_bool_fields_per_view_loaded': True, 'raw_RGB_depth_mask_mesh_pose_K_reads': 0})
    return {'rows': rb, 'summary': sb, 'certificate_QA': qb,
            'selected_metadata_complete': e['selected_metadata_complete'],
            'paired_truth_code': e['paired_truth_code'], 'counts': summary,
            'certificate_whitelist': e['certificate_whitelist']}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--entry', required=True)
    args = parser.parse_args()
    entry_path = guard(args.entry)
    e = json.loads(entry_path.read_bytes())
    stage = e.get('stage')
    if stage not in OUT or e.get('approved') is not True:
        raise ValueError('Separate Root-approved stage entry required')
    if stage == 'select' and e.get('approved_mixed_SOURCE_metadata_selection') is not True:
        raise ValueError('Mixed SOURCE metadata transport not authorized')
    if stage == 'probe' and e.get('approved_TRAIN22_certificate_payloads') is not True:
        raise ValueError('TRAIN certificate access not authorized')
    if not same(e['code'], binding(__file__)):
        raise ValueError('Code binding mismatch')
    for b in e['metadata_whitelist'] + e['certificate_whitelist']:
        identity(b)  # reject forbidden paths before opening any declared child
    protocol = metadata(e['protocol'], e)
    if not same(protocol['code'], e['code']) or not same(protocol['train_complete'], TRAIN_COMPLETE):
        raise ValueError('Implementation protocol mismatch')
    scope = metadata(e['root_scope'], e)
    if scope.get('approved') is not True or scope.get('stage') != stage:
        raise ValueError('Root scope stage/approval mismatch')
    if not same(scope['code'], e['code']) or not same(scope['protocol'], e['protocol']):
        raise ValueError('Root scope binding mismatch')
    if not same(scope['train_complete'], e['train_complete']):
        raise ValueError('Root scope TRAIN authority mismatch')
    # Scope cannot hash-bind itself. Entry adds the scope binding to its exact list.
    if {identity(x) for x in scope['metadata_whitelist']} | {identity(e['root_scope'])} != {identity(x) for x in e['metadata_whitelist']} or {identity(x) for x in scope['certificate_whitelist']} != {identity(x) for x in e['certificate_whitelist']}:
        raise ValueError('Root scope whitelist mismatch')
    if e['output_dir'] != str(OUT[stage]) or scope['output_dir'] != e['output_dir']:
        raise ValueError('Fixed fresh stage output required')
    out = guard(e['output_dir'])
    if out.exists():
        raise ValueError('No overwrite/retry')
    _, _, queries, pairs = train_inputs(e)
    out.mkdir(parents=True)
    try:
        result = select(e, out, pairs) if stage == 'select' else probe(e, out, queries, pairs)
        result.update(completed=True, PASS=True, stage=stage, code=e['code'], protocol=e['protocol'],
                      root_scope=e['root_scope'], entry=binding(entry_path), train_complete=e['train_complete'],
                      raw_RGB_depth_mask_mesh_pose_K_model_reads=0, native_labels_created=0,
                      training_admission=False, automatic_retry=False)
        complete = write(out / 'COMPLETE.json', result)
        print(json.dumps({'completed': True, 'PASS': True, 'stage': stage, 'complete': complete}), flush=True)
    except Exception as exc:
        write(out / 'FAILURE.json', {'completed': False, 'PASS': False, 'stage': stage,
              'error_type': type(exc).__name__, 'error': str(exc), 'automatic_retry': False,
              'code': e['code'], 'entry': binding(entry_path)})
        raise


if __name__ == '__main__':
    main()
