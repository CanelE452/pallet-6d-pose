#!/usr/bin/env python3
"""SOURCE TRAIN saved-point / saved-alpha diagnostic; never creates labels."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import traceback
import numpy as np

P = Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility')
V = P / 'query_visibility_source_train_native_point_control_alignment_20261025_v209'
H = V / 'native_control_authority_preparation_c'
V208 = P / 'query_visibility_source_train_native_point_state_grounding_20261025_v208'
V189 = P / 'query_visibility_sampled_domain_direct_end_20261024_v189'
SCOPE = V / 'ROOT_PROSPECTIVE_SOURCE_TRAIN_NATIVE_POINT_CONTROL_AUTHORITY_SCOPE.json'
SELECTION = V208 / 'preparation_a/SELECTED_SOURCE_TRAIN_BEFORE_GT.json'
POINT_COMPLETE = V208 / 'source_train_native_point_grounding_v2/COMPLETE.json'
POINT_ROWS = POINT_COMPLETE.parent / 'ALL_ORIGINAL_BEFORE_NATIVE_POINT_STATE_AND_LOCALITY.jsonl'
SOURCE_COMPLETE = V189 / 'synthetic_sampled_end_controls_v1/COMPLETE.json'
PRODUCER = V189 / 'direct_end_training_preparation_c/prepare_and_generate_synthetic_sampled_end.py'
CANDIDATE = PRODUCER.parent / 'synthetic_direct_end_candidate.py'
SOURCE_PROTOCOL = PRODUCER.parent / 'PROTOCOL_BEFORE_SYNTHETIC_END_CONTROL_BODIES.json'
PROTOCOL = H / 'PROTOCOL_BEFORE_TRAIN_ALPHA_PAYLOADS.json'
WHITELIST = H / 'TRAIN82_PACKET_FOREIGN_ALPHA_WHITELIST.json'
ASSOCIATIONS = H / 'ALL1968_TRAIN_POINT_ALPHA_ASSOCIATIONS.jsonl'
CHECKS = H / 'GENERATED_INVENTED_CHECKS.json'
VARIANTS = ('BEFORE', 'COVER', 'SHAM')
IDENTITY = ('original_global_index', 'original_occurrence_id', 'query_xy', 'L',
            'original_query_available', 'original_input_flags')


def require(value, message):
    if not value:
        raise AssertionError(message)


def safe(path):
    raw = str(path)
    require('v112' not in raw.lower(), 'V112 lexical rejection before filesystem')
    p = Path(raw)
    require(p.is_absolute() and '..' not in p.parts, 'Absolute literal path, no traversal')
    for item in reversed((p, *p.parents)):
        require(not item.is_symlink(), 'No symlinks: ' + str(item))
    return p


def digest(path, binary=False):
    p = safe(path)
    require(binary or p.suffix.lower() in ('.json', '.jsonl', '.py', '.md'), 'Metadata/code only')
    h = hashlib.sha256()
    with p.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def binding(path):
    return {'path': str(path), 'sha256': digest(path)}


def verify(b):
    require(set(b) == {'path', 'sha256'} and digest(b['path']) == b['sha256'], 'Exact text binding')
    return safe(b['path'])


def read(path):
    return json.loads(safe(path).read_text())


def lines(path):
    return [json.loads(x) for x in safe(path).read_text().splitlines() if x]


def write(path, value):
    with safe(path).open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def write_lines(path, values):
    with safe(path).open('x') as f:
        for value in values:
            f.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')


def alpha_relation(alpha, reference):
    """Positive taps at saved continuous xy; zero-weight neighbors never vote.

    These are raster opacity observations, not a ray recast or a new D/E/S label.
    A float64 value rounded to 1 is not used to certify coverage: every positive
    tap must actually be True. Mixed and outside taps remain separate UNKNOWN.
    """
    require(alpha.ndim == 2 and alpha.dtype == np.bool_, 'Boolean fullimage alpha')
    h, w = alpha.shape
    out = {'status': 'UNKNOWN', 'reason': None, 'positive_taps': [],
           'reference_in_canvas_extent': None, 'reference_in_integer_center_extent': None}
    if reference is None or len(reference) != 2 or not all(math.isfinite(x) for x in reference):
        out['reason'] = 'MISSING_OR_NONFINITE_SAVED_REFERENCE'
        return out
    x, y = map(float, reference)
    out['reference_in_canvas_extent'] = -.5 <= x < w - .5 and -.5 <= y < h - .5
    out['reference_in_integer_center_extent'] = 0 <= x <= w - 1 and 0 <= y <= h - 1
    ix, iy = math.floor(x), math.floor(y)
    dx, dy = x - ix, y - iy
    taps = ((ix, iy, (1-dx)*(1-dy)), (ix+1, iy, dx*(1-dy)),
            (ix, iy+1, (1-dx)*dy), (ix+1, iy+1, dx*dy))
    flags = []
    for tx, ty, weight in taps:
        if weight <= 0:
            continue
        inside = 0 <= tx < w and 0 <= ty < h
        flag = bool(alpha[ty, tx]) if inside else None
        out['positive_taps'].append({'xy': [tx, ty], 'weight': weight,
                                     'inside': inside, 'opaque': flag})
        flags.append(flag)
    if None in flags:
        out['reason'] = 'POSITIVE_TAP_OUTSIDE_SAVED_FRAME'
    elif all(flags):
        out['status'] = 'COVERED'
    elif not any(flags):
        out['status'] = 'CLEAR'
    else:
        out['status'] = 'EDGE_UNKNOWN'
        out['reason'] = 'MIXED_POSITIVE_TAP_OPACITY'
    return out


def invented_checks():
    z = np.zeros((4, 5), bool)
    one = ~z
    require(alpha_relation(z, [1.25, 1.5])['status'] == 'CLEAR', 'Clear fourtap')
    require(alpha_relation(one, [1.25, 1.5])['status'] == 'COVERED', 'Covered fourtap')
    a = z.copy(); a[1, 1] = True
    require(alpha_relation(a, [1, 1])['status'] == 'COVERED', 'Integer uses one positive tap')
    require(alpha_relation(a, [1.25, 1.5])['status'] == 'EDGE_UNKNOWN', 'Mixed edge retained')
    require(alpha_relation(one, [-.25, 1])['status'] == 'UNKNOWN', 'Outside positive tap')
    require(alpha_relation(one, [4, 3])['status'] == 'COVERED', 'Last integer center valid')
    require(alpha_relation(one, None)['status'] == 'UNKNOWN', 'Missing reference')
    require(alpha_relation(one, [float('nan'), 1])['status'] == 'UNKNOWN', 'Nonfinite reference')
    # Near-integer fractional neighbors remain votes even if summed weight rounds.
    a = one.copy(); a[1, 2] = False
    require(alpha_relation(a, [1 + 2**-50, 1])['status'] == 'EDGE_UNKNOWN', 'No threshold/rounding shortcut')
    rejected = False
    try:
        safe('/never_touched/V112/payload.npz')
    except AssertionError:
        rejected = True
    require(rejected, 'Lexical forbidden path check')
    return {'completed': True, 'PASS': True, 'groups': 4, 'assertions': 10,
            'invented_arrays_only': True, 'real_payload_stat_hash_header_decode': 0,
            'tests': ['clear/opaque', 'integer/edge/outside/nonfinite',
                      'tiny positive tap retained without threshold', 'V112 before filesystem']}


def prepare():
    # Exact text only. Binary paths and original GT bindings are inert strings.
    s, points, pc, sc = read(SELECTION), lines(POINT_ROWS), read(POINT_COMPLETE), read(SOURCE_COMPLETE)
    require(pc['completed'] and pc['PASS'] and sc['completed'] and sc['PASS'], 'Completed source metadata')
    require(pc['selection'] == binding(SELECTION), 'Original V208 selection binding')
    require(pc['outputs'][POINT_ROWS.name] == binding(POINT_ROWS), 'Derived point rows binding')
    require(sc['code'] == binding(PRODUCER) and sc['candidate'] == binding(CANDIDATE)
            and sc['protocol'] == binding(SOURCE_PROTOCOL), 'Declared saved-alpha producer')
    require(sc['attempts'] == s['input_metadata_bindings']['synthetic_census'], 'Source census lineage')
    pmap = {(r['base_index'], r['native_slot']): r for r in points}
    originals = s['original_BEFORE_native8']
    require(len(points) == len(pmap) == len(originals) == 328, 'All original 41x8 points')
    for r in originals:
        p = pmap[r['base_index'], r['native_slot']]
        require(p['role'] == r['role'] == 'TRAIN', 'TRAIN original only')
        require(all(p[k] == r[k] for k in IDENTITY), 'Original q/L/native identity')
        require(p['canonical_index'] == r['declared_CF_to_canonical'], 'CF native canonical join')
        require(p['new_visibility_target_created'] is False and p['training_admission'] is False, 'No targets/admission')
    require(Counter(r['point0_combined_state'] for r in points) == {'D': 287, 'S': 41}, 'Saved BEFORE point authority')
    rows, packet_map, seen = [], {}, set()
    for r in s['physical_control_rows']:
        require(r['role'] == 'TRAIN' and r['control_join_key'] == ['SYNTH_FINITE_END', r['attempt_id'], r['variant'], r['native_slot']], 'Original TRAIN control key')
        key = tuple(r['control_join_key']); require(key not in seen, 'Unique physical row'); seen.add(key)
        require(r['variant'] in VARIANTS and r['native_slot'] in range(8), 'All original variant/native')
        p = pmap[r['source_base_index'], r['native_slot']]
        require(all(p[k] == r[k] for k in IDENTITY), 'Control to original q/L/U identity')
        b = r['source_packet']
        expect = str(V189 / 'synthetic_sampled_end_controls_v1' / r['attempt_id'] / 'CONTROL_TRIPLET_INPUTS.npz')
        require(b['path'] == expect and len(b['sha256']) == 64 and 'v112' not in expect.lower(), 'Exact packet literal without filesystem access')
        if r['attempt_id'] in packet_map:
            require(packet_map[r['attempt_id']]['binding'] == b, 'One unchanged packet per attempt')
        else:
            packet_map[r['attempt_id']] = {'attempt_id': r['attempt_id'], 'base_index': r['source_base_index'],
                'binding': b, 'raw_hw': p['raw_hw'], 'role': 'TRAIN',
                'allowed_npz_members_to_decode': ['foreign_alpha'], 'expected_dtype': 'bool',
                'expected_shape': [3, *p['raw_hw']], 'variant_order_from_frozen_producer': list(VARIANTS)}
        rows.append(dict(r, requested_native_slot=int(r['attempt_id'].rsplit('NATIVE', 1)[1]),
            canonical_index=p['canonical_index'], raw_hw=p['raw_hw'],
            saved_original_point0_target_alone_state=p['point0_target_alone_state'],
            saved_original_point0_combined_state=p['point0_combined_state'],
            saved_anchor_projected_reference_xy=p['saved_anchor_projected_reference_xy'],
            original_crop_reference_locality=p['original_crop_reference_locality'],
            saved_original_region_state_context_only=p['region_state'],
            source_point_authority='SAVED_ORIGINAL_BEFORE_POINT0_ONLY',
            new_native_point_state=None, new_visibility_target_created=False, training_admission=False))
    require(len(rows) == 1968 and len(packet_map) == 82 and len({r['source_base_index'] for r in rows}) == 41, 'Full selected census')
    counts = Counter((r['attempt_id'], r['variant']) for r in rows)
    require(len(counts) == 246 and set(counts.values()) == {8}, 'Every attempt all3/all8, no weak/status filtering')
    write(CHECKS, invented_checks())
    write(WHITELIST, {'completed': True, 'PASS': True, 'packets': list(packet_map.values()),
        'counts': {'TRAIN_packets': 82, 'physical_rows': 1968, 'parents': 41, 'original_points': 328},
        'payload_access_during_preparation': 0, 'decoded_members_future_only': ['foreign_alpha']})
    write_lines(ASSOCIATIONS, rows)
    sources = {k: binding(p) for k,p in [('scope',SCOPE),('selection',SELECTION),('point_complete',POINT_COMPLETE),
        ('point_rows',POINT_ROWS),('source_complete',SOURCE_COMPLETE),('source_producer',PRODUCER),
        ('source_candidate',CANDIDATE),('source_protocol',SOURCE_PROTOCOL)]}
    write(PROTOCOL, {'completed': True, 'code': binding(__file__), 'sources': sources,
        'whitelist': binding(WHITELIST), 'associations': binding(ASSOCIATIONS), 'checks': binding(CHECKS),
        'algorithm': 'At saved continuous fullimage reference, inspect every strictly positive float64 bilinear tap: all True COVERED; all False CLEAR; mixed EDGE_UNKNOWN; missing/nonfinite/outside UNKNOWN. Integer reference has one positive tap. No epsilon or new threshold.',
        'opacity_authority': 'V189 paste_plane: rectangle AND (~finite BEFORE camera_depth OR cover_front_z < BEFORE camera_depth - 64*eps*max(1,abs(z),abs(depth))); saved foreign_alpha is actual opaque foreground composite raster, with BEFORE all False. Depth ties are absent from alpha; no stored tie mask is decoded.',
        'interpretation': 'COVERED/CLEAR describe saved integer raster support, not re-rendered continuous ray or a new native D/E/S label. Original point0 D/S, input availability, fullimage/crop algebraic locality, requested vs other native slot remain separate. finite-end DED/region/owner/changedRGB are never used as point supervision.',
        'future_access': 'Root explicit actual entry only: opaque SHA256 of exactly82 containers, np.load allow_pickle=False and foreign_alpha member only; no RGB/raw96/valid/target/owner/depth/endmask/SAM/GT member or source binding followed.',
        'retention': 'All1968 selected physical TRAIN rows/82 attempts/41 original parents/native8/threevariants, including all old unadmitted rows; no model selection.',
        'new_labels_or_admission': False, 'fit_forward': 0, 'original_q_L_native_U_sampler_unchanged': True})
    false = {'approved_SOURCE_TRAIN_point_alpha_alignment': False, 'source_TRAIN_only': True,
        'automatic_retry': False, 'decode_only_foreign_alpha': True, 'new_labels_or_admission': False,
        'fit': False, 'model_forward': False, 'root_actual_scope': None,
        'scope': binding(SCOPE), 'code': binding(__file__), 'protocol': binding(PROTOCOL),
        'checks': binding(CHECKS), 'whitelist': binding(WHITELIST), 'associations': binding(ASSOCIATIONS),
        'output_dir': str(V / 'source_train_native_point_control_alignment_v1')}
    write(H / 'ROOT_ACTUAL_ENTRY_FALSE_TEMPLATE.json', false)
    write(H / 'FINAL_READY.json', {'completed': True, 'PASS': True, 'preparation_only': True,
        'code': binding(__file__), 'protocol': binding(PROTOCOL), 'checks': binding(CHECKS),
        'whitelist': binding(WHITELIST), 'associations': binding(ASSOCIATIONS),
        'false_entry': binding(H / 'ROOT_ACTUAL_ENTRY_FALSE_TEMPLATE.json'),
        'counts': {'TRAIN_packets':82, 'physical_rows':1968, 'parents':41, 'original_points':328,
                   'saved_BEFORE_point_D':287, 'saved_BEFORE_point_S':41},
        'actual_payload_stat_hash_header_decode': 0, 'original_GT_reopened': 0,
        'new_labels_or_training_admission': False, 'fit_forward': 0})
    print(json.dumps({'completed': True, 'PASS': True, 'ready': binding(H/'FINAL_READY.json')}))


def run(entry_path):
    e = read(entry_path)
    require(e.get('approved_SOURCE_TRAIN_point_alpha_alignment') is True and e.get('source_TRAIN_only') is True,
            'Separate root actual SOURCE TRAIN authorization')
    require(e.get('automatic_retry') is False and e.get('decode_only_foreign_alpha') is True
            and e.get('new_labels_or_admission') is False and e.get('fit') is False
            and e.get('model_forward') is False, 'Bounded diagnostic only')
    require(e['root_actual_scope'] is not None, 'Separate root actual scope required')
    root_scope = verify(e['root_actual_scope'])
    require(root_scope.parent == V and root_scope.suffix == '.json', 'Current phase root scope only')
    for k,p in [('scope',SCOPE),('code',Path(__file__)),('protocol',PROTOCOL),('checks',CHECKS),
                ('whitelist',WHITELIST),('associations',ASSOCIATIONS)]:
        require(e[k] == binding(p), 'Frozen ' + k)
    protocol = read(PROTOCOL)
    for b in protocol['sources'].values():
        verify(b)  # Text/code only; no nested payload/GT bindings followed.
    require(read(CHECKS)['PASS'] is True, 'Frozen invented checks')
    whitelist, rows = read(WHITELIST), lines(ASSOCIATIONS)
    require(len(whitelist['packets']) == 82 and len(rows) == 1968, 'Complete fixed census')
    out = safe(e['output_dir'])
    require(out.parent == V and out != H and not out.exists(), 'Fresh current phase output child')
    out.mkdir()
    write(out/'STARTED.json', {'entry':binding(entry_path),'code':binding(__file__),
                              'protocol':binding(PROTOCOL),'automatic_retry':False})
    done, loaded = [], []
    by_attempt = defaultdict(list)
    for row in rows:
        by_attempt[row['attempt_id']].append(row)
    try:
        for packet in whitelist['packets']:
            b = packet['binding']; name = packet['attempt_id']
            expected = str(V189/'synthetic_sampled_end_controls_v1'/name/'CONTROL_TRIPLET_INPUTS.npz')
            require(b['path'] == expected and packet['allowed_npz_members_to_decode'] == ['foreign_alpha'], 'Exact single-member authority')
            require(digest(b['path'], binary=True) == b['sha256'], 'Opaque exact source container SHA')
            with np.load(safe(b['path']), allow_pickle=False) as archive:
                alpha = archive['foreign_alpha']  # The ONLY archive member accessed.
            require(alpha.dtype == np.bool_ and list(alpha.shape) == packet['expected_shape'], 'Saved fullimage boolean opacity shape')
            require(not alpha[0].any(), 'Declared BEFORE has no new foreground')
            for row in by_attempt[name]:
                relation = alpha_relation(alpha[VARIANTS.index(row['variant'])], row['saved_anchor_projected_reference_xy'])
                done.append(dict(row, foreground_alpha_relation=relation,
                    original_input_U_preserved=not row['original_query_available']))
            loaded.append({'binding':b, 'decoded_members':['foreign_alpha'], 'shape':list(alpha.shape)})
        # Preserve original selection order independent of packet iteration.
        order = {r['row_id']:i for i,r in enumerate(rows)}
        done.sort(key=lambda r:order[r['row_id']])
        require([r['row_id'] for r in done] == [r['row_id'] for r in rows], 'Every original row retained in original order')
        result = out/'ALL1968_TRAIN_NATIVE_POINT_ALPHA_ALIGNMENT.jsonl'; write_lines(result,done)
        summaries = {}
        for variant in VARIANTS:
            rr = [r for r in done if r['variant'] == variant]
            summaries[variant] = {'rows':len(rr), 'all_native':dict(Counter(r['foreground_alpha_relation']['status'] for r in rr)),
                'requested_native':dict(Counter(r['foreground_alpha_relation']['status'] for r in rr if r['native_slot']==r['requested_native_slot'])),
                'by_original_point0_state': {state:dict(Counter(r['foreground_alpha_relation']['status'] for r in rr if r['saved_original_point0_combined_state']==state)) for state in ('D','S','UNKNOWN')}}
        summary = {'completed':True,'PASS':True,'rows':1968,'packets':82,'parents':41,'original_points':328,
            'variants':summaries,'original_inputU_rows':sum(not r['original_query_available'] for r in done),
            'interpretation':protocol['interpretation'],'new_native_point_states_created':0,'new_labels_or_admission':False,
            'no_GT_reopen_RGB_owner_depth_targets_endmask_SAM_or_model_decode':True}
        write(out/'SUMMARY.json', summary)
        write(out/'COMPLETE.json',dict(summary,entry=binding(entry_path),code=binding(__file__),
            protocol=binding(PROTOCOL),whitelist=binding(WHITELIST),associations=binding(ASSOCIATIONS),
            decoded_packet_members=loaded,outputs={result.name:binding(result),'SUMMARY.json':binding(out/'SUMMARY.json')}))
        print(json.dumps({'completed':True,'PASS':True,'completion':binding(out/'COMPLETE.json'),'variants':summaries}))
    except BaseException:
        write_lines(out/'PARTIAL_ROWS_PRESERVED.jsonl', done)
        write(out/'FAILED_NO_AUTOMATIC_RETRY.json',{'completed':False,'traceback':traceback.format_exc(),'automatic_retry':False})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare', action='store_true')
    group.add_argument('--entry')
    args = parser.parse_args()
    if args.prepare:
        prepare()
    else:
        run(args.entry)
