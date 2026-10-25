"""V254: exact saved SOURCE scene-CF audit. No image/model/feature access."""
import argparse
import collections
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import struct
import sys

import numpy as np

PROTOCOL_SHA = 'a36c2b65c779b71110261bca43237612d4042a458a733fa01bbbadd991215668'
FAMILIES = ('SELF_SCENE15', 'BEFORE_ANCHOR4')
VARIANTS = ('BEFORE', 'COVER', 'NEAR_SHAM', 'FAR_SHAM')
SHAMS = ('NEAR_SHAM', 'FAR_SHAM')
MODELS = ('BASE', 'CONTROL', 'TREAT')
INPUT_KEYS = ('saved_scores', 'samples', 'groups', 'all_rows_join', 'point_mapping', 'render_input_scenes')
SAMPLE_KEYS = ('row_id', 'family', 'target', 'fit', 'available', 'condition')


def guard(path, fresh=False):
    p = Path(path)
    if not p.is_absolute() or any(s == '..' or any(t in s.lower() for t in ('v112', 'cache', '.pyc')) for s in p.parts):
        raise ValueError('Forbidden or nonabsolute artifact path')
    for a in reversed((p,) + tuple(p.parents)):
        if a.is_symlink():
            raise ValueError('Symlink artifact/ancestor')
    if fresh and p.exists():
        raise ValueError('Output must be fresh')
    return p


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read_binding(binding):
    p = guard(binding['path'])
    data = p.read_bytes()
    assert len(data) == binding['size_bytes'] and digest(data) == binding['sha256'], str(p)
    return data


def binding(path):
    p = guard(path)
    b = p.read_bytes()
    return {'path': str(p), 'sha256': digest(b), 'size_bytes': len(b)}


def canonical(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def bool_value(x):
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if str(x).lower() in ('true', '1'):
        return True
    if str(x).lower() in ('false', '0'):
        return False
    raise ValueError('Not a bool')


def fp_bytes(value):
    """Decoded numeric identity: nullable tree + LE FP64, never epsilon/JSON spelling."""
    if value is None:
        return b'N'
    if isinstance(value, (list, tuple)):
        return b'L' + struct.pack('<I', len(value)) + b''.join(fp_bytes(v) for v in value)
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)):
        raise ValueError('Expected nullable numeric value')
    return b'F' + struct.pack('<d', float(value))


def same_fp(values):
    return len({fp_bytes(v) for v in values}) == 1


def tri(values):
    if any(v is False for v in values):
        return False
    return None if any(v is None for v in values) else True


def status(value):
    return 'UNKNOWN' if value is None else ('PASS' if value else 'FAIL')


def group_fingerprint(g):
    assert len(g['q8']) == len(g['L8']) == len(g['original_available8']) == 8
    for q in g['q8']:
        assert q is None or len(q) == 2
    assert all(isinstance(v, bool) for v in g['original_available8'])
    payload = (fp_bytes(g['q8']) + fp_bytes(g['L8']) +
               np.asarray(g['original_available8'], dtype=np.bool_).tobytes() +
               fp_bytes(g['box_xyxy']) + fp_bytes(g['raw_hw']))
    return digest(payload)


def validate_census(groups, rows, samples):
    """Complete original groups/rows first, with aliases and masked rows intact."""
    assert len(groups) == 1767 and len(rows) == 15040
    assert [g['group_index'] for g in groups] == list(range(1767))
    assert len({g['group_id'] for g in groups}) == 1767
    assert [r['full_row_index'] for r in rows] == list(range(15040))
    assert len({r['row_id'] for r in rows}) == 15040
    assert set(samples) == set(SAMPLE_KEYS)
    for key in SAMPLE_KEYS:
        assert len(samples[key]) == 7617
    assert samples['condition'].shape == (7617, 32) and samples['condition'].dtype == np.float32
    assert samples['family'].dtype == samples['target'].dtype == np.int8
    assert samples['fit'].dtype == samples['available'].dtype == np.bool_
    assert np.isfinite(samples['condition']).all()
    assert set(np.unique(samples['target'])) <= {0, 1}
    assert set(np.unique(samples['family'])) == {0, 1}
    assert len(set(map(str, samples['row_id']))) == 7617
    slot_counts = collections.defaultdict(collections.Counter)
    known = {}
    group_sha = {g['group_index']: group_fingerprint(g) for g in groups}
    by_row = {}
    for r in rows:
        g = groups[r['group_index']]
        k = r['active_slot']
        assert isinstance(k, int) and 0 <= k < 8
        assert g['group_id'] == r['group_id'] and g['family_code'] == r['family_code']
        assert same_fp((g['q8'][k], r['q'])) and same_fp((g['L8'][k], r['L']))
        assert g['original_available8'][k] is r['original_query_available']
        slot_counts[r['group_index']][k] += 1
        by_row[r['row_id']] = r
        si = r['sample_index']
        if si is not None:
            assert isinstance(si, int) and 0 <= si < 7617 and si not in known
            assert str(samples['row_id'][si]) == r['row_id']
            assert int(samples['family'][si]) == r['family_code']
            assert int(samples['target'][si]) == r['target']
            assert bool(samples['fit'][si]) is r['fit']
            assert bool(samples['available'][si]) is r['crop_available']
            assert r['original_state'] == ('D' if r['target'] == 0 else 'E')
            if r['family_code'] == 0:
                assert r['native_slot'] == k and r['logical_aux_slot'] is None
                expected = np.zeros(8, np.float32)
                expected[k] = 1
                assert samples['condition'][si, :8].tobytes() == expected.tobytes()
            else:
                assert r['logical_aux_slot'] == k and r['native_slot'] is None
                assert not samples['condition'][si].any()
            known[si] = r
    assert set(known) == set(range(7617))
    assert all(set(slot_counts[i]) == set(range(8)) for i in range(1767))
    family_counts = collections.Counter(r['family_code'] for r in rows)
    assert family_counts == {0: 7040, 1: 8000}
    assert sum(r['sample_index'] is None for r in rows) == 7423
    assert sum(bool(x) for x in samples['fit']) == 5876
    assert sum(not bool(x) for x in samples['fit']) == 1741
    census = {
        'all_original_rows': len(rows), 'all_original_groups': len(groups),
        'known_samples': len(known), 'masked_rows_retained': 7423,
        'family_rows': dict(sorted(family_counts.items())),
        'original_state_counts': dict(sorted(collections.Counter(r['original_state'] for r in rows).items())),
        'source_role_counts': dict(sorted(collections.Counter(r['internal_source_role'] for r in rows).items())),
        'groups_all_active_slots_0_to_7_complete': True,
        'groups_with_alias_rows': sum(sum(c.values()) > 8 for c in slot_counts.values()),
        'extra_alias_memberships': sum(sum(c.values()) - 8 for c in slot_counts.values()),
        'exactly_8_rows_per_group_required': False,
        'known_condition32_finite_and_SYN_native_alignment': True,
        'census_before_any_known_or_score_filter': True,
        'typed_numeric_comparison': 'Decoded values cast LE FP64, nullable tree preserved; prior F32->F64 promotion preserved; tolerance0, not JSON-text equality',
    }
    return by_row, known, group_sha, census


def build_syn_rows(point_mapping, scenes, by_row, groups, group_sha, samples):
    assert len(point_mapping) == 7040
    assert len({m['row_id'] for m in point_mapping}) == 7040
    scene_map = {s['scene_index']: s for s in scenes}
    assert len(scene_map) == len(scenes)
    syn = []
    for m in point_mapping:
        r = by_row[m['row_id']]
        assert r['family_code'] == 0 and m['family'] in FAMILIES
        assert m['state_name'] == 'NOMINAL' and m['state_index'] == 0
        assert m['variant'] in VARIANTS
        k, scene = m['native_slot'], m['scene_index']
        assert 0 <= k < 8 and r['active_slot'] == r['native_slot'] == k
        g = groups[r['group_index']]
        s = scene_map.get(scene)
        checks = {}
        checks['actual_scene_variant_order'] = scene == 4 * m['base_index'] + VARIANTS.index(m['variant'])
        checks['actual_image_teacher_key'] = (
            m.get('source_teacher_key') == [scene, k] and m.get('actual_image_scene_index') == scene and
            m.get('actual_image_variant') == m['variant'] and
            m.get('source_image_original_global_index') == 8 * scene + k and
            m.get('actual_image_original_global_index') == 8 * scene + k and
            m.get('mapping_status') == 'EXACT_ORIGINAL_IMAGE_OWN_SCENE')
        checks['row_mapping_active_input'] = (
            same_fp((r['q'], m['query_xy'])) and same_fp((r['L'], m['L'])) and
            r['original_query_available'] is m['original_query_available'])
        anchor = m['anchor_scene_index']
        checks['anchor_identity'] = m['anchor_original_global_index'] == 8 * anchor + k
        if m['family'] == 'SELF_SCENE15':
            checks['anchor_identity'] &= anchor == scene and m['anchor_variant'] == m['variant']
        else:
            checks['anchor_identity'] &= anchor == 4 * m['base_index'] and m['anchor_variant'] == 'BEFORE'
        if s is None:
            checks['render_scene_identity'] = None
            checks['actual_image_binding'] = None
            render_hw_equal = None
        else:
            checks['render_scene_identity'] = s['base_index'] == m['base_index'] and s['variant'] == m['variant']
            image = g['image']
            inert = m.get('actual_image_source_binding_inert')
            full = s.get('full_RGB')
            checks['actual_image_binding'] = (
                isinstance(inert, dict) and isinstance(full, dict) and
                image.get('path') == inert.get('path') == full.get('path') and
                image.get('sha256') == inert.get('sha256') == full.get('sha256') and
                image.get('format') == 'npz' and image.get('member') == s.get('full_RGB_key') and
                s.get('full_RGB_format') == 'NPZ_RGB_UINT8_HWC')
            render_hw_equal = same_fp((g['raw_hw'], s.get('raw_hw')))
        declared = m.get('declared_CF_to_canonical')
        checks['declared_target_mapping_present'] = None if declared is None else isinstance(declared, int) and 0 <= declared < 8
        si = r['sample_index']
        condition_sha = None if si is None else digest(np.asarray(samples['condition'][si], dtype='<f4').tobytes())
        record = {
            'row_id': r['row_id'], 'full_row_index': r['full_row_index'], 'source_index': r['source_index'],
            'sample_index': si, 'augmentation_family': m['family'], 'sample_family': 0,
            'base_index': m['base_index'], 'native_slot': k, 'variant': m['variant'],
            'scene_index': scene, 'actual_image_scene_index': m.get('actual_image_scene_index'),
            'anchor_scene_index': anchor, 'declared_CF_to_canonical': declared,
            'fit': r['fit'], 'source_role': 'FIT' if r['fit'] else 'DEV',
            'internal_source_role': r['internal_source_role'], 'component_id': r['component_id'],
            'original_state': r['original_state'], 'target': r['target'],
            'original_query_available': r['original_query_available'], 'crop_available': r['crop_available'],
            'point0_target_alone_state': m.get('point0_target_alone_state', 'UNKNOWN'),
            'point0_combined_state': m.get('point0_combined_state', 'UNKNOWN'),
            'group_index': r['group_index'], 'group_id': r['group_id'],
            'group_input_fingerprint': group_sha[r['group_index']], 'condition32_sha256': condition_sha,
            'row_lineage': tri(list(checks.values())), 'row_checks': checks,
            'group_raw_hw': g['raw_hw'], 'render_raw_hw': None if s is None else s.get('raw_hw'),
            'group_vs_render_raw_hw_nullable_exact': render_hw_equal,
            'group_vs_render_raw_hw_status': 'UNKNOWN_GROUP_UNDECLARED' if g['raw_hw'] is None else status(render_hw_equal),
            'image_sha256': g['image']['sha256'],
            '_group': g, '_mapping': m, '_scene': s,
        }
        syn.append(record)
    assert {r['row_id'] for r in syn} == {r['row_id'] for r in by_row.values() if r['family_code'] == 0}
    return sorted(syn, key=lambda r: r['source_index'])


def attach_scores(syn, score_rows, samples):
    """Called only after complete group census and all7040 masked records exist."""
    assert len(score_rows) == 7617
    score_map = {}
    for score in score_rows:
        si = int(score['sample_index'])
        assert si not in score_map and 0 <= si < 7617
        assert score['row_id'] == str(samples['row_id'][si])
        assert int(score['family']) == int(samples['family'][si])
        assert int(score['target']) == int(samples['target'][si])
        assert score['role'] == ('FIT' if samples['fit'][si] else 'DEV')
        assert bool_value(score['available']) == bool(samples['available'][si])
        for model in MODELS:
            pred = int(score[model + '_prediction'])
            logit = float(score[model + '_E_logit'])
            selected = float(score[model + '_logit' + str(int(samples['family'][si]))])
            assert math.isfinite(logit) and logit == selected
            assert pred == (int(logit >= 0) if samples['available'][si] else -1)
        score_map[si] = score
    assert set(score_map) == set(range(7617))
    for r in syn:
        si = r['sample_index']
        r['saved_score_present'] = si is not None
        for model in MODELS:
            r[model + '_prediction'] = None if si is None else int(score_map[si][model + '_prediction'])
            r[model + '_E_logit'] = None if si is None else float(score_map[si][model + '_E_logit'])


def validate_triplet(members):
    """Separate declaration, geometry, known condition, labels, roles and scores."""
    if any(len(v) != 1 for v in members):
        return {'members_unique': False, 'lineage': None, 'input_exact': None,
                'condition32_exact': None, 'role': 'UNKNOWN', 'same_declared_component': None,
                'D_E_D_labels': False, 'groupIDs_equal_diagnostic_only': None,
                'actual_image_SHAs_different_diagnostic_only': None,
                'saved_reference_nullable_exact_diagnostic_only': None, 'axes': {},
                'reasons': ['ABSENT_OR_DUPLICATE_MEMBER']}
    rr = [v[0] for v in members]
    gg = [r['_group'] for r in rr]
    ss = [r['_scene'] for r in rr]
    mm = [r['_mapping'] for r in rr]
    configs = [None if s is None else s.get('source_base_config') for s in ss]
    mapping = [r['declared_CF_to_canonical'] for r in rr]
    config_present = all(isinstance(c, dict) and isinstance(c.get('file'), dict) and
                         c['file'].get('sha256') and c.get('row_index') is not None for c in configs)
    canonical_present = all(v is not None for v in mapping)
    lineage_axes = {
        'all_member_lineage': tri([r['row_lineage'] for r in rr]),
        'common_base': len({r['base_index'] for r in rr}) == 1,
        'common_native_address': len({r['native_slot'] for r in rr}) == 1,
        'common_declared_target_mapping': None if not canonical_present else len(set(mapping)) == 1,
        'same_declared_base_config_camera': None if not config_present else len({canonical(c) for c in configs}) == 1,
        'same_declared_style_factors': None if any(s is None for s in ss) else len({canonical([s.get('style'), s.get('factors')]) for s in ss}) == 1,
    }
    geometry_axes = {
        'q8_FP64_nullable_exact': same_fp([g['q8'] for g in gg]),
        'L8_FP64_nullable_exact': same_fp([g['L8'] for g in gg]),
        'availability8_bool_exact': len({np.asarray(g['original_available8'], np.bool_).tobytes() for g in gg}) == 1,
        'box_xyxy_FP64_nullable_exact': same_fp([g['box_xyxy'] for g in gg]),
        'raw_hw_nullable_exact': same_fp([g['raw_hw'] for g in gg]),
        'active_q_FP64_nullable_exact': same_fp([m['query_xy'] for m in mm]),
        'active_L_FP64_nullable_exact': same_fp([m['L'] for m in mm]),
        'active_U_original_exact': len({m['original_query_available'] for m in mm}) == 1,
        'active_native_exact': len({m['native_slot'] for m in mm}) == 1,
    }
    cond = [r['condition32_sha256'] for r in rr]
    condition = None if any(v is None for v in cond) else len(set(cond)) == 1
    geometry_axes['known_condition32_FP32_exact'] = condition
    role = 'FIT' if all(r['fit'] for r in rr) else ('DEV' if all(not r['fit'] for r in rr) else 'MIXED')
    component = None if any(r['component_id'] is None for r in rr) else len({r['component_id'] for r in rr}) == 1
    axes = {**lineage_axes, **geometry_axes}
    return {
        'members_unique': True, 'lineage': tri(list(lineage_axes.values())),
        'input_exact': tri(list(geometry_axes.values())), 'condition32_exact': condition,
        'role': role, 'same_declared_component': component,
        'D_E_D_labels': [r['original_state'] for r in rr] == ['D', 'E', 'D'] and [r['target'] for r in rr] == [0, 1, 0],
        'groupIDs_equal_diagnostic_only': len({r['group_id'] for r in rr}) == 1,
        'actual_image_SHAs_different_diagnostic_only': len({r['image_sha256'] for r in rr}) == 3,
        'saved_reference_nullable_exact_diagnostic_only': same_fp([m.get('saved_projected_reference_xy') for m in mm]),
        'axes': axes,
        'reasons': [k + ':' + status(v) for k, v in axes.items() if v is not True] +
                   ([] if role == 'FIT' else ['NOT_ALL_MEMBER_FIT']) +
                   ([] if component is True else ['COMPONENT_IDENTITY:' + status(component)]),
    }


def build_candidates(syn):
    cells = collections.defaultdict(list)
    bases = sorted({r['base_index'] for r in syn})
    for r in syn:
        cells[(r['augmentation_family'], r['base_index'], r['native_slot'], r['variant'])].append(r)
    candidates, responses = [], []
    for fam in FAMILIES:
        for base in bases:
            for native in range(8):
                for sham in SHAMS:
                    variants = ('BEFORE', 'COVER', sham)
                    members = [cells[(fam, base, native, v)] for v in variants]
                    check = validate_triplet(members)
                    rr = [v[0] if len(v) == 1 else None for v in members]
                    cid = fam + ':' + str(base) + ':' + str(native) + ':' + sham
                    cand = {'candidate_id': cid, 'augmentation_family': fam, 'base_index': base,
                            'native_slot': native, 'sham': sham, **check}
                    for v, rows in zip(variants, members):
                        prefix = 'SHAM' if v in SHAMS else v
                        cand[prefix + '_member_count'] = len(rows)
                        cand[prefix + '_row_ids'] = [r['row_id'] for r in rows]
                        cand[prefix + '_sample_indices'] = [r['sample_index'] for r in rows]
                        cand[prefix + '_states'] = [r['original_state'] for r in rows]
                        cand[prefix + '_group_indices'] = [r['group_index'] for r in rows]
                    for model in MODELS:
                        saved = all(r is not None and r[model + '_prediction'] in (0, 1) and
                                    r['saved_score_present'] and r['crop_available'] and
                                    r['original_query_available'] for r in rr)
                        exact_scored = saved and check['lineage'] is True and check['input_exact'] is True
                        primary = exact_scored and check['role'] == 'FIT' and check['D_E_D_labels']
                        z = [None if r is None else r[model + '_E_logit'] for r in rr]
                        pred = [None if r is None else r[model + '_prediction'] for r in rr]
                        resp = {
                            'candidate_id': cid, 'augmentation_family': fam, 'base_index': base,
                            'native_slot': native, 'sham': sham, 'role': check['role'], 'model': model,
                            'members_unique': check['members_unique'], 'lineage': check['lineage'],
                            'input_exact': check['input_exact'], 'D_E_D_labels': check['D_E_D_labels'],
                            'all_saved_scores_available': saved, 'exact_scored_triplet': exact_scored,
                            'primary_FIT_D_E_D': primary,
                            'BEFORE_E_logit': z[0], 'COVER_E_logit': z[1], 'SHAM_E_logit': z[2],
                            'BEFORE_prediction': pred[0], 'COVER_prediction': pred[1], 'SHAM_prediction': pred[2],
                            'prediction_sequence': '/'.join('U' if p is None or p == -1 else ('D' if p == 0 else 'E') for p in pred),
                            'predicted_D_E_D': None if not exact_scored else pred == [0, 1, 0],
                        }
                        for name, a, b in (('cover_before', 1, 0), ('sham_before', 2, 0), ('cover_sham', 1, 2)):
                            resp[name] = None if not exact_scored else z[a] - z[b]
                        responses.append(resp)
                    candidates.append(cand)
    return candidates, responses


def distribution(values):
    x = np.asarray(values, dtype=np.float64)
    if len(x) == 0:
        return {'n': 0, 'mean': None, 'median': None, 'min': None, 'q25': None, 'q75': None, 'max': None}
    assert np.isfinite(x).all()
    return {'n': len(x), 'mean': float(x.mean()), 'median': float(np.median(x)),
            'min': float(x.min()), 'q25': float(np.quantile(x, .25)),
            'q75': float(np.quantile(x, .75)), 'max': float(x.max())}


def summarize(responses):
    out = []
    for fam in FAMILIES:
        for sham in SHAMS:
            for role in ('ALL', 'FIT', 'DEV', 'MIXED', 'UNKNOWN'):
                for native in ('ALL',) + tuple(range(8)):
                    for model in MODELS:
                        rows = [r for r in responses if r['augmentation_family'] == fam and r['sham'] == sham and
                                r['model'] == model and (role == 'ALL' or r['role'] == role) and
                                (native == 'ALL' or r['native_slot'] == native)]
                        primary = [r for r in rows if r['primary_FIT_D_E_D']]
                        d = {'augmentation_family': fam, 'sham': sham, 'role': role, 'native_slot': native, 'model': model,
                             'all_candidates': len(rows), 'members_unique': sum(r['members_unique'] for r in rows),
                             'lineage_PASS': sum(r['lineage'] is True for r in rows),
                             'input_exact_PASS': sum(r['input_exact'] is True for r in rows),
                             'input_exact_UNKNOWN': sum(r['input_exact'] is None for r in rows),
                             'original_D_E_D_labels': sum(r['D_E_D_labels'] for r in rows),
                             'all_saved_scores_available': sum(r['all_saved_scores_available'] for r in rows),
                             'exact_scored_triplets_any_role': sum(r['exact_scored_triplet'] for r in rows),
                             'primary_FIT_D_E_D': len(primary),
                             'predicted_D_E_D': sum(r['predicted_D_E_D'] is True for r in primary),
                             'predicted_D_E_D_rate': None if not primary else sum(r['predicted_D_E_D'] is True for r in primary) / len(primary)}
                        for name in ('cover_before', 'sham_before', 'cover_sham'):
                            values = [r[name] for r in primary]
                            for stat, value in distribution(values).items():
                                d[name + '_' + stat] = value
                            for sign, op in (('positive', lambda x: x > 0), ('zero', lambda x: x == 0), ('negative', lambda x: x < 0)):
                                d[name + '_' + sign] = sum(op(x) for x in values)
                        for stat, value in distribution([abs(r['sham_before']) for r in primary]).items():
                            d['abs_sham_before_' + stat] = value
                        d['primary_prediction_sequences'] = dict(sorted(collections.Counter(r['prediction_sequence'] for r in primary).items()))
                        out.append(d)
    return out


def audit(payloads):
    """Pure computation; payloads supplied by explicit caller, never follow pointers."""
    assert set(payloads) == set(INPUT_KEYS)
    groups, rows, samples = payloads['groups'], payloads['all_rows_join'], payloads['samples']
    by_row, known, group_sha, census = validate_census(groups, rows, samples)
    scene_doc = payloads['render_input_scenes']
    assert scene_doc['completed'] is True
    syn = build_syn_rows(payloads['point_mapping'], scene_doc['rows'], by_row, groups, group_sha, samples)
    census['SYN7040_preserved_before_saved_score_join'] = len(syn)
    census['SYN_masked_rows_retained'] = sum(r['sample_index'] is None for r in syn)
    assert census['SYN_masked_rows_retained'] == 944
    census['SYN_state_counts'] = dict(sorted(collections.Counter(r['original_state'] for r in syn).items()))
    census['SYN_point0_combined_state_counts'] = dict(sorted(collections.Counter(r['point0_combined_state'] for r in syn).items()))
    attach_scores(syn, payloads['saved_scores'], samples)
    candidates, responses = build_candidates(syn)
    summaries = summarize(responses)
    for r in syn:
        for k in ('_group', '_mapping', '_scene'):
            del r[k]
    overview = [r for r in summaries if r['role'] == 'ALL' and r['native_slot'] == 'ALL']
    summary = {
        'completed': True, 'audit_kind': 'SAVED_SOURCE_CF_RESPONSE_NO_NEW_FORWARD', 'census': census,
        'models': list(MODELS), 'primary': 'all-member FIT, original D/E/D, exact declared lineage/input, available saved score',
        'overview': overview, 'candidate_count_including_missing': len(candidates),
        'candidate_axis_counts': {key: dict(sorted(collections.Counter(status(c.get(key)) for c in candidates).items()))
                                  for key in ('lineage', 'input_exact', 'condition32_exact')},
        'candidate_role_counts': dict(sorted(collections.Counter(c['role'] for c in candidates).items())),
        'reason_counts': dict(sorted(collections.Counter(reason for c in candidates for reason in c['reasons']).items())),
        'limits': [
            'NEAR and FAR strata share BEFORE/COVER; their counts are correlated and must not be added as independent scenes.',
            'SELF own-initializer and BEFORE-anchor input policies are separate; no epsilon/permutation/coordinate repair.',
            'Declared native address and synthetic point0 witness are not newly certified physical/native/wholepart/q-owner gold.',
            'Missing masked condition32 and missing scores stay UNKNOWN/null; COVER/SHAM names do not generate labels.',
            'No RGB/GT archive/model/checkpoint/feature/encoder read or inference; SOURCE-derived teacher metadata is read.',
            'SOURCE DEV metadata/saved outputs are decoded for fixed census/audit, not for model/data/threshold/recipe selection.',
            'Stored output threshold remains raw Elogit>=0; this audit creates no adoption or current/held improvement claim.',
        ],
        'new_optimizer_updates': 0, 'new_model_forwards': 0,
        'SOURCE_derived_teacher_metadata_read': True, 'current_or_held_payloads': 0,
    }
    return {'census': census, 'syn_rows': syn, 'candidates': candidates,
            'responses': responses, 'summaries': summaries, 'summary': summary}


def cell(value):
    return canonical(value) if isinstance(value, (dict, list, tuple)) else value


def write_results(output_dir, result):
    out = guard(output_dir, fresh=True)
    out.mkdir(parents=True)
    def js(name, value):
        (out / name).write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + '\n')
    def table(name, rows):
        assert rows
        keys = list(rows[0])
        assert all(set(r) == set(keys) for r in rows)
        with (out / name).open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=keys, lineterminator='\n')
            w.writeheader()
            w.writerows({k: cell(r[k]) for k in keys} for r in rows)
    js('ALL15040_CENSUS.json', result['census'])
    table('SYN7040_ROWS.csv', result['syn_rows'])
    table('CF_TRIPLETS.csv', result['candidates'])
    table('RESPONSE_MODEL_ROWS.csv', result['responses'])
    table('RESPONSE_SUMMARY.csv', result['summaries'])
    js('SUMMARY.json', result['summary'])
    return {name: binding(out / name) for name in (
        'ALL15040_CENSUS.json', 'SYN7040_ROWS.csv', 'CF_TRIPLETS.csv',
        'RESPONSE_MODEL_ROWS.csv', 'RESPONSE_SUMMARY.csv', 'SUMMARY.json')}


def run(inputs_bindings, output_dir):
    """Public replay API: six explicit paths only; core outputs path/time independent."""
    assert set(inputs_bindings) == set(INPUT_KEYS)
    raw = {key: read_binding(inputs_bindings[key]) for key in INPUT_KEYS}
    payloads = {}
    for key in ('groups', 'all_rows_join', 'point_mapping'):
        payloads[key] = [json.loads(line) for line in raw[key].decode('utf-8').splitlines() if line]
    payloads['render_input_scenes'] = json.loads(raw['render_input_scenes'])
    with np.load(io.BytesIO(raw['samples']), allow_pickle=False) as data:
        payloads['samples'] = {key: data[key] for key in SAMPLE_KEYS}
    payloads['saved_scores'] = list(csv.DictReader(io.StringIO(raw['saved_scores'].decode('utf-8'))))
    result = audit(payloads)
    outputs = write_results(output_dir, result)
    complete = {
        'completed': True, 'PASS': True, 'kind': 'SAVED_VALUE_AUDIT_PIPELINE_NOT_MODEL_PERFORMANCE_PASS',
        'inputs6': inputs_bindings, 'outputs': outputs,
        'summary': result['summary'], 'new_training': False, 'new_forward': False,
        'SOURCE_DEV_nonpixel_metadata_and_saved_outputs_loaded': True,
        'SOURCE_DEV_used_for_fit_or_selection': False,
        'current_or_held_payloads': 0, 'raw_RGB_GT_archive_weights_model_feature_reads': 0,
        'SOURCE_derived_teacher_metadata_read': True,
    }
    out = guard(output_dir)
    (out / 'COMPLETE.json').write_text(json.dumps(complete, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + '\n')
    return complete


def authorize(entry_path):
    entry = json.loads(guard(entry_path).read_text())
    if entry.get('execution_authorized') is not True:
        raise PermissionError('Root true entry required before any SOURCE input access')
    assert set(entry) == {'execution_authorized', 'code', 'root_protocol', 'inputs6', 'output_dir'}
    assert read_binding(entry['code']) == guard(__file__).read_bytes()
    assert Path(entry['code']['path']) == Path(__file__)
    pbytes = read_binding(entry['root_protocol'])
    assert digest(pbytes) == PROTOCOL_SHA
    protocol = json.loads(pbytes)
    assert set(entry['inputs6']) == set(INPUT_KEYS)
    for key in INPUT_KEYS:
        expected = protocol['inputs6'][key]
        assert entry['inputs6'][key] == {k: expected[k] for k in ('path', 'sha256', 'size_bytes')}
    assert entry['output_dir'] == protocol['output_dir']
    guard(entry['output_dir'], fresh=True)
    return entry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--entry', required=True)
    args = parser.parse_args()
    entry = authorize(args.entry)
    result = run(entry['inputs6'], entry['output_dir'])
    print(json.dumps({'completed': result['completed'], 'outputs': result['outputs'], 'new_forward': False}, sort_keys=True))


if __name__ == '__main__':
    main()
