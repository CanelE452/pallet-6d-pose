"""Independent scalar source-point audit, then saved-format and aggregate audit."""
from pathlib import Path
from collections import Counter
import datetime
import hashlib
import json
import math
import sys
import time

import numpy as np

V = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
OWNER = ('INCIDENT_TARGET_DECLARED_CF', 'OTHER_TARGET', 'FOREIGN', 'BG', 'AMBIGUOUS', 'INVALID_SUPPORT')
JOINT = ('VISIBLE_TARGET', 'EXTERNAL_OVER_BEFORE_TARGET', 'FOREIGN_WITHOUT_BEFORE_TARGET',
         'BACKGROUND_BOTH', 'AMBIGUOUS_EITHER', 'INCONSISTENT_CHANGE', 'INVALID_SUPPORT')
FLAG_KEYS = ('query_available', 'factual_query_available', 'feature_constructed', 'crop_constructed',
             'context_available', 'Q_inside', 'RGB_central4_available', 'native_continuous_valid',
             'native_nearest_valid')
FACTOR_KEYS = ('material', 'aspect', 'geometry_family', 'foreign_family', 'signed_quarter')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def bind(path):
    return {'path': str(path), 'sha256': sha(path)}


def read(path):
    return json.loads(Path(path).read_text())


def read_lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def check_binding(b):
    assert sha(b['path']) == b['sha256'], b['path']


def present(q):
    return q is not None and len(q) == 2 and all(math.isfinite(float(z)) for z in q) and list(q) != [-1., -1.]


def raw_cell(arrays, x, y):
    if not (0 <= x < arrays['part_id'].shape[1] and 0 <= y < arrays['part_id'].shape[0]):
        return None
    return (int(arrays['part_id'][y, x]), int(arrays['global_triangle_id'][y, x]),
            float(arrays['camera_depth'][y, x]), bool(arrays['ambiguity'][y, x]),
            int(arrays['tie_count'][y, x]))


def owner_category(cell, incident):
    if cell is None:
        return 'INVALID_SUPPORT'
    part, _, _, ambiguous, _ = cell
    if ambiguous:
        return 'AMBIGUOUS'
    if part == 999:
        return 'FOREIGN'
    if part < 0:
        return 'BG'
    return 'INCIDENT_TARGET_DECLARED_CF' if part in incident else 'OTHER_TARGET'


def causal_category(b, s):
    # Scalar case analysis from the frozen definition, independently of joint_map.
    if b is None or s is None:
        return 'INVALID_SUPPORT', None
    bp, bt, bz, ba, _ = b
    sp, st, sz, sa, _ = s
    if ba or sa:
        return 'AMBIGUOUS_EITHER', 'NONE'
    if bp == 999:
        return 'INCONSISTENT_CHANGE', 'BEFORE_FOREIGN_UNEXPECTED'
    before_target = bp >= 0
    scene_target = sp >= 0 and sp != 999
    if before_target:
        if sp == 999:
            if not (math.isfinite(bz) and bz > 0 and math.isfinite(sz) and sz > 0):
                return 'INCONSISTENT_CHANGE', 'INVALID_FOREIGN_OR_BEFORE_DEPTH'
            return ('EXTERNAL_OVER_BEFORE_TARGET', 'NONE') if sz < bz else ('INCONSISTENT_CHANGE', 'EXTERNAL_NOT_STRICTLY_FRONT')
        if sp < 0:
            return 'INCONSISTENT_CHANGE', 'TARGET_DISAPPEARED_TO_BG'
        if sp != bp:
            return 'INCONSISTENT_CHANGE', 'TARGET_PART_CHANGED'
        if st != bt:
            return 'INCONSISTENT_CHANGE', 'TARGET_TRIANGLE_CHANGED'
        if not (math.isfinite(bz) and bz > 0 and math.isfinite(sz) and sz > 0 and bz == sz):
            return 'INCONSISTENT_CHANGE', 'TARGET_DEPTH_CHANGED_OR_INVALID'
        return 'VISIBLE_TARGET', 'NONE'
    if sp < 0:
        return 'BACKGROUND_BOTH', 'NONE'
    if sp == 999:
        if math.isfinite(sz) and sz > 0:
            return 'FOREIGN_WITHOUT_BEFORE_TARGET', 'NONE'
        return 'INCONSISTENT_CHANGE', 'INVALID_FOREIGN_OR_BEFORE_DEPTH'
    if scene_target:
        return 'INCONSISTENT_CHANGE', 'BACKGROUND_TO_TARGET'
    return 'INCONSISTENT_CHANGE', 'OTHER_UNCLASSIFIED'


def pixel_details(arrays, x, y, incident):
    cell = raw_cell(arrays, x, y)
    result = dict(x=x, y=y, in_image=cell is not None, category=owner_category(cell, incident))
    if cell is None:
        result.update(part_id=None, global_triangle_id=None, camera_depth=None, ambiguity=None, tie_count=None)
    else:
        part, triangle, depth, ambiguous, ties = cell
        result.update(part_id=part, global_triangle_id=triangle,
                      camera_depth=depth if math.isfinite(depth) else None, ambiguity=ambiguous, tie_count=ties)
    return result


def direct_point(before, scene, q, incident):
    if not present(q):
        missing = dict(status='UNKNOWN_ORIGINAL_QUERY_MISSING', nearest=None, bilinear=None)
        return missing, dict(missing), None, None
    x, y = map(float, q)
    x0, y0 = math.floor(x), math.floor(y)
    dx, dy = x-x0, y-y0
    weights_s = {k: 0. for k in OWNER}
    weights_b = {k: 0. for k in OWNER}
    weights_j = {k: 0. for k in JOINT}
    supports_s, supports_b = [], []
    for ay in (0, 1):
        for ax in (0, 1):
            w = (dx if ax else 1-dx)*(dy if ay else 1-dy)
            px, py = x0+ax, y0+ay
            s = pixel_details(scene, px, py, incident)
            b = pixel_details(before, px, py, incident)
            joint_class, _ = causal_category(raw_cell(before, px, py), raw_cell(scene, px, py))
            weights_s[s['category']] += w
            weights_b[b['category']] += w
            weights_j[joint_class] += w
            supports_s.append(dict(weight=w, **s))
            supports_b.append(dict(weight=w, **b))
    nx, ny = math.floor(x+.5), math.floor(y+.5)
    nc, reason = causal_category(raw_cell(before, nx, ny), raw_cell(scene, nx, ny))
    def detailed(arrays, masses, footprint):
        basic = dict(TARGET=masses[OWNER[0]]+masses[OWNER[1]], FOREIGN=masses['FOREIGN'],
                     BG=masses['BG'], AMBIGUOUS=masses['AMBIGUOUS'], INVALID_SUPPORT=masses['INVALID_SUPPORT'])
        return dict(status='MEASURED_ORIGINAL_COORDINATE', coordinate_xy=[x, y],
                    nearest=pixel_details(arrays, nx, ny, incident),
                    bilinear=dict(owner5_plus_INVALID=masses, basic4_plus_INVALID=basic),
                    four_source_pixels=footprint, reference_centered_crop_generated=False)
    nearest_j = dict(class_name=nc, inconsistent_reason=reason, raw_pixel_xy=[nx, ny])
    return detailed(scene, weights_s, supports_s), detailed(before, weights_b, supports_b), weights_j, nearest_j


def numeric_exact(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def source_arrays(binding, used):
    check_binding(binding)
    with np.load(binding['path'], allow_pickle=False) as packet:
        arrays = {k: packet[k].copy() for k in ('part_id', 'ambiguity', 'camera_depth', 'global_triangle_id', 'tie_count')}
    assert arrays['part_id'].dtype == np.int32 and arrays['ambiguity'].dtype == bool
    assert arrays['camera_depth'].dtype == np.float64 and arrays['global_triangle_id'].dtype == np.int32
    assert all(a.shape == (512, 512) for a in arrays.values())
    used[binding['path']] = binding
    return arrays


def matches_group(r, split, group):
    if split != 'ALL' and r['split'] != split:
        return False
    if group == 'ALL':
        return True
    parts = group.split('::')
    kind = parts[0]
    if kind == 'native_variant':
        return r['native_slot'] == int(parts[1]) and r['variant'] == parts[2]
    if kind == 'native_slot':
        return r['native_slot'] == int(parts[1])
    if kind == 'variant':
        return r['variant'] == parts[1]
    if kind == 'semantic':
        return r['original_semantic_state'] == parts[1]
    return str(r['factors'][kind]) == parts[1]


def main():
    started = time.monotonic()
    complete_path = V/'literal_raster_census_v1/COMPLETE.json'
    assert sha(complete_path) == '025ea05d354404046bcaf92cef0af29653c77defd7e0a4c9faa7d9115fd29a96'
    complete = read(complete_path)
    protocol_path = V/'census_preparation/PROTOCOL_BEFORE_FULL_SOURCE_MASK_BODIES.json'
    protocol = read(protocol_path)
    check_binding(complete['protocol'])
    text_sources = {k: protocol[k] for k in ('original_queries', 'original_crop_facts', 'original_GT', 'original_scene_GT', 'original_scene_input', 'inventory', 'scene_inventory')}
    for b in text_sources.values():
        check_binding(b)
    queries = read_lines(protocol['original_queries']['path'])
    facts = read(protocol['original_crop_facts']['path'])['rows']
    # Filter by declared convention explicitly, rather than copying the producer's stride selection.
    primary = [r for r in read_lines(protocol['original_GT']['path']) if r['convention'] == 'CF_PRIMARY_DECLARED']
    gt_scenes = read(protocol['original_scene_GT']['path'])['rows']
    input_scenes = read(protocol['original_scene_input']['path'])['rows']
    inventory = read_lines(protocol['inventory']['path'])
    scene_inventory = read_lines(protocol['scene_inventory']['path'])
    assert len(queries) == len(facts) == len(primary) == len(inventory) == 6144
    assert len(gt_scenes) == len(input_scenes) == len(scene_inventory) == 768
    original = []
    for i, (q, f, t, r) in enumerate(zip(queries, facts, primary, inventory)):
        si, slot = divmod(i, 8)
        g, scene_input = gt_scenes[si], input_scenes[si]
        record = g['records_CF'][slot]
        assert q['scene_index'] == t['scene_index'] == si and q['native_slot'] == t['native_slot'] == slot
        assert f['occurrence_index'] == i and q['query_id'] == f['occurrence_id'] == t['query_id']
        assert q['query_xy'] == f['query_xy'] == t['query_xy']
        assert q['roi_side'] == f['L'] == t['roi_side']
        assert record['state'] == t['teacher_state'] == g['state8_CF'][slot]
        assert record['canonical_index'] == t['canonical_index'] == g['permCF_to_canonical'][slot]
        expected = dict(original_global_index=i, original_occurrence_id=q['query_id'], scene_index=si,
                        base_index=g['base_index'], native_slot=slot, split=g['split'], variant=g['variant'],
                        factors=scene_input['factors'], query_xy=q['query_xy'], L=q['roi_side'],
                        original_query_metadata=q, original_input_flags={k: f[k] for k in FLAG_KEYS},
                        selected_index=f['selected_index'], original_semantic_state=t['teacher_state'],
                        declared_canonical_index=t['canonical_index'], nominal_reference_xy=t['reference_xy'],
                        correspondence_boundary=t['correspondence_boundary'], position_band=t['position_band'],
                        q_reference_error_px=t['error_px'], incident_part_ids=record['incident_part_ids'],
                        primary_part_id=record['primary_part_id'], active_witness_indices=record['active_witness_indices'],
                        rgb96_SHA_inherited=f['rgb96_sha256'], valid96_SHA_inherited=f['valid96_sha256'], training_admission=False)
        assert r == expected, i
        original.append(expected)
    assert sum(not r['original_input_flags']['query_available'] for r in original) == 88
    assert sum(r['query_xy'] is None for r in original) == 56
    assert sum(r['split'] == 'TRAIN' for r in original) == 4096
    derived = []
    point_arrays = {name: np.full((6144, n), np.nan, np.float64) for name, n in [('scene_owner_point', 6), ('BEFORE_owner_point', 6), ('joint_point', 7)]}
    nearest_j = np.full(6144, -1, np.int8)
    used = {}
    for base in range(192):
        before_gt = gt_scenes[base*4]
        assert before_gt['variant'] == 'BEFORE'
        before = source_arrays(before_gt['raw_buffers'], used)
        for vi in range(4):
            si = base*4+vi
            gt = gt_scenes[si]
            assert scene_inventory[si]['scene_buffer'] == gt['raw_buffers']
            assert scene_inventory[si]['BEFORE_buffer'] == before_gt['raw_buffers']
            scene = before if vi == 0 else source_arrays(gt['raw_buffers'], used)
            for slot in range(8):
                index = si*8+slot
                r = original[index]
                s, b, joint, nearest = direct_point(before, scene, r['query_xy'], set(r['incident_part_ids']))
                derived.append(dict(original_global_index=index, scene_owner_point_details=s,
                                    BEFORE_owner_point_details=b, joint_nearest=nearest, joint_point=joint))
                if joint is not None:
                    point_arrays['scene_owner_point'][index] = list(s['bilinear']['owner5_plus_INVALID'].values())
                    point_arrays['BEFORE_owner_point'][index] = list(b['bilinear']['owner5_plus_INVALID'].values())
                    point_arrays['joint_point'][index] = list(joint.values())
                    nearest_j[index] = JOINT.index(nearest['class_name'])
        if (base+1) % 32 == 0:
            print(json.dumps(dict(independent_source_bases=base+1, source_rows=len(derived))), flush=True)
    assert len(used) == 768 and len(derived) == 6144
    for b in used.values():
        check_binding(b)
    # Seal source-derived evidence BEFORE reading saved result bodies.
    independent_path = OUT/'INDEPENDENT_SOURCE_POINT_RESULTS.jsonl'
    with independent_path.open('x') as stream:
        for r in derived:
            stream.write(json.dumps(r, allow_nan=False, separators=(',', ':'))+'\n')
    source_points_path = OUT/'INDEPENDENT_SOURCE_POINT_ARRAYS.npz'
    np.savez_compressed(source_points_path, **point_arrays, nearest_joint_class_index=nearest_j)

    check_binding(complete['rows'])
    saved = read_lines(complete['rows']['path'])
    assert len(saved) == 6144
    soft_from_json = {}
    for branch, n in [('scene_owner', 6), ('BEFORE_owner', 6), ('joint', 7)]:
        for sample in ('point', 'central4', 'central32'):
            soft_from_json[branch+'_'+sample] = np.full((6144, n), np.nan, np.float64)
    for i, (source, measured, orig) in enumerate(zip(derived, saved, original)):
        for key, value in orig.items():
            assert measured[key] == value, (i, key)
        assert measured['registration_authority'] == protocol['authority']
        assert measured['literal_raster_diagnostic_only'] is True
        assert measured['scene_owner_point_details'] == source['scene_owner_point_details'], (i, 'scene point')
        assert measured['BEFORE_owner_point_details'] == source['BEFORE_owner_point_details'], (i, 'before point')
        assert measured['joint_nearest'] == source['joint_nearest'], (i, 'joint nearest')
        for branch in ('scene_owner', 'BEFORE_owner', 'joint'):
            for sample in ('point', 'central4', 'central32'):
                value = measured['soft_distributions'][branch][sample]
                if value is not None:
                    soft_from_json[branch+'_'+sample][i] = value
    for key, expected in point_arrays.items():
        assert numeric_exact(expected, soft_from_json[key]), key

    all_npz = {}
    for split in ('TRAIN', 'VALIDATION'):
        check_binding(complete['sidecars'][split])
        with np.load(complete['sidecars'][split]['path'], allow_pickle=False) as z:
            packet = {key: z[key].copy() for key in z.files}
        ids = packet['original_global_index'].tolist()
        expected_ids = [i for i, r in enumerate(original) if r['split'] == split]
        assert ids == expected_ids
        assert packet['training_admission'].item() is False
        assert packet['initializer_physical_native_identity_certified'].item() is False
        assert packet['owner_class_order'].tolist() == list(OWNER)
        assert packet['joint_class_order'].tolist() == list(JOINT)
        for key in soft_from_json:
            expected = soft_from_json[key][expected_ids]
            assert numeric_exact(packet[key], expected), (split, key)
            if key not in all_npz:
                all_npz[key] = np.full_like(soft_from_json[key], np.nan)
            all_npz[key][ids] = packet[key]
        for local, i in enumerate(ids):
            r = original[i]
            for key in ('original_global_index', 'original_occurrence_id', 'scene_index', 'base_index', 'native_slot', 'split', 'variant', 'original_semantic_state', 'declared_canonical_index', 'correspondence_boundary'):
                assert packet[key][local].item() == r[key], (i, key)
            for key in FLAG_KEYS:
                assert packet[key][local].item() == r['original_input_flags'][key]
            for key in FACTOR_KEYS:
                assert packet[key][local].item() == r['factors'][key]
            assert packet['query_xy_missing'][local] == (r['query_xy'] is None)
            assert packet['L_missing'][local] == (r['L'] is None)
            assert packet['selected_index_missing'][local] == (r['selected_index'] is None)
            assert packet['nominal_reference_xy'][local].tolist() == r['nominal_reference_xy']
            if r['query_xy'] is None:
                assert np.isnan(packet['query_xy'][local]).all()
            else:
                assert packet['query_xy'][local].tolist() == r['query_xy']
            if r['L'] is None:
                assert math.isnan(float(packet['L'][local]))
            else:
                assert packet['L'][local] == r['L']
            assert packet['selected_index'][local] == (-1 if r['selected_index'] is None else r['selected_index'])
            assert packet['nearest_joint_class_index'][local] == nearest_j[i]

    check_binding(complete['aggregates'])
    aggregate = read(complete['aggregates']['path'])
    groups = aggregate['groups']
    max_sum_diff = max_mean_diff = 0.
    combinations = set()
    for g in groups:
        key = (g['split'], g['group'])
        assert key not in combinations
        combinations.add(key)
        ids = [i for i, r in enumerate(original) if matches_group(r, *key)]
        assert g['N'] == len(ids)
        assert g['original_inputU'] == sum(not original[i]['original_input_flags']['query_available'] for i in ids)
        assert g['semantic_counts'] == {s: sum(original[i]['original_semantic_state'] == s for i in ids) for s in 'DESU'}
        actual_nearest = Counter('UNKNOWN_MISSING' if nearest_j[i] < 0 else JOINT[int(nearest_j[i])] for i in ids)
        assert g['nearest_joint_counts'] == {name: actual_nearest.get(name, 0) for name in ('UNKNOWN_MISSING',)+JOINT}
        for name, array in all_npz.items():
            stats = g['soft_measures'][name]
            measured_ids = [i for i in ids if np.isfinite(array[i]).all()]
            assert stats['measured_N'] == len(measured_ids) and stats['missing_N'] == len(ids)-len(measured_ids)
            if not measured_ids:
                assert stats['sum_weights'] is None and stats['mean_weights'] is None
            else:
                for column in range(array.shape[1]):
                    # Stable independent scalar summation; allow only reduction-order rounding.
                    total = math.fsum(float(array[i, column]) for i in measured_ids)
                    mean = total/len(measured_ids)
                    ds = abs(total-stats['sum_weights'][column])
                    dm = abs(mean-stats['mean_weights'][column])
                    max_sum_diff, max_mean_diff = max(max_sum_diff, ds), max(max_mean_diff, dm)
                    assert ds <= 5e-10 and dm <= 2e-13, (key, name, column, ds, dm)
    expected_groups = {'ALL'}
    for field in ('native_slot', 'variant', 'semantic')+FACTOR_KEYS:
        values = range(8) if field == 'native_slot' else {r['original_semantic_state'] if field == 'semantic' else r['factors'][field] if field in FACTOR_KEYS else r[field] for r in original}
        expected_groups.update(field+'::'+str(value) for value in values)
    expected_groups.update(f'native_variant::{slot}::{variant}' for slot in range(8) for variant in ('BEFORE', 'COVER', 'NEAR_SHAM', 'FAR_SHAM'))
    assert combinations == {(split, group) for split in ('ALL', 'TRAIN', 'VALIDATION') for group in expected_groups}
    for b in text_sources.values():
        check_binding(b)
    for b in used.values():
        check_binding(b)
    report = dict(schema='V173_FULL_SOURCE_INDEPENDENT_SAVED_CENSUS_REVIEW_A_v1', completed=True, PASS=True,
                  UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(), code=bind(Path(__file__).resolve()),
                  complete=bind(complete_path), protocol=bind(protocol_path), source_metadata_bindings=text_sources,
                  saved_rows=complete['rows'], sidecars=complete['sidecars'], aggregate=complete['aggregates'],
                  independent_source_points=bind(independent_path), independent_source_point_arrays=bind(source_points_path),
                  method='Scalar Python case analysis and explicit four-source-pixel loop, no producer/helper imports. All source-derived point results sealed before saved JSONL/NPZ result bodies were loaded. Group totals independently recomputed with math.fsum.',
                  coverage=dict(original_buffers=768, original_rows=6144, TRAIN=4096, VALIDATION=2048,
                                missing_q_L_selected=56, original_inputU=88, measured_q_points=6088,
                                nearest_owner_pairs_checked=6088*2, nearest_joint_checked=6088,
                                four_pixel_source_pairs_checked=6088*4, source_point_soft_arrays_exact=3,
                                NPZ_JSONL_soft_arrays_exact=9, fixed_split_sidecars=2,
                                aggregate_group_rows=len(groups), aggregate_sum_max_abs_difference=max_sum_diff,
                                aggregate_mean_max_abs_difference=max_mean_diff,
                                original_metadata_qL_native_split_flags_null_semantic_all_preserved=True,
                                original_source768_hashes_before_after_and_final_match=True,
                                source_label_metadata_hashes_unchanged=True),
                  original_source_buffer_bindings=list(used.values()), runtime=dict(python=sys.executable, numpy=np.__version__),
                  limitations=['Only nearest and original-q four-source-pixel owner/joint values were independently resampled. central4/central32 were checked for exact saved NPZ-versus-JSONL parity and aggregate arithmetic, not independently resampled.',
                               'Full-frame raw joint pixel/reason diagnostic counts were not independently recomputed; this audit covers source q support plus all6144 saved rows and group tables.',
                               'INCIDENT membership remains relative to declared CF mapping; physical initializer native identity at CF/V4 boundary remains uncertified. Raster point conjunction does not rewrite original finite-witness D/E/S/U.'],
                  new_label_threshold_admission_model_fit_forward_render_private_current_V169_V112=0,
                  selected_model=None, fully_solved=False, seconds=time.monotonic()-started)
    report_path = OUT/'REVIEW.json'
    assert not report_path.exists()
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(PASS=True, review=bind(report_path), coverage=report['coverage'])), flush=True)


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        import traceback
        failure = OUT/'AUDIT_FAILURE_PRESERVED.json'
        if not failure.exists():
            failure.write_text(json.dumps(dict(completed=False, traceback=traceback.format_exc()), indent=2)+'\n')
        raise
