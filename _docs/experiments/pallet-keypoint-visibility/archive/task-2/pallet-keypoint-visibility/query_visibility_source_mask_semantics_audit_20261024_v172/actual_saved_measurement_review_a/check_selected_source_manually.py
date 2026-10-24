"""Independent saved-source point audit. No producer imports or model calls."""
from pathlib import Path
import datetime
import hashlib
import json
import math
import sys

import numpy as np

V = Path(__file__).resolve().parents[1]
NAMES = ('INCIDENT_TARGET', 'OTHER_TARGET', 'FOREIGN', 'BG', 'AMBIGUOUS', 'INVALID_SUPPORT')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def binding(path):
    return {'path': str(path), 'sha256': digest(path)}


def checked_read(path):
    return json.loads(Path(path).read_text())


def manual_pixel(arrays, x, y, incident):
    height, width = arrays['part_id'].shape
    if x < 0 or x >= width or y < 0 or y >= height:
        return {'x': x, 'y': y, 'in_image': False, 'category': 'INVALID_SUPPORT',
                'part_id': None, 'global_triangle_id': None, 'camera_depth': None,
                'ambiguity': None, 'tie_count': None}
    owner = int(arrays['part_id'][y, x])
    ambiguous = bool(arrays['ambiguity'][y, x])
    if ambiguous:
        category = 'AMBIGUOUS'
    elif owner == 999:
        category = 'FOREIGN'
    elif owner < 0:
        category = 'BG'
    elif owner in incident:
        category = 'INCIDENT_TARGET'
    else:
        category = 'OTHER_TARGET'
    depth = float(arrays['camera_depth'][y, x])
    return {'x': x, 'y': y, 'in_image': True, 'category': category,
            'part_id': owner, 'global_triangle_id': int(arrays['global_triangle_id'][y, x]),
            'camera_depth': depth if math.isfinite(depth) else None,
            'ambiguity': ambiguous, 'tie_count': int(arrays['tie_count'][y, x])}


def manual_point(arrays, coordinate, incident):
    # Inputs come from original readiness/geometry metadata, never saved measurements.
    x, y = map(float, coordinate)
    x0, y0 = math.floor(x), math.floor(y)
    dx, dy = x-x0, y-y0
    masses = {name: 0.0 for name in NAMES}
    footprint = []
    # Explicit four-source-pixel loop, not the producer's vectorized helper.
    for ay in (0, 1):
        for ax in (0, 1):
            wx = dx if ax else 1.0-dx
            wy = dy if ay else 1.0-dy
            weight = wx*wy
            cell = manual_pixel(arrays, x0+ax, y0+ay, incident)
            masses[cell['category']] += weight
            footprint.append({'weight': weight, **cell})
    assert math.isclose(sum(masses.values()), 1.0, abs_tol=2e-15)
    return {'coordinate_xy': [x, y],
            'nearest': manual_pixel(arrays, math.floor(x+.5), math.floor(y+.5), incident),
            'four_source_pixels': footprint, 'owner5_plus_INVALID': masses,
            'basic4_plus_INVALID': {'TARGET': masses['INCIDENT_TARGET']+masses['OTHER_TARGET'],
                                   'FOREIGN': masses['FOREIGN'], 'BG': masses['BG'],
                                   'AMBIGUOUS': masses['AMBIGUOUS'],
                                   'INVALID_SUPPORT': masses['INVALID_SUPPORT']}}


def compare_point(manual, saved):
    assert saved['coordinate_xy'] == manual['coordinate_xy']
    assert saved['nearest'] == manual['nearest']
    assert saved['four_source_pixels'] == manual['four_source_pixels']
    assert saved['bilinear']['owner5_plus_INVALID'] == manual['owner5_plus_INVALID']
    assert saved['bilinear']['basic4_plus_INVALID'] == manual['basic4_plus_INVALID']
    assert saved['reference_centered_crop_generated'] is False


def main():
    ready_path = V/'INPUT_PATH_READINESS.json'
    complete_path = V/'saved_mask_semantics_measurement_v1/COMPLETE.json'
    protocol_path = V/'measurement_preparation/PROTOCOL_BEFORE_SOURCE_MASK_DEPTH_BODIES.json'
    ready = checked_read(ready_path)
    protocol = checked_read(protocol_path)
    complete = checked_read(complete_path)
    selection = checked_read(ready['source_selection']['path'])
    witnesses = {row['global_index']: row for row in selection['unique_witnesses']}
    assert digest(complete_path) == '12d2d8375de1e573f021536176be81ed4aae1f0c10ece179deb2b5782dbd2b78'
    assert complete['readiness'] == binding(ready_path)
    assert complete['protocol'] == binding(protocol_path)
    source_computations = []
    source_hashes = []
    # Compute all independent source results before loading saved row results.
    for scene in ready['scenes']:
        index = scene['global_index']
        slot = index % 8
        original_query = scene['all8_original_input_queries'][slot]
        metadata = scene['projected_native_metadata']
        record = metadata['records_CF_metadata'][slot]
        source_specs = (
            ('SCENE', scene['selected_combined_scene_buffer']['inherited_binding']),
            ('BEFORE_TARGET_ONLY', scene['saved_target_only_all_parts_buffer']['inherited_binding']))
        for which, source in source_specs:
            source_path = Path(source['path'])
            assert digest(source_path) == source['sha256']
            with np.load(source_path, allow_pickle=False) as packet:
                keys = ('part_id', 'ambiguity', 'camera_depth', 'global_triangle_id', 'tie_count',
                        'native8_CF_xy', 'native8_canonical_xy', 'permCF_to_canonical', 'state8_CF',
                        'camera_K', 'camera_R', 'camera_t')
                arrays = {key: packet[key].copy() for key in keys}
                rgb_sha = None
                if which == 'SCENE':
                    rgb_sha = hashlib.sha256(np.ascontiguousarray(packet['RGB_uint8']).tobytes()).hexdigest()
            assert arrays['part_id'].dtype == np.int32
            assert arrays['ambiguity'].dtype == np.bool_
            assert arrays['camera_depth'].dtype == np.float64
            assert arrays['native8_CF_xy'].tolist() == metadata['native8_CF_xy']
            assert arrays['permCF_to_canonical'].tolist() == metadata['permCF_to_canonical']
            assert np.array_equal(arrays['native8_canonical_xy'][arrays['permCF_to_canonical']],
                                  arrays['native8_CF_xy'])
            for key in ('K', 'R', 't'):
                assert arrays['camera_'+key].tolist() == metadata['camera'][key]
            gt = checked_read(scene['original_metadata']['GT_scenes']['path'])
            variant = scene['variant'] if which == 'SCENE' else 'BEFORE'
            original_gt = next(row for row in gt['rows'] if row['variant'] == variant)
            assert arrays['state8_CF'].tolist() == original_gt['state8_CF']
            if which == 'SCENE':
                original_rgb_sha = witnesses[index]['decoded_RGB_sha256_inherited']
                assert rgb_sha == original_rgb_sha
            q_point = manual_point(arrays, original_query['query_xy'], record['incident_part_ids'])
            ref_point = manual_point(arrays, metadata['native8_CF_xy'][slot], record['incident_part_ids'])
            source_computations.append({'global_index': index, 'scene_index': scene['scene_index'],
                                        'buffer': which, 'source_binding': source,
                                        'selected_RGB_decoded_sha256': rgb_sha,
                                        'original_q_point': q_point,
                                        'nominal_reference_point': ref_point})
            assert digest(source_path) == source['sha256']
            source_hashes.append(source)

    rowfile = Path(complete['rows']['path'])
    assert digest(rowfile) == complete['rows']['sha256']
    saved_rows = [json.loads(line) for line in rowfile.read_text().splitlines() if line]
    assert len(saved_rows) == 24
    saved_index = {row['original_global_index']: row for row in saved_rows}
    assert len(saved_index) == 24
    original_flags = {row['occurrence_index']: row for row in protocol['all24_original_input_flags']}
    for scene in ready['scenes']:
        for slot, qrow in enumerate(scene['all8_original_input_queries']):
            index = scene['scene_index']*8+slot
            row = saved_index[index]
            record = scene['projected_native_metadata']['records_CF_metadata'][slot]
            assert row['original_qL_metadata'] == qrow
            assert row['original_input_flags'] == original_flags[index]
            assert row['original_inputU'] == (not original_flags[index]['query_available'])
            assert row['original_semantic_record_CF'] == record
            assert row['canonical_index'] == record['canonical_index']
            assert row['incident_part_ids'] == record['incident_part_ids']
            assert row['nominal_reference_xy'] == record['projected_reference_xy']
            assert row['association_authority'] == protocol['association_authority']
            assert row['new_label'] is None and row['training_admitted'] is False
            assert row['query_id'] == qrow['query_id']
            assert row['variant'] == qrow['variant']
            assert row['is_original_selected_witness'] == (index == scene['global_index'])
    for source in source_computations:
        saved = saved_index[source['global_index']]['measurements'][source['buffer']]
        compare_point(source['original_q_point'], saved['original_q_point'])
        compare_point(source['nominal_reference_point'], saved['nominal_reference_diagnostic_point_only'])
    assert sum(row['is_original_selected_witness'] for row in saved_rows) == 3
    assert sum(row['original_inputU'] for row in saved_rows) == complete['original_inputU_preserved'] == 0
    report = {'schema': 'V172_INDEPENDENT_ACTUAL_SAVED_MEASUREMENT_REVIEW_A_v1',
              'completed': True, 'PASS': True, 'UTC': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'code': binding(Path(__file__).resolve()), 'complete': binding(complete_path),
              'readiness': binding(ready_path), 'protocol': binding(protocol_path),
              'saved24_rows': binding(rowfile),
              'source_definition_review': protocol['source_definition_review'],
              'runtime': {'python': sys.executable, 'numpy': np.__version__},
              'method': 'Independent scalar Python4-loop directly from original six NPZ part_id/ambiguity/depth/triangle/tie arrays; original q/reference/incident sets from bound source metadata. Source computations completed before saved24 result load. No measurement helper imported or copied.',
              'checked': {'original_buffers_hashed_before_and_after': 6,
                          'selected_scene_decoded_RGB_hashes_compared': 3,
                          'direct_points': 12, 'nearest_scalar_comparisons': 12,
                          'four_source_pixel_comparisons': 48,
                          'six_class_bilinear_distributions_exact': 12,
                          'all24_original_identity_qL_semantic_flags_preserved': True,
                          'all24_projection_perm_camera_and_state_metadata_preserved': True,
                          'original_inputU_preserved': 0,
                          'all24_count_and_three_selected_retained': True,
                          'central32_recomputed': False},
              'original_buffer_bindings': source_hashes, 'independent_source_point_results': source_computations,
              'blocking_issues': [],
              'limitations': ['INCIDENT_TARGET is relative to declared CF association; initializer physical native identity remains uncertified at CF/V4 boundary.',
                              'Point ownership is distinct from original registered finite-ray native D/E/S state. No original semantic/q/L/mapping correction, labels or admission created.',
                              'This audit checks selected3 q/reference points on SCENE+BEFORE, not all24 crop grids or all central32 samples.'],
              'new_models_features_forward_fit_render_private_current_V112_or_coordinate_selection': 0}
    output = V/'actual_saved_measurement_review_a/REVIEW.json'
    assert not output.exists()
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'path': str(output), 'sha256': digest(output), 'PASS': True}))


if __name__ == '__main__':
    main()
