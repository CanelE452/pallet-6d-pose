"""V177 metadata-only readiness. Never opens image/array/feature/score bodies."""
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

P = Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility')
V = P / 'query_visibility_public_held_blind_center_owner_review_20261024_v177'
S = P / 'query_visibility_mr6d_full_public_source_20261024_v153'
OUT = V / 'metadata_readiness_c'
SCOPE = V / 'ROOT_SCOPE_BEFORE_PUBLIC_HELD_PIXELS_OR_NEW_OWNER_ANNOTATIONS.json'
EXPECTED_SCOPE = 'c2018965ace4b5cbe4d4ef9a0ddc816d10514c73f4db3a0ff13c9f8a76ca9c70'


def sha(b):
    return hashlib.sha256(b).hexdigest()


def metadata(path, expected=None, jsonl=False):
    path = Path(path)
    assert path.suffix in ('.json', '.jsonl', '.py')
    b = path.read_bytes()
    if expected is not None:
        assert sha(b) == expected, str(path)
    binding = {'path': str(path), 'sha256': sha(b)}
    if path.suffix == '.py':
        return b.decode(), binding
    return ([json.loads(line) for line in b.decode().splitlines() if line]
            if jsonl else json.loads(b)), binding


def pathstat(path):
    """Path availability only: no open, header read, or body hash."""
    path = Path(path)
    st = path.stat()
    assert path.is_file() and st.st_size > 0
    return {'exists': True, 'regular_file': True, 'bytes': st.st_size,
            'body_or_header_read': False, 'sha256_recomputed': False}


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                               allow_nan=False) + '\n')


def main():
    scope, scope_b = metadata(SCOPE, EXPECTED_SCOPE)
    facts, facts_b = metadata(scope['source_feature_facts']['path'],
                              scope['source_feature_facts']['sha256'], True)
    queries, queries_b = metadata(scope['source_queries']['path'],
                                  scope['source_queries']['sha256'], True)
    complete, complete_b = metadata(S / 'full230_native_DINO_v1/FEATURE_COMPLETE.json',
                                    '8e3a0ca35b04bf0834f96140ec2c9b3d018f6652552680ee0ddbedf6dddc9b70')
    protocol, protocol_b = metadata(complete['protocol']['path'], complete['protocol']['sha256'])
    frames, frames_b = metadata(complete['inputs']['path'], complete['inputs']['sha256'])
    crops, crops_b = metadata(complete['crop_manifest']['path'], complete['crop_manifest']['sha256'])
    rgb, rgb_b = metadata(S / 'full_source_v1/RGB230_INPUT_ONLY_MANIFEST.json',
                          '437ceec1608d5bec77f1d567c6aa912ce0b2ab91eddf2e12078874704d1b5141')
    r0, r0_b = metadata(complete['source_initializer']['path'], complete['source_initializer']['sha256'])
    code, code_b = metadata(complete['code']['path'], complete['code']['sha256'])
    _, local_b = metadata(protocol['local_crop_code']['path'], protocol['local_crop_code']['sha256'])
    _, context_b = metadata(protocol['target_crop_code']['path'], protocol['target_crop_code']['sha256'])
    public, public_b = metadata(P / 'query_visibility_direct_query_owner_control_20261024_v175/source_fit_preparation/PROTOCOL_BEFORE_SOURCE_FEATURE_OR_TRAIN_SOFT_BODIES.json',
                                '5899feb545bd04c6cd2476f9978347002420ab20674c0f3e215513f135b1e2a2')
    assert complete['completed'] and complete['PASS'] and complete['rows'] == 1840
    assert len(facts) == len(queries) == 1840
    assert len(frames['frames']) == len(rgb['rows']) == len(crops['sheets']) == 230
    assert frames['context_box_mode'] == complete['context_box_mode'] == 'FROZEN_INITIAL_Q8_UNCLIPPED_UNEXPANDED_MINMAX'
    assert 'rgb96=hwc,valid96=valid,global_index=np.arange(i*8,i*8+8,dtype=np.int32),crop_constructed=construct' in code
    assert "available=qdata['continuous_valid'].copy()" in code
    assert 'occurrence_unit=np.arange(N,dtype=np.int32)' in code
    held = [r for r in facts if r['split'] == 'SOURCE_VALIDATION']
    assert len(held) == 360
    frame_ids = sorted({r['frame_index'] for r in held})
    assert frame_ids == list(range(73, 118))
    by_frame = {r['frame_index']: r for r in frames['frames']}
    rgb_by_frame = {r['frame_index']: r for r in rgb['rows']}
    crop_by_frame = {r['frame_index']: r for r in crops['sheets']}
    assert len({r['query_id'] for r in held}) == 360
    assert Counter(r['native_slot'] for r in held) == Counter({k: 45 for k in range(8)})
    public_indices = set(public['public_known_indices'])
    assert len(public_indices) == 253
    assert not public_indices.intersection(r['global_index'] for r in held)
    assert all(facts[i]['global_index'] == i and facts[i]['split'] == 'SOURCE_DEVELOPMENT'
               for i in public_indices)
    frame_inventory = []
    for i in frame_ids:
        f, r, c = by_frame[i], rgb_by_frame[i], crop_by_frame[i]
        assert f['split'] == r['split'] == 'SOURCE_VALIDATION'
        assert f['source_id'] == r['source_id'] == c['source_id']
        assert f['canonical_RGB_NPY'] == r['canonical_RGB_NPY']
        assert f['decoded_RGB_sha256'] == r['decoded_RGB_sha256']
        frame_inventory.append({
            'frame_index': i, 'source_id': f['source_id'],
            'RGB96_packet': c['crop_packet'],
            'RGB96_packet_pathstat': pathstat(c['crop_packet']['path']),
            'canonical_RGB': f['canonical_RGB_NPY'],
            'canonical_RGB_pathstat': pathstat(f['canonical_RGB_NPY']['path']),
            'original_RGB': {'path': r['image'], 'sha256': r['image_sha256']},
            'original_RGB_pathstat': pathstat(r['image']),
        })
    rows = []
    for f in held:
        g, i, k = f['global_index'], f['frame_index'], f['native_slot']
        q, fr, r, c = queries[g], by_frame[i], rgb_by_frame[i], crop_by_frame[i]
        assert g == i * 8 + k
        for key in ['query_id', 'frame_index', 'frame_id', 'source_id', 'split', 'native_slot', 'query_xy', 'L']:
            assert f[key] == q[key], (g, key)
        assert f['frame_id'] == fr['id'] == r['id']
        assert f['source_id'].split(':')[0] == 'val/000001'
        assert f['native_continuous_valid'] == q['native_continuous_valid']
        assert f['nearest_valid'] == q['query_nearest_pixel_valid']
        assert f['query_id'] == fr['id'] + ':native' + str(k)
        same_frame = [z for z in held if z['frame_index'] == i]
        assert all(z['context_box_xyxy'] == f['context_box_xyxy'] for z in same_frame)
        rows.append({
            'original_global_index': g, 'original_feature_index': g, 'occurrence_unit': g,
            'query_id': f['query_id'], 'frame_index': i, 'frame_id': f['frame_id'],
            'source_id': f['source_id'], 'split': f['split'], 'native_slot': k,
            'query_xy': f['query_xy'], 'L': f['L'], 'no_detection': q['no_detection'],
            'original_query_available': f['native_continuous_valid'],
            'feature_constructed': f['feature_constructed'],
            'nearest_valid': f['nearest_valid'],
            'local_central4_available': f['local_central4_available'],
            'context_available': f['context_available'], 'Q_inside': f['Q_inside'],
            'support_fractions': f['fractions'],
            'original_context_box_xyxy': f['context_box_xyxy'],
            'context_box_mode': frames['context_box_mode'],
            'raw_hw': fr['raw_hw'], 'canonical_RGB': fr['canonical_RGB_NPY'],
            'decoded_RGB_sha256': fr['decoded_RGB_sha256'],
            'original_RGB': {'path': r['image'], 'sha256': r['image_sha256']},
            'RGB96_packet': c['crop_packet'], 'packet_native_row': k,
            'original_array_hashes_inherited_not_recomputed': {key: f[key] for key in [
                'RGB96_HWC_sha256', 'RGB96_CHW_sha256', 'valid96_sha256',
                'target224_F32_HWC_sha256', 'target_valid_sha256']},
            'new_owner_annotation': None, 'training_admission': False,
        })
    count = lambda key: {str(k).lower(): v for k, v in Counter(r[key] for r in rows).items()}
    assert count('original_query_available') == {'true': 252, 'false': 108}
    assert count('feature_constructed') == {'true': 336, 'false': 24}
    assert count('local_central4_available') == {'true': 248, 'false': 112}
    center_available_invalid = sum(r['original_query_available'] and not r['local_central4_available'] for r in rows)
    assert center_available_invalid == 4
    assert all(not r['local_central4_available'] or r['original_query_available'] for r in rows)
    manifest = OUT / 'ALL360_HELD_INPUT_ONLY.jsonl'
    manifest.write_text(''.join(json.dumps(r, ensure_ascii=False, allow_nan=False) + '\n' for r in rows))
    manifest_b = {'path': str(manifest), 'sha256': sha(manifest.read_bytes()), 'rows': 360}
    reviewed_direct_metadata = [scope, complete, protocol, frames, r0]
    direct_v61 = any('v61' in json.dumps(x).lower() for x in reviewed_direct_metadata)
    assert not direct_v61
    review = {
        'schema': 'V177_METADATA_ONLY_PUBLIC_HELD_OWNER_READINESS_C_v1',
        'completed': True, 'PASS': True, 'UTC': datetime.now(timezone.utc).isoformat(),
        'scope': scope_b, 'code': {'path': str(Path(__file__).resolve()), 'sha256': sha(Path(__file__).read_bytes())},
        'inputs': {'facts': facts_b, 'queries': queries_b, 'feature_complete': complete_b,
                   'feature_protocol': protocol_b, 'frame_inputs': frames_b, 'crop_manifest': crops_b,
                   'RGB_manifest': rgb_b, 'initializer_seal_metadata': r0_b,
                   'producer_text': code_b, 'public253_training_protocol': public_b},
        'output_manifest': manifest_b,
        'denominators': {'rows': 360, 'frames': 45, 'sequence': 'val/000001',
                         'original_global_indices': [584, 943], 'frame_indices': [73, 117],
                         'native_slots_0_through_7_rows_each': 45,
                         'input_available': 252, 'original_input_U': 108,
                         'feature_constructed': 336, 'no_detection_missing_q_L_rows': 24,
                         'no_detection_frames': 3, 'nearest_valid': 251,
                         'center2x2_allvalid': 248, 'center2x2_invalid': 112,
                         'input_available_but_center2x2_invalid': 4,
                         'context_available': 336, 'Q_inside': 336},
        'identity_checks': {'all360_original_q_L_queryID_split_native_equal': True,
                            'global_feature_occurrence_index_equals_frame_times8_plus_input_native_slot': True,
                            'all8_same_original_context_per_frame': True,
                            'weak253_known_indices_all_SOURCE_DEVELOPMENT': True,
                            'held360_intersection_weak253': 0},
        'pathstat_only_assets': {'frame_bindings': frame_inventory,
                                'RGB96_packets_present': 45, 'canonical_RGB_NPY_present': 45,
                                'original_RGB_jpg_present': 45,
                                'array_image_body_headers_or_sha_reads': 0},
        'cache_contract': {
            'producer_lines': {'path': code_b['path'], 'identity': [46, 63], 'crop_and_context': [139, 155],
                               'saved_packet': [165, 167], 'availability_and_row_index': [181, 187]},
            'RGB96_packet_keys': {'rgb96': 'uint8[8,96,96,3] HWC', 'valid96': 'bool[8,96,96]',
                                 'global_index': 'int32[8]', 'crop_constructed': 'bool[8]'},
            'central32': 'rgb96[native_slot,32:64,32:64,:] exact subset',
            'center2x2': 'rgb96[native_slot,47:49,47:49,:]; valid96 same slice; no overlay in raw pixels',
            'RGB96_regeneration_needed': False,
            'pixel_stage_required': 'Verify bound cache SHA, dtype/shape/global indices, inherited per-row RGB96/valid hashes before rendering; this readiness did not open caches.',
            'target224_cached_in_packet': False,
            'context_source': 'Original initializer-selected q8 unclipped/unexpanded minmax, copied from frozen input facts; not a GT selected-instance bbox or pose/reference crop.',
            'future_context_render': 'Use bound original/canonical RGB and saved original_context_box_xyxy with exact V107 sample_target_context; verify inherited target224/valid hashes. Full RGB contextual display is also available. No encoder or new feature extraction required.',
            'pinned_local_crop': dict(local_b, function='crop(rgb,q,L,n)', definition_line=74),
            'pinned_context_crop': dict(context_b, function='sample_target_context(rgb,box,out_hw=(224,224))', definition_line=32),
            'feature_packet_binding_inherited_only_no_body_or_hash_read': complete['features'],
        },
        'target_identity_authority': 'Input packet identifies the frozen initializer proposal by q8/context and original source identity, not a pixel-accurate instance-ownership annotation. Future selected-target ambiguity must remain UNKNOWN; no GT/reference coordinates may be shown.',
        'v61_direct_lineage': {'status': 'NOT_LOCATED_IN_REVIEWED_DIRECT_V153_LINEAGE',
                               'searched_only_in_memory_metadata': ['V177 scope', 'V153 feature complete', 'V153 feature protocol', 'V153 frame inputs', 'V153 initializer seal'],
                               'project_search': False, 'no_global_absence_claim': True,
                               'OLD6144_synthetic_v61_not_substituted_for_MR1840_original_q': True},
        'future_review_constraints': [
            'All360/all45/allnative8 retained, never score-selected; original input U108 stays U.',
            'Center2x2 invalid112 (including4 otherwise input-available) cannot be natural-background evidence; future annotation remains UNKNOWN for invalid, mixed, or ambiguous target identity.',
            'Future model-output-blind assistant-weak T/F/BG/U is evaluation-only and separate from original native D/E, native physical identity, part visibility, and coordinate quality.',
            'Same public validation video was historically exposed through weak native supervision review and repeated source metrics/crop/ranking/threshold diagnostics. New owner annotations can be prediction-blind but are not pristine independent/external validation or human/pixel gold.',
            'Freeze render/annotation protocol before any new pixels; disagreement remains UNKNOWN; no fit/calibration/threshold/checkpoint/architecture/data/arm selection from new held labels.',
        ],
        'this_task_activity': {'source_metadata_and_code_only': True, 'pixel_or_NPZ_header_body_reads': 0,
                               'feature_weight_score_body_reads': 0, 'new_annotations': 0,
                               'model_encoder_initializer_fit_calls': 0, 'private_current_or_V112_access': 0,
                               'project_searches': 0, 'q_L_xy_PnP_changes': 0},
        'ready_for_future_render_protocol_preparation': True,
        'actual_pixels_or_annotations_authorized_by_this_receipt': False,
        'training_admission': False, 'selected_model': None, 'adopted': False, 'fully_solved': False,
    }
    write(OUT / 'REVIEW.json', review)
    print(json.dumps({'completed': True, 'rows': 360, 'frames': 45, 'inputU': 108,
                      'center_invalid': 112, 'output_manifest': manifest_b,
                      'review': {'path': str(OUT / 'REVIEW.json'), 'sha256': sha((OUT / 'REVIEW.json').read_bytes())}}))


if __name__ == '__main__':
    main()
