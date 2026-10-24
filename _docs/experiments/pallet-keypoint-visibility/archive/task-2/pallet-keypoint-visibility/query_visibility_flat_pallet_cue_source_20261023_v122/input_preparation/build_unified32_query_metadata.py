"""Construct approved public source-emulated query geometry from JSON only."""
import ast
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

H = Path(__file__).resolve().parent
V = H.parent
P = V.parent
POLICY = V / 'ROOT_CONCRETE_CORPUS_POLICY_BEFORE_COMPOSITES.json'
TOMIE_FRAMES = P / 'query_visibility_public_flat_pallet_queries_20261023_v104/initializer_only_outputs/SELECTED_FRAMES4_INPUT_ONLY.jsonl'
BOX_CODE = P / 'query_visibility_consistent_query_context_20261023_v119/metadata_preparation/EXACT_CURRENT_INITIAL_CONTEXT_BOX_AST.py'
L_CODE = P.parent / 'external_ray_proxy_head_20261022_v60/head_and_roi.py'
ANCHOR_UV = [[.5, .5], [0., 0.], [1., 0.], [1., 1.], [0., 1.], [.5, 0.], [.5, 1.], [0., .5]]
OFFSETS = [(dx, dy) for dy in (-3, 0, 3) for dx in (-3, 0, 3)]


def bind(path):
    path = Path(path)
    return dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def read_rows(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s]


def checked_rows(binding):
    assert bind(binding['path']) == binding
    return read_rows(binding['path'])


def function(path, name):
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name]
    assert len(nodes) == 1
    def require(condition, message):
        if not condition:
            raise ValueError(message)
    ns = dict(np=np, require=require)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), ns)
    return ns[name]


def write(path, obj):
    with path.open('x') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def main():
    policy = json.loads(POLICY.read_text())
    assert policy['visual_decision']['conditional_visible_cue_AUX_admitted'] is True
    assert not policy['allowed_now']['head_or_encoder_forward_yet'] and not policy['allowed_now']['fit']
    sources = policy['cue_sources']
    loco = checked_rows(sources['loco_input'])
    le = checked_rows(sources['loco_evidence'])
    tomie = checked_rows(sources['tomie_cues'])
    instances = checked_rows(sources['tomie_instances'])
    frames = read_rows(TOMIE_FRAMES)
    assert (len(loco), len(le), len(tomie), len(instances), len(frames)) == (10, 10, 22, 7, 4)
    context_fn = function(BOX_CODE, 'initial_context_box')
    native_fn = function(L_CODE, 'native_grid')
    canonical, evidence, queries = [], [], []
    for li, row in enumerate(loco):
        e = le[li]
        assert row['cue_record_id'] == e['cue_record_id']
        group = f"LOCO_SUBSET_{row['source_subset']}::IMAGE_{row['official_image_id']}"
        canonical.append(dict(cue_index=li, cue_id=row['cue_record_id'], source='LOCO',
                              source_cue_index=li, input_id=row['input_id'], frame_index=row['frame_index'],
                              split=row['split'], raw_split=row['split'], development_group='LOCO_ORIGINAL_FOLD',
                              source_frame_group=group, target_instance_id=f"LOCO_ANN:{row['official_target_annotation_id']}",
                              full_RGB=row['full_RGB'], full_RGB_format='NPY', decoded_RGB_sha256=row['decoded_RGB_sha256'],
                              original_JPEG=row['original_JPEG'], raw_hw=row['raw_hw'], target_bbox_xyxy=row['official_target_bbox_xyxy'],
                              target_bbox_provenance='EXISTING_OFFICIAL_ANNOTATION_TARGET',
                              original_actual_initializer_provenance=row['original_initializer_context_provenance']))
        evidence.append(dict(cue_index=li, cue_id=row['cue_record_id'], source='LOCO', split=row['split'],
                             reference_xy=e['reference_xy_unchanged'],
                             original_regions=[dict(kind='HALF_OPEN_PIXEL_EDGE', bbox=e['uncertainty_bbox_half_open_xyxy_unchanged'])],
                             before_visible_weak_supported=True, root_conditional_AUX_admission=bind(POLICY),
                             original_record_unchanged=e, original_provenance=dict(file=sources['loco_evidence'], row_index=li),
                             original_initializer_association='UNKNOWN', native_mapping='UNKNOWN', human_gold=False,
                             per_query_material_or_exact_corner_gold=False))
    for ti, cue in enumerate(tomie):
        ci = 10 + ti
        ii, instance = next((j, r) for j, r in enumerate(instances) if r['visual_instance_id'] == cue['visual_instance_id'])
        frame = next(r for r in frames if r['frame_index'] == cue['frame_index'])
        assert cue['RGB'] == instance['RGB'] == frame['original_image']
        assert cue['native_hw'] == instance['native_hw'] == frame['raw_hw']
        assert cue['visual_target_bbox_xyxy_native_half_open'] == instance['visual_bbox_xyxy_native_half_open']
        assert cue['weak_state_proposal'] == 'D_VISIBLE_STRUCTURAL_CUE_CONDITIONAL_AI'
        canonical.append(dict(cue_index=ci, cue_id=cue['cue_id'], source='TOMIE', source_cue_index=ti,
                              input_id=f"TOMIE_PUBLIC:{cue['camera']}:{cue['source_frame_name']}", frame_index=cue['frame_index'],
                              split='VALIDATION', raw_split='TOMIE_DEVELOPMENT_STRESS', development_group='TOMIE_DEVELOPMENT_STRESS',
                              source_frame_group=cue['capture_cluster'], target_instance_id=cue['visual_instance_id'],
                              full_RGB=frame['canonical_RGB'], full_RGB_format='NPY', decoded_RGB_sha256=frame['decoded_RGB_sha256'],
                              original_JPEG=cue['RGB'], raw_hw=cue['native_hw'], target_bbox_xyxy=cue['visual_target_bbox_xyxy_native_half_open'],
                              target_bbox_provenance='ROOT_VIEWED_AI_WEAK_VISUAL_INSTANCE_ENVELOPE_NOT_OFFICIAL_ENTITY',
                              original_actual_initializer_provenance=dict(file=bind(TOMIE_FRAMES), row_index=cue['frame_index'])))
        evidence.append(dict(cue_index=ci, cue_id=cue['cue_id'], source='TOMIE', split='VALIDATION',
                             reference_xy=cue['reference_xy_native'],
                             original_regions=[dict(kind='CLOSED_NATIVE_UNCERTAINTY', bbox=cue['uncertainty_bbox_xyxy_closed'])],
                             original_cell_footprint_half_open_xyxy=cue['uncertainty_native_cell_footprint_xyxy_half_open'],
                             before_visible_weak_supported=True, root_conditional_AUX_admission=bind(POLICY),
                             original_record_unchanged=cue, original_visual_instance_unchanged=instance,
                             original_provenance=dict(file=sources['tomie_cues'], row_index=ti),
                             original_instance_provenance=dict(file=sources['tomie_instances'], row_index=ii),
                             original_initializer_association='UNKNOWN', official_entity_association='UNKNOWN',
                             native_mapping='UNKNOWN', human_gold=False, per_query_material_or_exact_corner_gold=False))
    assert len(canonical) == len(evidence) == 32
    for row, e in zip(canonical, evidence):
        ci = row['cue_index']
        bbox = np.asarray(row['target_bbox_xyxy'], np.float64)
        ref = np.asarray(e['reference_xy'], np.float64)
        assert bbox.shape == (4,) and np.isfinite(bbox).all() and (bbox[2:] > bbox[:2]).all()
        assert ref.shape == (2,) and np.isfinite(ref).all()
        generated = []
        for ji, offset in enumerate(OFFSETS):
            q8 = bbox[:2] + np.asarray(ANCHOR_UV) * (bbox[2:] - bbox[:2])
            q8[0] = ref + offset
            q = q8[0]
            context = context_fn(q8)
            _, _, side = native_fn(q8, row['raw_hw'])
            h, w = row['raw_hw']
            continuous = bool(0 <= q[0] < w and 0 <= q[1] < h)
            nearest = bool(0 <= np.floor(q[0] + .5) < w and 0 <= np.floor(q[1] + .5) < h)
            assert context is not None and (q >= context[:2]).all() and (q <= context[2:]).all()
            generated.append(dict(query_index=9 * ci + ji, query_id=f"V122_CUE_{ci:02d}::J{ji}",
                                  cue_index=ci, cue_id=row['cue_id'], jitter_index=ji, offset_xy=list(offset),
                                  source=row['source'], input_id=row['input_id'], split=row['split'], raw_split=row['raw_split'],
                                  development_group=row['development_group'], source_frame_group=row['source_frame_group'],
                                  target_instance_id=row['target_instance_id'], full_RGB=row['full_RGB'], full_RGB_format=row['full_RGB_format'],
                                  decoded_RGB_sha256=row['decoded_RGB_sha256'], raw_hw=row['raw_hw'],
                                  target_bbox_xyxy=row['target_bbox_xyxy'], query_xy=q.tolist(), L=side,
                                  initial_points=q8.tolist(), context_box_xyxy=context.tolist(),
                                  q_native_continuous_valid=continuous, q_nearest_valid=nearest, q_native_valid=continuous and nearest,
                                  source_emulated_not_initializer=True, fixed_carrier_slot=0, carrier_is_native_identity=False,
                                  original_initializer_touched=False, context_Q_inside_geometry=True,
                                  emulation_grid='NEW_SOURCE_EMULATED_QUERY_GRID',
                                  composite_native_recipe_shared_all9=True))
        base = generated[4]
        assert base['offset_xy'] == [0, 0] and base['query_xy'] == e['reference_xy']
        row.update(L0=base['L'], base_q8_at_r=base['initial_points'], base_context_box_xyxy=base['context_box_xyxy'],
                   base_input={k: base[k] for k in ('input_id', 'split', 'raw_split', 'raw_hw', 'query_xy', 'L',
                               'initial_points', 'context_box_xyxy', 'source_emulated_not_initializer', 'fixed_carrier_slot',
                               'carrier_is_native_identity', 'source_frame_group', 'original_initializer_touched')},
                   query_indices=[r['query_index'] for r in generated], anchor_uv8=ANCHOR_UV,
                   source_emulated_not_initializer=True, original_initializer_touched=False,
                   carrier_native_part_identity=False)
        for qrow in generated:
            qrow['L0'] = row['L0']
        queries.extend(generated)
    assert [r['query_index'] for r in queries] == list(range(288))
    assert Counter(r['split'] for r in canonical) == {'TRAIN': 6, 'VALIDATION': 26}
    assert Counter(r['split'] for r in queries) == {'TRAIN': 54, 'VALIDATION': 234}
    assert all(r['q_native_valid'] for r in queries)
    outputs = {}
    for filename, data in (('CUES32_INPUT_ONLY.jsonl', canonical), ('CUES32_WEAK_EVIDENCE_GT_ONLY.jsonl', evidence),
                           ('QUERIES288_INPUT_ONLY.jsonl', queries)):
        path = H / filename
        with path.open('x') as f:
            for r in data:
                f.write(json.dumps(r, ensure_ascii=False, separators=(',', ':'), allow_nan=False) + '\n')
        outputs[filename] = bind(path)
    ls, l0s = [r['L'] for r in queries], [r['L0'] for r in canonical]
    summary = dict(completed=True, metadata_only=True, policy=bind(POLICY), code=bind(Path(__file__)),
                   source_bindings=sources, TOMIE_native_RGB_metadata=bind(TOMIE_FRAMES),
                   context_operator=bind(BOX_CODE), L_operator=bind(L_CODE), outputs=outputs,
                   cues=32, query_conditions=288, source_cues=dict(LOCO=10, TOMIE=22),
                   cue_folds=dict(TRAIN=6, VALIDATION=26), query_folds=dict(TRAIN=54, VALIDATION=234),
                   TOMIE_group='TOMIE_DEVELOPMENT_STRESS, one complete episode/camera/time group; TRAIN0',
                   LOCO_original_cue_folds=dict(TRAIN=6, VALIDATION=4),
                   source_frame_groups=len({r['source_frame_group'] for r in canonical}),
                   unique_native_RGB=len({r['full_RGB']['path'] for r in canonical}),
                   anchors_uv8=ANCHOR_UV, carrier='q0 only; no native-part identity or nearest/GT-best slot',
                   L0_px=dict(min=min(l0s), max=max(l0s)), per_query_L_px=dict(min=min(ls), max=max(ls)),
                   queries_in_native_continuous_and_nearest=288, Q_inside_context=288,
                   native_composition_attempts=256,
                   composition_fixed_L0_across_all9_jitters=True,
                   planned_occurrence_order='17*query_index BEFORE; +1+2*donor_rank COVER; +2+2*donor_rank SHAM, rank0..7',
                   planned_rows=dict(BEFORE=288, COVER=2304, SHAM=2304, total=4896),
                   original_reference_uncertainty_folds_and_initializer_unchanged=True,
                   weak_evidence='Root-viewed conditional AUX component/cue visibility; not per-query material owner, human/native gold or independent derivatives.',
                   actual_RGB_masks_features_checkpoint_model_fit_current_V112_PnP_reads=0)
    write(H / 'UNIFIED32_QUERY_METADATA_SUMMARY.json', summary)
    print(json.dumps(dict(summary=bind(H / 'UNIFIED32_QUERY_METADATA_SUMMARY.json'), outputs=outputs,
                         L0_range=summary['L0_px'], L_range=summary['per_query_L_px'], query_folds=summary['query_folds']), indent=2))


if __name__ == '__main__':
    main()
