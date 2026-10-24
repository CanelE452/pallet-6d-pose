"""Build only public input/evidence JSON metadata; never opens image/model payloads."""
import ast
import hashlib
import json
from pathlib import Path

import numpy as np

H = Path(__file__).resolve().parent
V = H.parent
P = V.parent
POLICY = V / 'ROOT_PUBLIC_BASELINE_POLICY_BEFORE_FORWARD.json'
L_CODE = P.parent / 'external_ray_proxy_head_20261022_v60/head_and_roi.py'
BOX_CODE = P / 'query_visibility_consistent_query_context_20261023_v119/metadata_preparation/EXACT_CURRENT_INITIAL_CONTEXT_BOX_AST.py'


def bind(path):
    path = Path(path)
    return dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def write(path, value):
    with path.open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def functions(path, name):
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name]
    assert len(nodes) == 1
    def require(condition, message):
        if not condition:
            raise ValueError(message)
    ns = dict(np=np, require=require)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), ns)
    return ns[name]


def main():
    policy = json.loads(POLICY.read_text())
    source = Path(policy['source_pairs']['path'])
    assert bind(source) == policy['source_pairs']
    pairs = [json.loads(s) for s in source.read_text().splitlines()]
    assert len(pairs) == 37
    assert [r['before_cue_view_index'] for r in pairs] == sorted(r['before_cue_view_index'] for r in pairs)
    context_fn = functions(BOX_CODE, 'initial_context_box')
    native_fn = functions(L_CODE, 'native_grid')
    inputs, evidence = [], []
    offsets = [(dx, dy) for dy in (-3, 0, 3) for dx in (-3, 0, 3)]
    same_original_uncertainty9 = 0
    for pi, pair in enumerate(pairs):
        assert pair['pair_index'] == pi
        r = np.asarray(pair['reference_xy_unchanged'], np.float64)
        original_before = np.asarray(pair['before']['actual_initializer']['initial_points'], np.float64)
        assert original_before.shape == (8, 2)
        grid = r + np.asarray(offsets, np.float64)
        same_original_uncertainty9 += int(np.array_equal(grid, pair['uncertainty9_xy_GT_only_unchanged']))
        for ji, offset in enumerate(offsets):
            q8 = original_before.copy()
            q8[0] = grid[ji]
            context = context_fn(q8)
            assert context is not None
            hw = pair['before']['raw_hw']
            assert hw == pair['after']['raw_hw']
            _, _, side = native_fn(q8, hw)
            q = q8[0]
            assert (q >= context[:2]).all() and (q <= context[2:]).all()
            for view in ('before', 'after'):
                original = pair[view]
                h, w = original['raw_hw']
                continuous = bool(0 <= q[0] < w and 0 <= q[1] < h)
                nearest = bool(0 <= np.floor(q[0] + .5) < w and 0 <= np.floor(q[1] + .5) < h)
                centre_domain = bool(-.5 <= q[0] < w - .5 and -.5 <= q[1] < h - .5)
                # Exact V90 crop validity geometry, without opening or creating RGB.
                axis = (np.arange(96, dtype=np.float64) + .5) / 96 - .5
                x, y = np.meshgrid(q[0] + axis * side, q[1] + axis * side)
                valid = (x >= -.5) & (x < w - .5) & (y >= -.5) & (y < h - .5)
                central4 = bool(valid[47:49, 47:49].all())
                # Exact V107 full-positive-neighbor support at native context224 grid.
                x0, y0, x1, y1 = context
                gx = x0 + (np.arange(224) + .5) * (x1 - x0) / 224
                gy = y0 + (np.arange(224) + .5) * (y1 - y0) / 224
                support_x = (gx >= 0) & (gx <= w - 1)
                support_y = (gy >= 0) & (gy <= h - 1)
                context_fraction = float(np.float32(np.outer(support_y, support_x).mean()))
                native_valid = continuous and nearest
                factual = native_valid and central4
                oi = len(inputs)
                qid = f'V121_PAIR_{pi:03d}::J{ji}::{view}'
                provenance = dict(file=policy['source_pairs'], row_index=pi,
                                  json_pointer=f'/{view}/actual_initializer')
                inputs.append(dict(
                    occurrence_index=oi, query_id=qid, pair_index=pi, pair_id=pair['pair_id'],
                    jitter_index=ji, offset_xy=list(offset), view=view,
                    input_id=original['frame_input_id'], source='PALLOX',
                    raw_split=pair['split'], split='TRAIN' if pair['split'] == 'TRAIN_LAB' else 'VALIDATION',
                    full_RGB=original['rawRGB'], full_RGB_format='PNG',
                    decoded_RGB_sha256=original['decoded_RGB_sha256'], raw_hw=hw,
                    query_xy=q.tolist(), L=side, initial_points=q8.tolist(),
                    context_box_xyxy=context.tolist(),
                    source_emulated_not_initializer=True, emulation_grid='NEW_SOURCE_EMULATED_QUERY_GRID',
                    fixed_carrier_slot=0, carrier_slot_is_native_part_identity=False,
                    original_query_continuous_valid=continuous, query_nearest_pixel_valid=nearest,
                    query_rgb96_centre_domain_valid=centre_domain, RGB_central4_valid_geometry=central4,
                    q_native_valid=native_valid, local_central4_available=central4,
                    factual_query_available=factual, context_available_geometry=context_fraction > 0,
                    target_valid_fraction_geometry=context_fraction,
                    local_valid_fraction_geometry=float(np.float32(valid.mean())),
                    central_valid_fraction_geometry=float(np.float32(valid[32:64, 32:64].mean())),
                    query_available_geometry=factual and context_fraction > 0,
                    context_Q_inside_geometry=True,
                    same_BEFORE_tuple_q_L_context_used_for_both_views=True,
                    source_actual_metadata=provenance,
                    native_semantic_mapping='UNKNOWN', human_gold=False,
                ))
                evidence.append(dict(
                    occurrence_index=oi, query_id=qid, pair_index=pi, cue_id=pair['cue_id'],
                    cue_index=pair['cue_index'], pair_id=pair['pair_id'], jitter_index=ji,
                    view=view, raw_split=pair['split'],
                    original_teacher_label_unchanged=original['weak_teacher_label_unchanged'],
                    original_label_meaning_unchanged=original['weak_teacher_label_meaning_unchanged'],
                    original_label_reason_unchanged=original['weak_teacher_label_reason_unchanged'],
                    original_reference_xy_unchanged=pair['reference_xy_unchanged'],
                    original_uncertainty_bbox_xyxy_half_open_unchanged=pair['uncertainty_bbox_xyxy_half_open_unchanged'],
                    original_uncertainty9_xy_unchanged=pair['uncertainty9_xy_GT_only_unchanged'],
                    original_teacher_provenance=original['teacher_record_provenance'],
                    original_source_cue_provenance=pair['source_cue_provenance'],
                    registration_evidence_unchanged=pair['registration_evidence_unchanged'],
                    source_pair_record=dict(file=policy['source_pairs'], row_index=pi),
                    duplicate_group_link_only=pair['duplicate_group_link_only'],
                    duplicate_group_size=pair['duplicate_group_size'], duplicate_weight=pair['duplicate_weight'],
                    native_semantic_mapping='UNKNOWN', human_gold=False, official_gold=False,
                    training_admission=False, new_labels_created=False,
                    label_scope='Existing weak expected structural-cue-region visibility, not material membership at every q or an exact/native corner.',
                ))
            before, after = inputs[-2:]
            assert before['view'] == 'before' and after['view'] == 'after'
            for key in ('query_xy', 'initial_points', 'L', 'context_box_xyxy', 'raw_hw'):
                assert before[key] == after[key]
    assert len(inputs) == len(evidence) == 666
    assert [r['occurrence_index'] for r in inputs] == list(range(666))
    assert sum(r['split'] == 'TRAIN' for r in inputs) == 270
    assert sum(r['split'] == 'VALIDATION' for r in inputs) == 396
    assert all(r['query_available_geometry'] for r in inputs)
    H.mkdir(exist_ok=True)
    ip = H / 'ALL666_QUERY_EMULATION_INPUT_ONLY.jsonl'
    ep = H / 'ALL666_ORIGINAL_WEAK_EVIDENCE_GT_ONLY.jsonl'
    for path, records in ((ip, inputs), (ep, evidence)):
        with path.open('x') as f:
            for row in records:
                f.write(json.dumps(row, ensure_ascii=False, separators=(',', ':'), allow_nan=False) + '\n')
    sides = np.asarray([r['L'] for r in inputs])
    summary = dict(
        completed=True, metadata_only=True, root_policy=bind(POLICY), code=bind(__file__),
        input_rows=bind(ip), weak_original_evidence=bind(ep), source_pairs=policy['source_pairs'],
        context_operator=bind(BOX_CODE), L_operator=bind(L_CODE),
        rows=666, cue_records=37, physical_frame_pairs=22, unique_native_RGB=44,
        order='Original37 pair order, jitter dy then dx in(-3,0,3), view BEFORE then AFTER',
        folds=dict(TRAIN_LAB=270, VAL_HALL=396),
        L_px=dict(min=float(sides.min()), median=float(np.median(sides)), max=float(sides.max())),
        all333_paired_q_L_context_identical=True, original_actual_q8_L_metadata_unchanged=True,
        emulated_q_in_native_continuous_domain=sum(r['original_query_continuous_valid'] for r in inputs),
        emulated_q_in_nearest_domain=sum(r['query_nearest_pixel_valid'] for r in inputs),
        factual_query_available_geometry=sum(r['factual_query_available'] for r in inputs),
        context_available_geometry=sum(r['context_available_geometry'] for r in inputs),
        combined_query_available_geometry=sum(r['query_available_geometry'] for r in inputs),
        Q_inside_context=666, U_rows_retained=0,
        new_grid_name='NEW_SOURCE_EMULATED_QUERY_GRID', original_uncertainty9_exact_match_pairs=same_original_uncertainty9,
        not_independent_samples='666 correlated views;37 weak cue records/22 physical pairs from one bin, sourceVAL development-exposed.',
        source_only_baseline='One frozen C14 diagnostic; no new labels, fit, threshold or model selection.',
        actual_RGB_feature_checkpoint_model_current_V112_fit_PnP_reads=0,
    )
    write(H / 'SUMMARY.json', summary)
    print(json.dumps(dict(summary=bind(H / 'SUMMARY.json'), input_rows=bind(ip), weak_evidence=bind(ep), rows=666), indent=2))


if __name__ == '__main__':
    main()
