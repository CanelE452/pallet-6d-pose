"""V91 GT-only physical cue/view/tile join. No image or feature reader.

Keep every V89 input tile and every original cue ID. Weak structural cue-region
visibility is NOT target-material pixel membership or native8 semantic gold.
"""
from collections import Counter, defaultdict
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
import numpy as np

HERE = Path(__file__).resolve().parent
CONTRACT = HERE / 'TEACHER_CONTRACT_BEFORE_BUILD.json'
D_STATE = 'APPARENT_SAME_CUE_VISIBLE_AI'
E_STATE = 'EXTERNAL_FOREGROUND_BLOCKER_COVERS_CUE_REGION_AI'


def bind(path):
    p = Path(path); raw = p.read_bytes()
    return dict(path=str(p.resolve()), sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))


def read(path): return json.loads(Path(path).read_text())
def lines(path): return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def verify(b):
    assert bind(b['path']) == b, b['path']
    return Path(b['path'])


def save(path, data):
    with Path(path).open('x') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False); f.write('\n')


def dump(path, rows):
    with Path(path).open('x') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(',', ':'), allow_nan=False) + '\n')


def coverage(box, window):
    if box is None: return 'MISSING_REFERENCE_U'
    if window[0] <= box[0] and window[1] <= box[1] and box[2] <= window[2] and box[3] <= window[3]:
        return 'FULL'
    return 'PARTIAL' if max(box[0], window[0]) < min(box[2], window[2]) and max(box[1], window[1]) < min(box[3], window[3]) else 'MISS'


def unc9(box):
    """Equal-weight 3x3 cell centres of the unchanged half-open uncertainty box."""
    if box is None: return None
    x0, y0, x1, y1 = box
    return np.asarray([[x0 + (j + .5) * (x1 - x0) / 3,
                        y0 + (i + .5) * (y1 - y0) / 3]
                       for i in range(3) for j in range(3)], np.float64)


def norm_xy(points, query, side):
    return (2 * (np.asarray(points, np.float64) - np.asarray(query, np.float64)) / float(side)).astype(np.float32)


def support_coefficients(normalized9):
    """Exact mathematical bilinear coefficients for F32 grid, align_corners=False.

    Border clipping is on continuous map coordinates BEFORE neighbours. The
    region statistic is mean of nine interpolated logits, not nine BCE labels.
    """
    coordinates = (np.asarray(normalized9, np.float32).astype(np.float64) + 1) * 9 / 2 - .5
    coordinates = np.clip(coordinates, 0, 8)
    coefficients = defaultdict(float)
    for x, y in coordinates:
        x0, y0 = int(math.floor(x)), int(math.floor(y))
        x1, y1 = min(x0 + 1, 8), min(y0 + 1, 8)
        dx, dy = x - x0, y - y0
        for xx, yy, w in ((x0, y0, (1-dx)*(1-dy)), (x1, y0, dx*(1-dy)),
                          (x0, y1, (1-dx)*dy), (x1, y1, dx*dy)):
            if w > 0: coefficients[(yy, xx)] += w / 9
    assert abs(sum(coefficients.values()) - 1) < 1e-12
    return dict(coefficients)


def decode_label(cue, view):
    """Return label0D/1E/-1U, reason; never repair an unsupported named cue."""
    if cue['reference_xy'] is None:
        assert cue['uncertainty_bbox_xyxy_half_open'] is None
        return -1, 'MISSING_OR_UNRESOLVED_BEFORE_REFERENCE_U'
    if cue['corpus'] == 'V85_CANONICAL294':
        source = cue['source_record_unchanged']
        if source['proposal_rule_reason'] == 'SECONDARY_BEFORE_PART_IDENTITY_NOT_SUPPORTED_TRUE':
            return -1, 'SECONDARY_BEFORE_PART_IDENTITY_UNSUPPORTED_BOTH_VIEWS_U'
        if view == 'before':
            assert source['primary_record_unchanged']['before_state'] == 'APPARENT_VISIBLE_STRUCTURAL_CUE_AI'
            return 0, 'ORIGINAL_BEFORE_OBSERVED_STRUCTURAL_CUE_CONDITIONAL_AI'
        observed = source['proposed_aux_after_state']
    else:
        assert cue['corpus'] == 'V84_SEED8_SEPARATE_6LINKED_REGIONS'
        if view == 'before': return 0, 'ORIGINAL_SEED_BEFORE_OBSERVED_CUE_CONDITIONAL_AI'
        values = list(cue['after_state'].values())
        if len(values) != 2 or len(set(values)) != 1:
            return -1, 'SEED_ORIGINAL_AFTER_REVIEW_DISAGREEMENT_U'
        observed = values[0]
    if observed == D_STATE: return 0, 'APPARENT_SAME_CUE_VISIBLE_CONDITIONAL_AI'
    if observed == E_STATE: return 1, 'FOREIGN_OPAQUE_BLOCKER_ENTIRE_UNCERTAINTY_CONDITIONAL_AI'
    return -1, 'ORIGINAL_AFTER_UNCLEAR_OR_MOVEMENT_U'


def generated(path):
    assert coverage([0, 0, 2, 2], [0, 0, 2, 2]) == 'FULL'
    assert coverage([0, 0, 2, 2], [2, 0, 4, 2]) == 'MISS'
    assert coverage([0, 0, 2, 2], [1, 0, 3, 2]) == 'PARTIAL'
    assert coverage(None, [0, 0, 2, 2]) == 'MISSING_REFERENCE_U'
    u = unc9([0, 0, 3, 6])
    assert np.array_equal(u, [[.5, 1], [1.5, 1], [2.5, 1], [.5, 3], [1.5, 3], [2.5, 3], [.5, 5], [1.5, 5], [2.5, 5]])
    assert (u[:, 0] < 3).all() and (u[:, 1] < 6).all()
    # Exact grid feature centres q+(j-4)*L/9 must sample their own map cell.
    for y in range(9):
        for x in range(9):
            point = np.array([[(x-4)/9, (y-4)/9]])
            n = norm_xy(np.repeat(point, 9, axis=0), [0, 0], 1)
            co = support_coefficients(n)
            assert abs(co.get((y, x), 0) - 1) < 3e-7
    # Border support and all-nine mean preserve a constant map, even out of map.
    for n in (np.zeros((9, 2)), np.full((9, 2), 9.), np.full((9, 2), -9.)):
        co = support_coefficients(n); assert abs(sum(co.values()) - 1) < 1e-12
    # Region BCE(mean(logits)) differs from imposing nine independent labels.
    z = np.array([-2.] * 4 + [0.] + [2.] * 4)
    bce = lambda x: np.logaddexp(0, x)
    assert bce(z.mean()) < bce(z).mean()
    # One cue/view has .5 per scale, then duplicates separately downweight.
    w = np.array([.5/3]*3 + [.5/2]*2)
    assert np.isclose(w.sum(), 1) and np.isclose((w*.5).sum(), .5)
    unsupported = dict(corpus='V85_CANONICAL294', reference_xy=[1, 1], source_record_unchanged=dict(
        proposal_rule_reason='SECONDARY_BEFORE_PART_IDENTITY_NOT_SUPPORTED_TRUE'))
    assert decode_label(unsupported, 'before')[0] == decode_label(unsupported, 'after')[0] == -1
    missing = dict(reference_xy=None, uncertainty_bbox_xyxy_half_open=None)
    assert decode_label(missing, 'before')[0] == decode_label(missing, 'after')[0] == -1
    save(path, dict(PASS=True, builder=bind(__file__), generated_only=True,
                    halfopen_FULL_PARTIAL_MISS=True, unchanged_unc9_cellcentres=True,
                    align_corners_false_feature_centres=True, border_coefficients=True,
                    one_region_BCE_not_9_pixel_labels=True, per_scale_and_duplicate_mass=True,
                    missing_and_secondaryunsupported_both_view_U=True,
                    image_feature_model_current_PnP_reads=0))


def main():
    contract = read(CONTRACT); lock = read(HERE / 'BUILD_ENTRYLOCK_BEFORE_GENERATION.json')
    assert lock['builder'] == bind(__file__) and lock['contract'] == bind(CONTRACT)
    verify(lock['generated']); assert read(lock['generated']['path'])['PASS'] is True
    for item in contract['bindings'].values(): verify(item)
    b = contract['bindings']
    cues = lines(b['cues302']['path']); coverage604 = lines(b['coverage604']['path'])
    tiles = lines(b['grid12900']['path']); frames = lines(b['frames100']['path'])
    union = read(b['seed_union']['path'])
    assert len(cues) == 302 and len(coverage604) == 604 and len(tiles) == 12900 and len(frames) == 100
    assert len({q['cue_id'] for q in cues}) == 302
    assert sum(c['corpus'] == 'V85_CANONICAL294' for c in cues) == 294
    assert sum(c['reference_xy'] is None for c in cues) == 151
    assert union['possible_duplicate_groups_before_only'] == contract['seed_duplicate_original_groups']
    fmap = {(f['pair_id'], f['view']): f for f in frames}
    tidx = {t['tile_id']: i for i, t in enumerate(tiles)}
    assert len(tidx) == 12900
    byframe = defaultdict(list)
    for i, tile in enumerate(tiles): byframe[tile['frame_input_id']].append((i, tile))
    assert len(byframe) == 100 and all(len(x) == 129 for x in byframe.values())
    covmap = {(r['cue_id'], r['view']): r for r in coverage604}; assert len(covmap) == 604
    duplicate = {}
    for group in contract['seed_duplicate_original_groups']:
        for key in group:
            assert key not in duplicate
            duplicate[key] = dict(group='SEED_LINK:' + '|'.join(group), size=len(group))
    cueviews = []; units = []; scalar_csv_rows = []
    cells = defaultdict(lambda: {0: defaultdict(float), 1: defaultdict(float)})
    operator_groups = defaultdict(lambda: {0: [], 1: []})
    for ci, cue in enumerate(cues):
        assert cue['native8_semantic_id'] is None
        ref = cue['reference_xy']; box = cue['uncertainty_bbox_xyxy_half_open']
        assert (ref is None) == (box is None)
        seed = cue['corpus'].startswith('V84')
        key = cue['slot_or_proposal_id']
        dg = duplicate.get(key, dict(group='SEED_SINGLETON:' + key, size=1)) if seed else dict(group=cue['cue_id'], size=1)
        dw = 1 / dg['size']
        if not seed:
            original = cue['source_record_unchanged']['primary_record_unchanged']
            assert ref == original['reference_xy'] and box == original['uncertainty_bbox_xyxy_half_open']
        points = unc9(box)
        for view in ('before', 'after'):
            vi = len(cueviews); frame = fmap[(cue['pair_id'], view)]; h, w = frame['raw_hw']
            label, reason = decode_label(cue, view)
            inside = box is not None and 0 <= box[0] < box[2] <= w and 0 <= box[1] < box[3] <= h
            if ref is not None:
                assert len(ref) == 2 and all(math.isfinite(v) for v in ref)
                assert box[0] <= ref[0] < box[2] and box[1] <= ref[1] < box[3]
            observed = covmap[(cue['cue_id'], view)]
            assert observed['frame_input_id'] == frame['frame_input_id']
            assert observed['reference_xy_unchanged'] == ref and observed['uncertainty_bbox_unchanged'] == box
            full = {128: [], 256: []}
            observed_cov = {}
            for i, tile in byframe[frame['frame_input_id']]:
                assert tile['raw_hw'] == frame['raw_hw'] and tile['image_sha256'] == frame['image_sha256']
                cv = coverage(box, tile['window_xyxy_half_open']); observed_cov[tile['tile_id']] = cv
                if cv == 'FULL': full[tile['roi_side']].append(i)
            assert sorted(k for k, cv in observed_cov.items() if cv == 'FULL') == sorted(observed['full_tile_ids'])
            if label in (0, 1): assert inside and all(len(full[s]) > 0 for s in (128, 256))
            cvrow = dict(cue_view_index=vi, cue_index=ci, cue_id=cue['cue_id'], pair_id=cue['pair_id'],
                corpus=cue['corpus'], slot_or_proposal_id=key, view=view, split=frame['split'],
                frame_input_id=frame['frame_input_id'], image_sha256=frame['image_sha256'], raw_hw=frame['raw_hw'],
                teacher_label=label, label_meaning={-1:'UNKNOWN_U', 0:'OBSERVED_CUE_VISIBLE_CONDITIONAL_AI', 1:'FOREIGN_BLOCKER_COVERS_CUE_REGION_CONDITIONAL_AI'}[label],
                label_reason=reason, source_observation_unchanged=cue['before_state'] if view == 'before' else cue['after_state'],
                reference_xy_unchanged=ref, uncertainty_bbox_xyxy_half_open_unchanged=box,
                uncertainty9_xy_GT_only=None if points is None else points.tolist(), uncertainty_inside_original_image=inside,
                duplicate_group_link_only=dg['group'], duplicate_group_size=dg['size'], duplicate_weight=dw,
                full_tiles_per_scale={str(s): len(full[s]) for s in (128, 256)},
                native8_semantic_id=None, human_gold=False, pixel_material_membership=False,
                GT_forward_allowed=False, primary_source_label_loss_admitted=label in (0, 1) and inside)
            cueviews.append(cvrow)
            for ti, tile in byframe[frame['frame_input_id']]:
                cv = observed_cov[tile['tile_id']]; full_tile = cv == 'FULL'
                known = label in (0, 1); admitted = known and inside and full_tile
                geometry_weight = .5 / len(full[tile['roi_side']]) if full_tile and inside else 0.
                base_loss_weight = geometry_weight if known else 0.
                loss_weight = base_loss_weight * dw
                nr = None if ref is None else norm_xy(ref, tile['query_xy'], tile['roi_side'])
                n9 = None if points is None else norm_xy(points, tile['query_xy'], tile['roi_side'])
                row = dict(tile_index=ti, cue_view_index=vi, teacher_label=label,
                    training_label=label if admitted else -1, coverage=cv, loss_admitted=admitted,
                    full=full_tile, uncertainty_inside_image=inside, geometry_tile_weight=geometry_weight,
                    base_loss_weight=base_loss_weight, duplicate_weight=dw, loss_weight=loss_weight,
                    reference_normalized_xy=nr, uncertainty9_normalized_xy=n9)
                units.append(row)
                scalar_csv_rows.append([len(units)-1, ti, tile['tile_id'], vi, cue['cue_id'], view,
                    frame['split'], tile['roi_side'], label, row['training_label'], cv,
                    int(inside), int(admitted), geometry_weight, base_loss_weight, dw, loss_weight])
                if admitted:
                    coeff = support_coefficients(n9)
                    signature = tuple((yx, float(v).hex()) for yx, v in sorted(coeff.items()))
                    operator_groups[(ti, signature)][label].append(cue['cue_id'])
                    for (yy, xx), coeff_value in coeff.items():
                        cells[(ti, yy, xx)][label][cue['cue_id']] += coeff_value * loss_weight
    assert len(cueviews) == 604 and len(units) == 77916
    # Check the original complete tuple order, not a new GT-filtered unit list.
    with open(b['coverage77916']['path'], newline='') as f:
        old = list(csv.DictReader(f))
    assert len(old) == len(units)
    for row, legacy in zip(scalar_csv_rows, old):
        assert row[2] == legacy['tile_id'] and row[4] == legacy['cue_id'] and row[5] == legacy['view']
        assert row[10] == legacy['continuous_coverage']
    unit_to_view = np.asarray([u['cue_view_index'] for u in units], np.int32)
    base_weights = np.asarray([u['base_loss_weight'] for u in units], np.float64)
    weights = np.asarray([u['loss_weight'] for u in units], np.float64)
    labels = np.asarray([v['teacher_label'] for v in cueviews], np.int8)
    per_view_mass = np.bincount(unit_to_view, weights=weights, minlength=604)
    per_view_base = np.bincount(unit_to_view, weights=base_weights, minlength=604)
    assert np.allclose(per_view_base, (labels != -1).astype(float), atol=1e-14, rtol=0)
    assert np.allclose(per_view_mass, [(v['duplicate_weight'] if v['teacher_label'] != -1 else 0.) for v in cueviews], atol=1e-14, rtol=0)
    for s in (128, 256):
        subset = np.asarray([tiles[u['tile_index']]['roi_side'] == s for u in units])
        mass = np.bincount(unit_to_view[subset], weights=base_weights[subset], minlength=604)
        assert np.allclose(mass, (labels != -1)*.5, atol=1e-14, rtol=0)
    # No scatter/overwrite: conflicting sparse support contributions stay separate.
    collisions = []
    for (ti, yy, xx), mapping in sorted(cells.items()):
        if mapping[0] and mapping[1]:
            collisions.append(dict(tile_index=ti, tile_id=tiles[ti]['tile_id'], map_y=yy, map_x=xx,
                D_cue_ids=sorted(mapping[0]), E_cue_ids=sorted(mapping[1]),
                D_effective_weighted_coefficient=sum(mapping[0].values()),
                E_effective_weighted_coefficient=sum(mapping[1].values())))
    exact_operators = [dict(tile_index=key[0], D_cue_ids=values[0], E_cue_ids=values[1])
                       for key, values in operator_groups.items() if values[0] and values[1]]
    out = HERE / 'corpus_build_v1'; out.mkdir(exist_ok=False)
    dump(out / 'ALL302_CUES_SOURCE_IDS_GT_ONLY.jsonl', cues)
    dump(out / 'ALL604_CUE_VIEW_TEACHERS_GT_ONLY.jsonl', cueviews)
    with (out / 'ALL77916_TILE_CUE_TRAINING_UNITS_GT_ONLY.csv').open('x', newline='') as f:
        writer = csv.writer(f); writer.writerow(['unit_index', 'tile_index', 'tile_id', 'cue_view_index', 'cue_id', 'view',
            'split', 'L', 'teacher_label', 'training_label', 'coverage', 'unc_inside_image', 'loss_admitted',
            'geometry_tile_weight', 'base_loss_weight', 'duplicate_weight', 'loss_weight']); writer.writerows(scalar_csv_rows)
    arrays = {}
    for k in ('tile_index', 'cue_view_index'): arrays[k] = np.asarray([u[k] for u in units], np.int32)
    for k in ('teacher_label', 'training_label'): arrays[k] = np.asarray([u[k] for u in units], np.int8)
    for k in ('loss_admitted', 'full', 'uncertainty_inside_image'): arrays[k] = np.asarray([u[k] for u in units], bool)
    for k in ('geometry_tile_weight', 'base_loss_weight', 'duplicate_weight', 'loss_weight'): arrays[k] = np.asarray([u[k] for u in units], np.float64)
    for k, shape in (('reference_normalized_xy', (2,)), ('uncertainty9_normalized_xy', (9, 2))):
        arrays[k] = np.asarray([np.full(shape, np.nan, np.float32) if u[k] is None else u[k] for u in units], np.float32)
        arrays[k + '_safe'] = np.nan_to_num(arrays[k], nan=0., posinf=0., neginf=0.)
    arrays['coverage_code'] = np.asarray([{'FULL':0, 'PARTIAL':1, 'MISS':2, 'MISSING_REFERENCE_U':3}[u['coverage']] for u in units], np.int8)
    np.savez_compressed(out / 'ALL77916_TILE_CUE_TRAINING_UNITS_GT_ONLY.npz', **arrays)
    dump(out / 'D_E_BILINEAR_CELL_SUPPORT_COLLISIONS_GT_ONLY.jsonl', collisions)
    dump(out / 'EXACT_REGION_OPERATOR_D_E_COLLISIONS_GT_ONLY.jsonl', exact_operators)
    strata = []
    for split in ('TRAIN_LAB', 'VAL_HALL'):
        for view in ('before', 'after'):
            vv = [v for v in cueviews if v['split'] == split and v['view'] == view]
            strata.append(dict(split=split, view=view, raw_label_counts=dict(Counter(str(v['teacher_label']) for v in vv)),
                duplicate_weighted_label_counts={str(c): sum(v['duplicate_weight'] for v in vv if v['teacher_label'] == c) for c in (-1, 0, 1)}))
    summary = dict(completed=True, source_cues=302, original_IDs_preserved=302, cue_views=604,
        canonical_cues=294, seed_original_cues=8, seed_linked_distinct_regions=6,
        input_tiles=12900, all_cue_view_tile_combinations=77916,
        raw_view_label_counts=dict(Counter(str(v['teacher_label']) for v in cueviews)),
        duplicate_weighted_view_label_counts={str(c): sum(v['duplicate_weight'] for v in cueviews if v['teacher_label'] == c) for c in (-1, 0, 1)},
        label_counts_by_split_view=strata, view_reason_counts=dict(Counter(v['label_reason'] for v in cueviews)),
        loss_admitted_training_units=int(np.count_nonzero(arrays['loss_admitted'])),
        full_geometry_units=int(np.count_nonzero(arrays['full'])),
        training_label_counts=dict(Counter(str(x) for x in arrays['training_label'].tolist())),
        unweighted_known_cue_view_loss_mass=float(base_weights.sum()), effective_known_cue_view_loss_mass=float(weights.sum()),
        uncertainty_outside_image_defined_views=sum(v['reference_xy_unchanged'] is not None and not v['uncertainty_inside_original_image'] for v in cueviews),
        unsupported_secondary_cue_ids=sorted({v['cue_id'] for v in cueviews if v['label_reason'] == 'SECONDARY_BEFORE_PART_IDENTITY_UNSUPPORTED_BOTH_VIEWS_U'}),
        bilinear_9x9_cells_with_both_D_E_contributions=len(collisions),
        distinct_tiles_with_D_E_cell_overlap=len({v['tile_index'] for v in collisions}),
        exact_region_operator_D_E_collisions=len(exact_operators),
        collision_handling='All sparse cue contributions kept separately; no last-writer overwrite, dense raster label, teacher repair or cue relocation.',
        source_split='21 nonseed canonical TRAIN pairs +1 seed TRAIN pair;28 VAL pairs; same physical bin/environment-only holdout',
        native8_human_semantic_gold=False, target_instance_identity_input=False,
        semantics='Conditional AI observed physical structural cue-region visibility; no pixel-material membership claim.',
        checks=dict(all_input_tiles_preserved=True, all_original_ID_coordinates_uncertainty_preserved=True,
            all_original_combinations_order_exact=True, null151_both_views_U=True,
            secondaryunsupported2_both_views_U=True, input_metadata_unchanged=bind(b['grid12900']['path']) == b['grid12900'],
            known_view_per_scale_base_weight_half=True, known_view_base_weight_sum_one=True,
            duplicate_effective_mass_preserved=True, unlabelled_background_loss_zero=True),
        schema={k: dict(shape=list(v.shape), dtype=str(v.dtype)) for k, v in arrays.items()},
        contract=bind(CONTRACT), builder=bind(__file__), entry=bind(HERE / 'BUILD_ENTRYLOCK_BEFORE_GENERATION.json'),
        image_mask_depth_feature_weight_current_reads=0, model_forward_fit_PnP=0)
    assert summary['raw_view_label_counts'] == {'0':261, '-1':306, '1':37}
    assert summary['duplicate_weighted_view_label_counts'] == {'-1':306., '0':257., '1':37.}
    assert abs(summary['effective_known_cue_view_loss_mass'] - 294) < 1e-10
    assert all(summary['checks'].values())
    save(out / 'GT_CORPUS_SUMMARY_AND_SCHEMA.json', summary)
    files = [bind(p) for p in sorted(out.iterdir()) if p.is_file()]
    save(out / 'GT_CORPUS_SEAL.json', dict(completed=True, PASS=True, files=files,
         contract=bind(CONTRACT), builder=bind(__file__), source_bindings=contract['bindings'],
         no_actual_input_filter_or_forward=True, source_teacher_only=True, current_admission=False))
    print(json.dumps(dict(summary=bind(out / 'GT_CORPUS_SUMMARY_AND_SCHEMA.json'), seal=bind(out / 'GT_CORPUS_SEAL.json'),
        raw_counts=summary['raw_view_label_counts'], weighted_counts=summary['duplicate_weighted_view_label_counts'],
        collision_cells=len(collisions), exact_operator_collisions=len(exact_operators)), ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('stage', choices=['generated', 'build'])
    args = parser.parse_args()
    if args.stage == 'generated': generated(HERE / 'GENERATED_TEACHER_GEOMETRY_TESTS.json')
    else: main()
