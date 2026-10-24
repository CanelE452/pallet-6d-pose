"""Pure sampled-domain finite end-locality arithmetic (V189). No asset/image/model imports or loaders.

Input metadata is a prospective direct semantic-end observation, not an old
coarse material bbox renamed as an exact native corner. Original V90 RGB uses
F64 pixel-centre four-tap bilinear + uint8 rounding; nearest is diagnostic only.
"""
import copy
import math

REAL_AUTHORITY = 'SOURCE_RGB_SEMANTIC_END_WEAK'
SYNTH_AUTHORITY = 'CONSTRUCTIVE_SOURCE_FIXED_NEAR_ANCHOR_END'
END_FRACTION_OF_EXISTING_V127_STEP = 0.2


def require(value, message):
    if not value:
        raise ValueError(message)


def finite_pair(q):
    return (isinstance(q, (list, tuple)) and len(q) == 2 and
            all(isinstance(x, (int, float)) and math.isfinite(x) for x in q))


def rgb_taps(point, raw_hw):
    """Original V90 geometric support, before final clipped uint8 rounding."""
    x, y = point
    h, w = raw_hw
    if not (-.5 <= x < w - .5 and -.5 <= y < h - .5):
        return [(None, 1.)]  # V90 zeros entire invalid output sample.
    ix, iy = math.floor(x), math.floor(y)
    dx, dy = x - ix, y - iy
    taps = []
    for ax, ay, weight in ((0, 0, (1-dx)*(1-dy)), (1, 0, dx*(1-dy)),
                           (0, 1, (1-dx)*dy), (1, 1, dx*dy)):
        if weight <= 0:
            continue
        px, py = ix + ax, iy + ay
        taps.append(((px, py) if 0 <= px < w and 0 <= py < h else None, weight))
    return taps


def exact96_support(q, L, raw_hw, subset=(0, 96)):
    require(finite_pair(q) and math.isfinite(L) and L > 0, 'Original finite q/L')
    require(0 <= subset[0] < subset[1] <= 96, 'Fixed96 subset')
    mass, invalid = {}, []
    for j in range(*subset):
        for i in range(*subset):
            point = (q[0] + ((i+.5)/96-.5)*L, q[1] + ((j+.5)/96-.5)*L)
            for pixel, weight in rgb_taps(point, raw_hw):
                if pixel is None:
                    invalid.append(weight)
                else:
                    mass.setdefault(pixel, []).append(weight)
    denominator = (subset[1]-subset[0])**2
    return ({p: math.fsum(values)/denominator for p, values in mass.items()},
            math.fsum(invalid)/denominator)


def nearest_diagnostic(point, raw_hw):
    """Never used in RGB footprint or end-label decisions."""
    h, w = raw_hw
    x, y = (math.floor(v+.5) for v in point)
    return (x, y) if 0 <= x < w and 0 <= y < h else None


def candidate_footprint(enclosures, raw_hw):
    """Conservative positive-tap union of closed continuous end enclosures.

    For [x0,x1], every potentially positive linear-interpolation source pixel
    lies in integer floor(x0)..ceil(x1), including exact-integer endpoints.
    No extent clipping/shrinking or invisible-candidate deletion is allowed.
    """
    h, w = raw_hw
    pixels = set()
    outside = False
    require(bool(enclosures), 'All plausible semantic-end candidates required')
    for box in enclosures:
        require(len(box) == 4 and all(math.isfinite(x) for x in box), 'Finite candidate enclosure')
        x0, y0, x1, y1 = box
        require(x0 <= x1 and y0 <= y1, 'Closed candidate bounds')
        outside |= not (0 <= x0 <= x1 < w and 0 <= y0 <= y1 < h)
        for y in range(math.floor(y0), math.ceil(y1)+1):
            for x in range(math.floor(x0), math.ceil(x1)+1):
                if 0 <= x < w and 0 <= y < h:
                    pixels.add((x, y))
                else:
                    outside = True
    return pixels, outside


def sample96_extent(q, L):
    """Closed min/max ORIGINAL V90 sample coordinates, not raw support union."""
    require(finite_pair(q) and isinstance(L, (int, float)) and math.isfinite(L) and L > 0,
            'Original finite q/L')
    a, b = (.5/96-.5)*L, (95.5/96-.5)*L
    return [q[0]+a, q[1]+a, q[0]+b, q[1]+b]


def end_related96_mask(q, L, raw_hw, end_pixels):
    """Bool list[96][96]: a positive VALID original RGB tap touches end set."""
    return [[any(p is not None and weight > 0 and p in end_pixels
                 for p, weight in rgb_taps((q[0]+((i+.5)/96-.5)*L,
                                           q[1]+((j+.5)/96-.5)*L), raw_hw))
             for i in range(96)] for j in range(96)]


def sampled_candidate_relation(q, L, raw_hw, enclosures):
    """No endpoint alternative shrinks or disappears at downsampling tap holes.

    Each closed continuous enclosure must be inside the original sample-centre
    extent and have >=1 valid positive-tap output intersection. ALL conservative
    candidate raw pixels still require opacity, even if not sampled by RGB96.
    This is crop opportunity, not an independent material/end identity label.
    """
    extent = sample96_extent(q, L)
    rows = []
    for box in enclosures:
        pixels, outside = candidate_footprint([box], raw_hw)
        mask = end_related96_mask(q, L, raw_hw, pixels)
        count = sum(sum(line) for line in mask)
        inside = (extent[0] <= box[0] <= box[2] <= extent[2] and
                  extent[1] <= box[1] <= box[3] <= extent[3])
        rows.append(dict(candidate_enclosure_xyxy=list(box),
                         continuous_enclosure_inside_original96_sample_extent=inside,
                         candidate_raw_footprint_in_frame=not outside,
                         positive_valid_RGB96_sample_intersections=count))
    return dict(original96_sample_coordinate_extent_xyxy=extent, candidates=rows,
                all_enclosures_inside_original96_sample_extent=bool(rows) and all(
                    r['continuous_enclosure_inside_original96_sample_extent'] for r in rows),
                every_enclosure_has_positive_valid_original96_sample=bool(rows) and all(
                    r['positive_valid_RGB96_sample_intersections'] > 0 for r in rows),
                all_candidate_raw_footprints_in_frame=bool(rows) and all(
                    r['candidate_raw_footprint_in_frame'] for r in rows))


def rectangle_pixels(box):
    require(len(box) == 4 and all(type(x) is int for x in box), 'Frozen integer half-open paste bbox')
    x0, y0, x1, y1 = box
    require(x0 < x1 and y0 < y1, 'Positive rectangle')
    return {(x, y) for y in range(y0, y1) for x in range(x0, x1)}


def near_anchor_quads(anchor, signs, existing_step):
    """Synthetic convention only: three incident squares at 0.2*V127 step.

    Existing V127 index0 anchor and nearest (.2) edge/face witness spacing fix
    the new locality prospectively. It is not a real native correspondence.
    """
    require(len(anchor) == len(signs) == 3 and all(s in (-1, 1) for s in signs), 'Declared constructive anchor/signs')
    require(math.isfinite(existing_step) and existing_step > 0, 'Original constructive step')
    side = END_FRACTION_OF_EXISTING_V127_STEP * existing_step
    result = []
    for axis in range(3):
        a, b = [i for i in range(3) if i != axis]
        vertices = []
        for u, v in ((0, 0), (1, 0), (1, 1), (0, 1)):
            p = list(anchor)
            p[a] -= signs[a]*side*u
            p[b] -= signs[b]*side*v
            vertices.append(p)
        result.append(vertices)
    return result


def evaluate_plan(original, endpoint, placement):
    """Every attempted row is returned; no image or label admission occurs."""
    before = copy.deepcopy(original)
    require(original['development_role'] in ('SOURCE_TRAIN', 'SOURCE_DEVELOPMENT'), 'Source TRAIN only')
    require(type(original['native_slot']) is int and 0 <= original['native_slot'] < 8, 'Original native0..7')
    result = {'original_input': before, 'endpoint_observation': copy.deepcopy(endpoint),
              'placement': copy.deepcopy(placement), 'status': 'UNKNOWN', 'reasons': [],
              'direct_end_weak_states': dict(BEFORE='UNKNOWN', COVER='UNKNOWN', SHAM='UNKNOWN'),
              'new_native_visibility_target': -1, 'new_quality_target': -1,
              'training_admission': False, 'broad_support_is_not_native_decoder': True}
    reasons = result['reasons']
    q, L = original.get('query_xy'), original.get('L')
    if not finite_pair(q) or not isinstance(L, (int, float)) or not math.isfinite(L) or L <= 0:
        reasons.append('ORIGINAL_Q_L_MISSING'); return result
    if not original['original_query_available']:
        reasons.append('ORIGINAL_INPUT_U'); return result
    authority = endpoint.get('authority')
    localized = (authority in (REAL_AUTHORITY, SYNTH_AUTHORITY) and
                 endpoint.get('native_slot') == original['native_slot'] and
                 endpoint.get('selected_instance_supported') is True and
                 endpoint.get('intended_part_supported') is True and
                 endpoint.get('queried_end_localization_supported') is True and
                 endpoint.get('all_plausible_candidates_enclosed') is True and
                 endpoint.get('before_end_visible_supported') is True and
                 bool(endpoint.get('evidence_binding')) and bool(endpoint.get('end_locality_id')))
    if not localized:
        reasons.append('DIRECT_END_OBSERVATION_OR_COMPLETE_CANDIDATE_LOCALITY_UNKNOWN'); return result
    candidates, outside = candidate_footprint(endpoint['candidate_enclosures_xyxy'], original['raw_hw'])
    full, invalid = exact96_support(q, L, original['raw_hw'])
    c32, _ = exact96_support(q, L, original['raw_hw'], (32, 64))
    c4, _ = exact96_support(q, L, original['raw_hw'], (47, 49))
    result.update(end_candidate_raw_pixels=len(candidates), original_rgb96_invalid_mass=invalid,
                  end_candidate_fully_in_rgb96_support=not outside and candidates <= full.keys(),
                  weighted_end_enclosure_diagnostic={name: math.fsum(w for p, w in mass.items() if p in candidates)
                                                     for name, mass in (('full96', full), ('central32', c32), ('central4', c4))})
    sampled = sampled_candidate_relation(q, L, original['raw_hw'], endpoint['candidate_enclosures_xyxy'])
    result['sampled_domain_relation'] = sampled
    result['unsampled_candidate_raw_pixels_retained_for_opacity'] = len(candidates-full.keys())
    if outside or not candidates or not all(sampled[k] for k in (
            'all_enclosures_inside_original96_sample_extent',
            'every_enclosure_has_positive_valid_original96_sample',
            'all_candidate_raw_footprints_in_frame')):
        reasons.append('END_OUTSIDE_OR_NO_POSITIVE_ORIGINAL96_SAMPLED_OPPORTUNITY_UNKNOWN_NOT_EXTERNAL'); return result
    require(all(type(d) is int for d in placement['sham_translation_xy']) and len(placement['sham_translation_xy']) == 2, 'Frozen integer same-object translation')
    cover = rectangle_pixels(placement['cover_bbox_xyxy'])
    dx, dy = placement['sham_translation_xy']
    sham = {(x+dx, y+dy) for x, y in cover}
    h, w = original['raw_hw']
    controls = {'cover_bbox_contains_all_end_support': candidates <= cover,
                'sham_bbox_disjoint_all_end_support': not bool(candidates & sham),
                'cover_and_sham_in_frame': all(0 <= x < w and 0 <= y < h for x, y in cover | sham),
                'same_bbox_and_positive_mass_translation': len(cover) == len(sham) and (dx != 0 or dy != 0),
                'sham_has_some_original_rgb96_support': bool(sham & full.keys())}
    surrounding = endpoint.get('surrounding_material_witnesses', [])
    supported_surrounding = [tuple(o['raw_pixel_xy']) for o in surrounding
                            if o.get('material_observed') is True and o.get('same_selected_part_supported') is True
                            and o.get('evidence_binding')]
    controls['certified_surrounding_part_retained_by_both_bbox_transforms'] = bool(supported_surrounding) and all(
        p in full and p not in candidates and p not in cover and p not in sham for p in supported_surrounding)
    result['geometry_controls'] = controls
    if not all(controls.values()):
        reasons.extend(name for name, valid in controls.items() if not valid)
        result['status'] = 'GEOMETRY_FAILURE_RETAINED'; return result
    result['status'] = 'GEOMETRY_READY_PIXEL_PROOFS_PENDING'
    reasons.append('NO_DIRECT_END_LABEL_OR_ADMISSION_BEFORE_ACTUAL_OPACITY_RGB_AND_WITNESS_PROOF')
    require(original == before, 'Original metadata unchanged')
    return result


def close_actual_proof(plan, proof):
    """Metadata contract for a later separately authorized actual pixel proof."""
    result = copy.deepcopy(plan)
    if plan['status'] != 'GEOMETRY_READY_PIXEL_PROOFS_PENDING':
        return result
    names = ('original_q_L_native_flags_and_all8_inputs_unchanged',
             'before_original_RGB96_valid96_target_context_exact',
             'every_candidate_positive_RGB_tap_cover_alpha_one',
             'every_candidate_positive_RGB_tap_sham_alpha_zero',
             'distinct_foreign_material_supported', 'foreign_in_front_paste_composite_or_source_order',
             'same_transformed_foreign_RGB_alpha_cover_and_sham', 'equal_positive_alpha_pasted_mass',
             'background_outside_paste_exact', 'original_crop_cover_RGB_bytes_changed',
             'original_crop_sham_has_donor_RGB_support',
             'end_related_original96_COVER_bytes_changed',
             'end_related_original96_SHAM_bytes_unchanged',
             'certified_surrounding_material_RGB_unchanged_both')
    if not all(proof.get(name) is True for name in names) or not proof.get('evidence_bindings'):
        result['status'] = 'PIXEL_PROOF_FAILURE_RETAINED'
        result['reasons'] = [name for name in names if proof.get(name) is not True]
        return result
    result['status'] = 'CONTROLLED_DIRECT_END_WEAK_PAIR_SUPPORTED'
    result['direct_end_weak_states'] = dict(BEFORE='VISIBLE_WEAK', COVER='EXTERNALLY_BLOCKED_WEAK', SHAM='VISIBLE_WEAK')
    result['authority'] = plan['endpoint_observation']['authority'] + '_WITH_SAMPLED_DOMAIN_CONTROLLED_FOREIGN_INTERVENTION'
    result['actual_proof_metadata'] = copy.deepcopy(proof)
    result['reasons'] = ['DIRECT_END_STATE_SEPARATE_FROM_OLD_NATIVE_D_E_S_CENTER_OWNER_AND_BROAD_SUPPORT2']
    return result
