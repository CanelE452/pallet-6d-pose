"""State-independent local cue union on frozen automatic visible anchors.

This module reads no annotations, model weights, pose banks or references.  It
reuses the original v2 line fitter and conditional uncertainty estimator.  A
cached, independently bound typed FACE observation may replace a point only
when the new union has no complete alternative to q0.
"""
from __future__ import annotations

from copy import deepcopy
from itertools import combinations
import hashlib
import json

import cv2
import numpy as np

from scripts.research.pallet_visibility_curriculum_20261013_v1 import visible_bi_correction_v2 as OLD
from scripts.research.pallet_visibility_curriculum_20261013_v1.interior_observation import NEIGHBORS
from scripts.research.pallet_foundation_mask_20261013_v1.uncertainty import intersection_uncertainty

SCHEMA = 'COMMON_LOCAL_OBSERVATION_UNION_V1'
CONFIGURATION = deepcopy(OLD.DEFAULT_CONFIG)
SOURCE = 'visible_bi_correction_v2.DEFAULT_CONFIG/_line + boundary_refine._fit_line'
TRACE_FIELDS = ('mask_inside', 'mask_outside', 'mask_transition', 'mask_context_transition',
                'rgb_normal_gradient', 'rgb_gradient_alignment', 'rgb_color_transition', 'normal_offset_px')


def _json(value):
    if isinstance(value, dict):
        return {str(k): _json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json(value.tolist())
    if isinstance(value, np.generic):
        return _json(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _hash(value):
    return hashlib.sha256(json.dumps(_json(value), sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _covariance(value, name):
    covariance = np.asarray(value, float)
    if covariance.shape != (2, 2) or not np.isfinite(covariance).all() or not np.allclose(covariance, covariance.T, atol=1e-12, rtol=0):
        raise ValueError(name + ' must be finite symmetric 2x2')
    np.linalg.cholesky(covariance)
    return covariance


def physical_line_source_id(record):
    """Identity for proven reuse of the same actual inlier pixel samples.

    Reversing sample order or normal orientation does not create new image
    evidence. Different pixel sets are not certified independent by this ID.
    """
    points = np.asarray(record['observed_points'], float)
    inliers = np.asarray(record['inlier_mask'], bool)
    points = points[inliers]
    order = np.lexsort((points[:, 1], points[:, 0]))
    return 'pixels:' + _hash(dict(inlier_pixel_samples=points[order].tolist()))


def _sampling_trace(image, mask, lab, gradients, q, k, mate, guide, center):
    """Both cue summaries use exactly the v2 along/offset search coordinates.

    Cue-specific probes retain their original units and meanings.  Both mask
    and RGB statistics are measured at each cue's selected samples, including
    cross-cue statistics.  No actual B/I state is read by this function.
    """
    cfg = CONFIGURATION; length = float(np.linalg.norm(guide))
    if length < 10.:
        return dict(query_grid_sha256=None, cue_summaries={}), None
    direction = guide / length; normal = np.array([-direction[1], direction[0]])
    if normal @ (center - q[k]) < 0:
        normal = -normal
    end = min(cfg['along_end_px'], .8 * length)
    if end <= cfg['along_start_px']:
        return dict(query_grid_sha256=None, cue_summaries={}), None
    along = np.linspace(cfg['along_start_px'], end, int(cfg['along_samples']))
    offsets = np.arange(-cfg['normal_search_radius_px'], cfg['normal_search_radius_px'] + .01, cfg['normal_step_px'])
    xy = q[k] + along[:, None, None] * direction + offsets[None, :, None] * normal
    probe = cfg['boundary_probe_px']; context = cfg['boundary_context_px']
    inside = sum(OLD._sample(mask, xy + normal * probe * s) for s in (.5, 1., 1.5)) / 3
    outside = sum(OLD._sample(mask, xy - normal * probe * s) for s in (.5, 1., 1.5)) / 3
    inward = OLD._sample(mask, xy + normal * context); outward = OLD._sample(mask, xy - normal * context)
    x, y = (OLD._sample(v, xy) for v in gradients)
    energy = np.linalg.norm(x * normal[0] + y * normal[1], axis=-1)
    magnitude = np.sqrt(np.sum(x * x + y * y, axis=-1))
    alignment = energy / np.maximum(magnitude, 1e-12)
    color = np.linalg.norm(OLD._sample(lab, xy + normal * 1.5) - OLD._sample(lab, xy - normal * 1.5), axis=-1)
    contrast = inside - outside; h, w = image.shape[:2]
    def in_frame(margin):
        a, b = xy + normal * margin, xy - normal * margin
        return ((a[..., 0] >= 0) & (a[..., 0] < w - 1) & (a[..., 1] >= 0) & (a[..., 1] < h - 1)
                & (b[..., 0] >= 0) & (b[..., 0] < w - 1) & (b[..., 1] >= 0) & (b[..., 1] < h - 1))
    valid_b = ((inside >= cfg['minimum_inside']) & (outside <= cfg['maximum_outside'])
               & (contrast >= cfg['minimum_transition']) & (inward >= .50) & (outward <= .45)
               & in_frame(max(context, 1.5 * probe)))
    valid_i = ((energy >= cfg['minimum_gradient']) & (alignment >= cfg['minimum_gradient_alignment'])
               & (color >= cfg['minimum_color_transition']) & in_frame(1.5))
    scores = dict(B=.65 * contrast + .15 * (inward - outward) - .015 * np.abs(offsets),
                  I=energy * alignment - .05 * np.abs(offsets))
    values = dict(mask_inside=inside, mask_outside=outside, mask_transition=contrast,
                  mask_context_transition=inward - outward, rgb_normal_gradient=energy,
                  rgb_gradient_alignment=alignment, rgb_color_transition=color,
                  normal_offset_px=np.broadcast_to(offsets, energy.shape))
    summaries = {}
    for cue, valid in (('B', valid_b), ('I', valid_i)):
        score = np.where(valid, scores[cue], -np.inf); best = score.argmax(axis=1)
        keep = np.isfinite(score[np.arange(len(along)), best]); rows = np.flatnonzero(keep)
        mask_safe = in_frame(max(context, 1.5 * probe))[rows, best[keep]]
        summary_values = {}
        for name, array in values.items():
            safe = mask_safe if name.startswith('mask_') else np.ones(len(rows), dtype=bool)
            observed_values = array[rows, best[keep]][safe]
            summary_values[name] = dict(value=float(np.median(observed_values)) if len(observed_values) else 0.,
                                       valid=bool(len(observed_values)), sample_count=int(len(observed_values)))
        summaries[cue] = dict(selected_sample_count=int(keep.sum()), total_samples=len(along),
            valid_grid_fraction=float(valid.mean()), selected_offsets_px=offsets[best[keep]].tolist(),
            selected_grid_indices=np.stack([rows, best[keep]], axis=1).tolist(),
            values=summary_values)
    grid_hash = hashlib.sha256(b'V2-ALONG-OFFSET-XY\0' + np.ascontiguousarray(xy, dtype='<f8').tobytes()).hexdigest()
    return dict(query_grid_sha256=grid_hash, grid_shape=list(xy.shape), along_px=along.tolist(),
                offsets_px=offsets.tolist(), direction_xy=direction.tolist(), inward_normal_xy=normal.tolist(),
                probe_offsets_px=dict(boundary=[probe * s for s in (.5, 1., 1.5)] + [context], rgb=[1.5]),
                cue_summaries=summaries), xy


def _ray(image, mask, lab, gradients, q, k, mate, projection, cue, sampling):
    ray = OLD._line(image, mask, lab, gradients, q, k, mate, projection[mate] - projection[k], projection.mean(0), cue, CONFIGURATION)
    ray = deepcopy(ray); ray['original_v2_edge'] = ray['edge']
    # Each pair uses distinct mates, so the original numeric edge index is
    # unique inside that pair and preserves the original v2 bootstrap seeds.
    ray.update(native_corner=k, native_neighbor=mate,
               source_line_id=f'k{k}:m{mate}:cue{cue}', cue=cue,
               query_grid_sha256=sampling['query_grid_sha256'],
               cue_summary=deepcopy(sampling.get('cue_summaries', {}).get(cue, {})))
    if ray.get('supported'):
        observed = np.asarray(ray['observed_points']); inlier = np.asarray(ray['inlier_mask'], bool)
        n = np.asarray(ray['normal']); tangent = np.array([-n[1], n[0]])
        positions = observed[inlier] @ tangent
        ray['actual_inlier_tangent_interval_px'] = [float(positions.min()), float(positions.max())]
        ray['physical_source_id'] = physical_line_source_id(ray)
    return ray


def _keep(k, state, point, raw_measurements):
    sigma = CONFIGURATION['fallback_boundary_sigma_px' if state == 'B' else 'fallback_interior_sigma_px']
    covariance = np.eye(2) * sigma ** 2; reason = 'original_v2_state_fallback_no_matching_raw_CAL'
    matched = None
    for row in raw_measurements or []:
        if row.get('native_corner') != k:
            continue
        if matched is not None:
            raise ValueError('Duplicate original raw CAL native corner')
        matched = row
    if matched is not None and matched.get('state') == state and np.array_equal(np.asarray(matched.get('xy'), float), point):
        original = _covariance(matched['covariance_px2'], 'original raw CAL covariance')
        if matched.get('source') == 'VISIBLE_BACKBONE_FALLBACK' and np.array_equal(original, covariance):
            covariance = original
            reason = 'matching_original_raw_CAL_covariance'
        else:
            reason = 'original_v2_fallback_prior_CAL_conditional_noise_or_source_does_not_match_raw'
    candidate = dict(candidate_id=f'k{k}:keep', kind='keep', native_corner=k, q0=point.tolist(), raw_intersection_xy=None,
                blend_alpha=0., applied_xy=point.tolist(), covariance_px2=covariance.tolist(),
                covariance_source=reason, conditional_covariance=None, source_line_ids=[], physical_source_ids=[],
                source_cues=[], source_neighbors=[], intersection_sine=None,
                tie_break_key=[0, float(point[0]), float(point[1]), ''])
    if reason == 'matching_original_raw_CAL_covariance':
        if 'fit_covariance_px2' in matched:
            candidate['fit_covariance_px2'] = _covariance(matched['fit_covariance_px2'], 'matching raw CAL fit covariance').tolist()
        if 'fit_reliability_weight' in matched:
            reliability = float(matched['fit_reliability_weight'])
            if not np.isfinite(reliability) or reliability <= 0:
                raise ValueError('Matching raw CAL fit reliability must be finite positive')
            candidate['fit_reliability_weight'] = reliability
        candidate['original_measurement_source'] = matched.get('source')
    return candidate


def _full_candidates(k, point, rays, image_hw):
    h, w = image_hw; result = []; rejected = []
    for a, b in combinations([r for r in rays if r.get('supported')], 2):
        if a['native_neighbor'] == b['native_neighbor']:
            continue
        ids = [a['source_line_id'], b['source_line_id']]
        matrix = np.asarray([a['normal'], b['normal']], float); sine = abs(float(np.linalg.det(matrix)))
        if sine < CONFIGURATION['minimum_intersection_sine']:
            rejected.append(dict(source_line_ids=ids, reason='parallel_or_ill_conditioned_lines')); continue
        intersection = np.linalg.solve(matrix, -np.asarray([a['offset'], b['offset']]))
        if (not np.isfinite(intersection).all() or not 0 <= intersection[0] < w or not 0 <= intersection[1] < h
                or np.linalg.norm(intersection - point) > CONFIGURATION['maximum_corner_shift_px']):
            rejected.append(dict(source_line_ids=ids, raw_intersection_xy=intersection.tolist(), reason='intersection_outside_image_or_v2_trust_radius')); continue
        correction = dict(edge_records=[a, b], corner_records=[dict(corner=k, refined=True,
                          candidate=intersection.tolist(), incident_shared_edges=[a['edge'], b['edge']])])
        covariances, _ = intersection_uncertainty(correction, repetitions=int(CONFIGURATION['line_resampling']),
            block_length=3, normal_floor_px=1., seed=int(CONFIGURATION['seed']))
        conditional = covariances.get(k)
        if not conditional or conditional['successful_pair_resamples'] < 16:
            rejected.append(dict(source_line_ids=ids, reason='insufficient_line_resamples')); continue
        covariance = _covariance(conditional['covariance_px2'], 'candidate conditional covariance')
        variance = float(np.linalg.eigvalsh(covariance).max()); sigma2 = CONFIGURATION['detector_sigma_px'] ** 2
        alpha = min(CONFIGURATION['maximum_alpha'], sigma2 / (sigma2 + variance))
        applied = point + alpha * (intersection - point)
        combined = (1 - alpha) ** 2 * np.eye(2) * sigma2 + alpha ** 2 * covariance + np.eye(2)
        result.append(dict(candidate_id=f'k{k}:intersection:' + '+'.join(ids), kind='intersection', native_corner=k,
            q0=point.tolist(), raw_intersection_xy=intersection.tolist(), blend_alpha=float(alpha), applied_xy=applied.tolist(),
            covariance_px2=combined.tolist(), covariance_source='own_v2_bootstrap_blended_with_detector_plus_identity',
            conditional_covariance=deepcopy(conditional), source_line_ids=ids,
            physical_source_ids=[a['physical_source_id'], b['physical_source_id']], source_cues=[a['cue'], b['cue']],
            source_neighbors=[a['native_neighbor'], b['native_neighbor']], intersection_sine=sine,
            tie_break_key=[1, float(applied[0]), float(applied[1]), '+'.join(ids)]))
    # Numeric duplicates are one action. Stable source order chooses a
    # representative and keeps every duplicate's provenance as an alias.
    deduplicated = []
    for candidate in sorted(result, key=lambda r: r['tie_break_key']):
        duplicate = next((r for r in deduplicated if np.array_equal(r['applied_xy'], candidate['applied_xy'])), None)
        if duplicate is None:
            candidate['equivalent_source_pairs'] = []; deduplicated.append(candidate)
        else:
            duplicate['equivalent_source_pairs'].append(deepcopy(candidate))
    return deduplicated, rejected


def _bound_partial(row, k, state, binding, q0):
    if row.get('native_corner') != k or row.get('state') != state:
        raise ValueError('Cached typed partial native/state mismatch')
    if not row.get('replaces_complete_BI_likelihood') or not row.get('explicit_complete_BI_likelihood_revocation'):
        raise ValueError('Typed partial must explicitly replace the complete likelihood')
    existing = row.get('image_binding', {})
    if existing.get('sha256') != binding.get('sha256') or existing.get('image_hw') != binding.get('image_hw'):
        raise ValueError('Cached typed partial image binding mismatch')
    archive = row['association_provenance']
    if not np.array_equal(np.asarray(archive['original_raw_xy'], float), q0):
        raise ValueError('Cached typed partial does not share the original N0 anchor')
    original = dict(native_corner=k, state=state, xy=deepcopy(archive['original_observed_xy']),
        covariance_px2=deepcopy(archive['original_observed_covariance_px2']),
        fit_covariance_px2=deepcopy(archive['original_effective_fit_covariance_px2']),
        fit_reliability_weight=archive['original_fit_reliability_weight'],
        source=archive['original_measurement_source'], protection_px=CONFIGURATION['visible_protection_px'])
    _covariance(original['covariance_px2'], 'partial original covariance')
    _covariance(original['fit_covariance_px2'], 'partial original effective covariance')
    result = deepcopy(row)
    for line in result['lines']:
        record = line.get('provenance', {}).get('line_record')
        if record is not None:
            line['physical_source_ids'] = [physical_line_source_id(record)]
    return result, original


def observe(image_bgr, mask, native_points, geometry, states, *, frame_id, image_binding,
            shared_partial=None, raw_measurements=None):
    """Return the common numeric candidate bank; scorer conditions are absent.

    ``shared_partial`` is the exact, same-N0 cached FACE boundary_observations
    list. ``raw_measurements`` can provide *original* CAL covariance only when
    its coordinates exactly equal this anchor's uncorrected N0 q0.
    """
    image = np.asarray(image_bgr); mask = np.asarray(mask, float); q = np.asarray(native_points, float)
    states = list(states)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 1:
        raise ValueError('Actual nonempty uint8 BGR image required')
    if mask.shape != image.shape[:2] or not np.isfinite(mask).all() or np.any((mask < 0) | (mask > 1)):
        raise ValueError('Same bound mask probabilities in [0,1] required')
    if q.shape not in ((8, 2), (9, 2)) or len(states) != 8 or any(s not in ('B','I','S','O','T','U') for s in states):
        raise ValueError('Native q8/9 and authoritative automatic states required')
    if list(image.shape[:2]) != list(geometry['image_hw']) or image_binding.get('image_hw') != list(image.shape[:2]):
        raise ValueError('Geometry/image binding canvas mismatch')
    file_sha = image_binding.get('sha256')
    if not isinstance(file_sha, str) or len(file_sha) != 64 or any(c not in '0123456789abcdef' for c in file_sha):
        raise ValueError('Original image file SHA256 binding required')
    pixel_sha = OLD.image_sha256(image)
    if image_binding.get('image_pixel_sha256', pixel_sha) != pixel_sha:
        raise ValueError('Image pixel binding mismatch')
    hidden = np.asarray(geometry['initial_self_hidden'], bool)
    projection = np.asarray(geometry['initial_projection_xy'], float)
    xyz = np.asarray(geometry['native_xyz'], float)
    if hidden.shape != (8,) or projection.shape != (8, 2) or xyz.shape != (8, 3) or not np.isfinite(projection).all() or not np.isfinite(xyz).all():
        raise ValueError('Finite original registered native geometry required')
    if any((s == 'S') != bool(hidden[k]) for k, s in enumerate(states)):
        raise ValueError('Initial self-hidden gates must remain fixed')
    lab = cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32), (3, 3), .8)
    gradients = cv2.Sobel(lab, cv2.CV_32F, 1, 0, ksize=3) / 8., cv2.Sobel(lab, cv2.CV_32F, 0, 1, ksize=3) / 8.
    partial_by_id = {}
    for row in shared_partial or []:
        k = row['native_corner']
        if k in partial_by_id:
            raise ValueError('Duplicate cached typed partial native corner')
        partial_by_id[k] = row
    rows = []; h, w = image.shape[:2]
    binding = dict(image_binding, image_pixel_sha256=pixel_sha)
    for k, state in enumerate(states):
        row = dict(native_corner=k, pixel_queried=False, candidates=[], rays=[], samplings=[],
                   partial_line=None, partial_original_visible=None)
        rows.append(row)
        if state not in ('B', 'I'):
            row['reason'] = 'excluded_before_local_pixel_query'; continue
        if not np.isfinite(q[k]).all() or not 0 <= q[k, 0] < w or not 0 <= q[k, 1] < h:
            raise ValueError('Visible anchor must retain actual in-frame q0')
        row.update(pixel_queried=True, q0=q[k].tolist(), initial_projection_xy=projection[k].tolist(),
                   native_xyz=xyz[k].tolist())
        for mate in NEIGHBORS[k]:
            if hidden[mate] or states[mate] == 'S':
                continue
            sampling, _ = _sampling_trace(image, mask, lab, gradients, q, k, mate, projection[mate] - projection[k], projection.mean(0))
            sampling['native_neighbor'] = mate; row['samplings'].append(sampling)
            row['rays'].extend(_ray(image, mask, lab, gradients, q, k, mate, projection, cue, sampling) for cue in ('B', 'I'))
        alternatives, rejected = _full_candidates(k, q[k], row['rays'], image.shape[:2])
        keep = _keep(k, state, q[k], raw_measurements)
        # A measured zero-motion hypothesis is numerically the existing q0
        # action; keep takes the deterministic tie and retains the source IDs.
        aliases = [r for r in alternatives if np.array_equal(r['applied_xy'], keep['applied_xy'])]
        keep['equivalent_source_pairs'] = deepcopy(aliases)
        alternatives = [r for r in alternatives if not np.array_equal(r['applied_xy'], keep['applied_xy'])]
        row.update(candidates=[keep] + alternatives, rejected_pairs=rejected,
                   full_alternative_count=len(alternatives), reason='common_union_with_full_alternatives' if alternatives else 'q0_only')
        if not alternatives and k in partial_by_id:
            row['partial_line'], row['partial_original_visible'] = _bound_partial(partial_by_id[k], k, state, binding, q[k])
            row['reason'] = 'existing_typed_partial_replaces_full_likelihood'
    queried = [r['native_corner'] for r in rows if r['pixel_queried']]
    return _json(dict(schema=SCHEMA, frame_id=str(frame_id), states=states, corners=rows, image_binding=binding,
        image_hw=list(image.shape[:2]), initial_projection_xy=projection, native_xyz=xyz,
        cf_extents=np.asarray(geometry.get('cf_extents', np.ptp(xyz, axis=0)), float),
        configuration=deepcopy(CONFIGURATION), configuration_source=SOURCE,
        queried_native_ids=queried, sampling_sha256=_hash([r['samplings'] for r in rows]),
        provenance=dict(GT_inputs=False, state_specific_neighbor_filter=False, both_cues_for_every_legal_incident_neighbor=True,
            cue_selection_depends_on_scorer_condition=False, old_winner_candidates_reused=False,
            maximum_lines_per_corner=6, maximum_intersections_before_deduplication=12,
            learned_candidate_averaging=False, partial_rule='same_N0_cached_FACE_no_full_alternative',
            v2_all_intersection_agreement_veto_removed_for_both_arms=True,
            line_support_rule='original_v2_fit_inlier_count_fraction_span_rms',
            conditional_covariance_is_semantic_probability=False)))
