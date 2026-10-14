"""Keep in-canvas image queries when an uncertain coarse projection is outside.

The original S gate and independently certified T have precedence. A coarse
projection alone cannot suppress an available non-S image observation. Frozen
RGB features, classifier parameters and thresholds are reused unchanged.
"""
import cv2
import numpy as np

from . import state_visibility_v2 as V


SCHEMA = 'STATE_VISIBILITY_QUERY_CONSISTENT_V4'


def _nonself_queries(image, raw, hidden, roles, projected):
    V._geometry_inputs(image, raw, hidden, roles, projected)
    masked = np.array(raw, dtype=float, copy=True)
    masked[:8][np.asarray(hidden,dtype=bool)] = [-1.,-1.]
    return masked


def geometry_state_gates_v4(image_bgr, raw_points, initial_self_hidden,
                            initial_roles, initial_projection_xy, *,
                            independent_truncation=None):
    raw_points = _nonself_queries(image_bgr, raw_points, initial_self_hidden,
                                 initial_roles, initial_projection_xy)
    records, eligible = V.geometry_state_gates(
        image_bgr, raw_points, initial_self_hidden, initial_roles,
        initial_projection_xy, independent_truncation=independent_truncation)
    ids = set(eligible.tolist())
    for record in records:
        if (record['state'] == 'T' and not record['physical_truncation_certified']
                and record['raw_query_out_of_frame'] is False):
            record.update(state='U', abstained=True,
                          reason='in_canvas_query_overrides_uncertain_coarse_T',
                          coarse_T_overridden_by_available_image_query=True,
                          initial_projection_was_outside=True)
            ids.add(record['native_id'])
    return records, np.asarray(sorted(ids), dtype=int)


def infer_state_visibility_v4(image_bgr, raw_points, initial_self_hidden,
                              initial_roles, initial_projection_xy, model, *,
                              pallet_mask=None, independent_truncation=None,
                              feature_extractor=None, probability_predictor=None):
    raw_points = _nonself_queries(image_bgr, raw_points, initial_self_hidden,
                                 initial_roles, initial_projection_xy)
    records, ids = geometry_state_gates_v4(
        image_bgr, raw_points, initial_self_hidden, initial_roles,
        initial_projection_xy, independent_truncation=independent_truncation)
    radius = V.patch_radius(raw_points, ids)
    family = ''
    if len(ids):
        if not isinstance(model, V.StateVisibilityModel):
            model = V.load_state_model(str(model))
        family = model.descriptor.get('feature_family', '')
        rgb26 = family.startswith('NON_S_RGB26')
        if rgb26:
            local, details = V.extract_nonS_rgb26(image_bgr, raw_points, ids)
            spatial = None
            radius = details['patch_radius_px']
            if 'GEOMETRY' in family:
                local = np.column_stack([
                    local, extract_geometry_ownership_v4(
                        image_bgr, raw_points, ids, initial_projection_xy)])
            if 'RAW_SPATIAL' in family:
                spatial = V.spatial_features(
                    image_bgr, np.asarray(raw_points, float)[:8], ids,
                    V.patch_radius(raw_points, ids), model.frozen_backbone)
        elif pallet_mask is None:
            from .segmentation import segment
            binding = model.descriptor['pallet_segmenter']
            weights = V.ROOT / binding['weights_repo_relative']
            if V.sha256(weights) != binding['weights_sha256']:
                raise ValueError('Frozen pallet segmenter changed')
            pallet_mask = segment(image_bgr, weights, model.frozen_backbone.device)
        if not rgb26:
            local, spatial = (feature_extractor or V.extract_state_features)(
                image_bgr, raw_points, ids, radius, model.frozen_backbone,
                pallet_mask=pallet_mask)
        probabilities = np.asarray((probability_predictor or V.predict_actual_visibility)(
            local, spatial, model), float)
        if (probabilities.shape != (len(ids),) or not np.isfinite(probabilities).all()
                or np.any((probabilities < 0) | (probabilities > 1))):
            raise ValueError('One finite visibility probability per eligible query required')
        threshold = float(model.descriptor['visible_threshold'])
        margin = float(model.descriptor.get('abstention_margin', 0.))
        if not 0 <= margin <= threshold <= 1 or threshold + margin > 1:
            raise ValueError('Invalid frozen visibility threshold')
        for k, probability in zip(ids, probabilities):
            record = records[k]
            state = (record['initial_role'] if probability >= threshold + margin
                     else 'O' if probability < threshold - margin else 'U')
            if state == 'O' and model.descriptor.get('external_O_enabled', True) is not True:
                state = 'U'
            record.update(state=state, p_visible=float(probability), rgb_evaluated=True,
                          abstained=state == 'U',
                          reason='human_DIRECT_vs_EXTERNAL_RGB_classifier',
                          external_O_enabled=model.descriptor.get('external_O_enabled', True))
    height,width = image_bgr.shape[:2]
    projected = np.asarray(initial_projection_xy)
    projected_supported = [int(k) for k in ids if
        0 <= projected[k,0] < width and 0 <= projected[k,1] < height]
    return dict(states=[record['state'] for record in records], records=records,
                rgb_queried_ids=ids.tolist(), p_visible=[r['p_visible'] for r in records],
                diagnostics=dict(schema=SCHEMA, patch_radius_px=radius,
                    self_excluded_before_RGB=True, coordinates_changed=False,
                    coordinate_quality_is_visibility=False,
                    projected_points_sampled=bool(projected_supported and 'GEOMETRY' in family),
                    projected_RGB_queried_ids=projected_supported if 'GEOMETRY' in family else [],
                    projected_RGB_suppressed_OOB_ids=[int(k) for k in ids
                        if k not in projected_supported] if 'GEOMETRY' in family else [],
                    projected_OOB_feature_policy='zero_missing_12D_block_no_border_extrapolation',
                    projected_coordinates_used_as_observations=False,
                    feature_family=family or None,
                    state_target='human_DIRECT_VISIBLE_vs_EXTERNAL_OCCLUDED',
                    conditional_T_is_independent_physical_fact=False,
                    coarse_T_overridden_ids=[r['native_id'] for r in records
                        if r.get('coarse_T_overridden_by_available_image_query')],
                    U_is_internal_abstention=True))


def extract_geometry_ownership_v4(image_bgr, raw_points, eligible_ids, initial_projection_xy):
    """Bounded image cues at raw/non-S initial-model guidance locations.

    Palette mismatch remains a feature. Edge continuation compares image
    gradients along initially registered non-S edges near and away from the
    query. Initial projections provide directions/context only, never a
    corrected observation or a visibility target. No S coordinate is passed
    to scene construction or sampled as an RGB keypoint query.
    """
    from . import appearance as A
    ids = np.asarray(eligible_ids, int)
    q, projected = np.full((8, 2), np.nan), np.full((8, 2), np.nan)
    q[ids] = np.asarray(raw_points, float)[:8][ids]
    projected[ids] = np.asarray(initial_projection_xy, float)[ids]
    valid = np.zeros(8, bool)
    valid[ids] = True
    scene = A._scene(np.asarray(image_bgr), q, ~valid, valid, A._config(None))
    radius = float(scene["radius"])
    center = np.mean(projected[ids], axis=0) if len(ids) else np.zeros(2)
    # Native cuboid edges; neighbours that are S or otherwise excluded are
    # removed BEFORE any image samples or direction computations.
    neighbours = ((1, 3, 4), (0, 2, 5), (1, 3, 6), (0, 2, 7),
                  (0, 5, 7), (1, 4, 6), (2, 5, 7), (3, 4, 6))
    def sample(array, point):
        h, w = array.shape[:2]
        if not (0 <= point[0] < w and 0 <= point[1] < h):
            return None
        return float(cv2.getRectSubPix(np.asarray(array, np.float32), (1, 1), tuple(map(float, point)))[0, 0])
    def average(array, points, absolute=False):
        values = [sample(array, point) for point in points]
        values = [abs(value) if absolute else value for value in values if value is not None]
        return float(np.mean(values)) if values else 0.
    rows = []
    for k in ids:
        direction = center - projected[k]
        direction /= max(float(np.linalg.norm(direction)), 1e-8)
        tangent = np.array([-direction[1], direction[0]])
        edge_dirs = []
        for j in neighbours[k]:
            if not valid[j]:
                continue
            edge = projected[j] - projected[k]
            if np.linalg.norm(edge) > 1e-6:
                edge_dirs.append(edge / np.linalg.norm(edge))
        values = []
        for source in (q[k], projected[k]):
            h, w = image_bgr.shape[:2]
            if not (0 <= source[0] < w and 0 <= source[1] < h):
                # No replicated-border patch may stand in for absent RGB.
                values.extend([0.] * 12)
                continue
            patch = cv2.getRectSubPix(scene["lab"], (5, 5), tuple(map(float, source))).astype(float)
            lab = np.median(patch.reshape(-1, 3), axis=0)
            palette = np.asarray(scene["palette"]["centers_lab"], float)
            distance = min(4., float(np.min(np.linalg.norm((palette-lab)/[48., 20., 20.], axis=1)))) if len(palette) else 4.
            chroma = float(np.linalg.norm(lab[1:]-128.) / 128.)
            inward = [average(scene["soft_material"], [source + direction*radius*d + tangent*radius*t for t in (-.25, 0, .25)]) for d in (1., 2.)]
            outward = [average(scene["soft_material"], [source - direction*radius*d + tangent*radius*t for t in (-.25, 0, .25)]) for d in (1., 2.)]
            normal_near, normal_far, tangent_near, tangent_far = [], [], [], []
            for edge in edge_dirs:
                normal = np.array([-edge[1], edge[0]])
                gnormal = scene["gx"]*normal[0] + scene["gy"]*normal[1]
                gtangent = scene["gx"]*edge[0] + scene["gy"]*edge[1]
                near = [source+edge*radius*d for d in (.5, 1., 1.5)]
                far = [source+edge*radius*d for d in (2., 3., 4.)]
                normal_near.append(average(gnormal, near, True)/32.)
                normal_far.append(average(gnormal, far, True)/32.)
                tangent_near.append(average(gtangent, near, True)/32.)
                tangent_far.append(average(gtangent, far, True)/32.)
            wedge = [source+direction*radius*d+tangent*radius*t for d in (.5, 1., 1.5, 2.) for t in (-.5, 0., .5)]
            soft = [sample(scene["soft_material"], point) for point in wedge]
            soft = [s for s in soft if s is not None]
            values.extend([distance, chroma, *inward, *outward,
                float(np.mean(normal_near)) if normal_near else 0., float(np.mean(normal_far)) if normal_far else 0.,
                float(np.mean(tangent_near)) if tangent_near else 0., float(np.mean(tangent_far)) if tangent_far else 0.,
                float(np.mean(soft)) if soft else 0., float(np.mean(np.asarray(soft)<.25)) if soft else 1.])
        rows.append(values)
    result = np.asarray(rows, float).reshape(len(ids), len(V.OWNERSHIP_FEATURE_NAMES))
    if not np.isfinite(result).all():
        raise ValueError("Geometry ownership features must be finite")
    return result
