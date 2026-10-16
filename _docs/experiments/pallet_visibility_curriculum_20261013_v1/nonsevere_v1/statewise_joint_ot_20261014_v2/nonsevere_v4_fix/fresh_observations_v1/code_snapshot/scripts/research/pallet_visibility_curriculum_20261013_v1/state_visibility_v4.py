"""Keep in-canvas image queries when an uncertain coarse projection is outside.

The original S gate and independently certified T have precedence. A coarse
projection alone cannot suppress an available non-S image observation. Frozen
RGB features, classifier parameters and thresholds are reused unchanged.
"""
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
                    local, V.extract_geometry_ownership(
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
    return dict(states=[record['state'] for record in records], records=records,
                rgb_queried_ids=ids.tolist(), p_visible=[r['p_visible'] for r in records],
                diagnostics=dict(schema=SCHEMA, patch_radius_px=radius,
                    self_excluded_before_RGB=True, coordinates_changed=False,
                    coordinate_quality_is_visibility=False,
                    projected_points_sampled=bool(len(ids) and 'GEOMETRY' in family),
                    projected_RGB_queried_ids=ids.tolist() if 'GEOMETRY' in family else [],
                    projected_coordinates_used_as_observations=False,
                    feature_family=family or None,
                    state_target='human_DIRECT_VISIBLE_vs_EXTERNAL_OCCLUDED',
                    conditional_T_is_independent_physical_fact=False,
                    coarse_T_overridden_ids=[r['native_id'] for r in records
                        if r.get('coarse_T_overridden_by_available_image_query')],
                    U_is_internal_abstention=True))
