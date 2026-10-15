"""Reuse a validated observed B--B edge normal at its other fallback endpoint.

The immutable v4.6 producer supplies own partial B lines. This version adds
only a missing incident-edge factor for a second initial non-S/current B hull
endpoint. Both projections lie on the same caller-associated physical cuboid
edge. Its native association remains conditional: no new point, RGB query,
line fit, GT coordinate or U/S cost is created. Already fully corrected B
observations are never replaced. Shared factors use a declared additional
variance inflation of two; this is not a fitted correlation/noise model.
"""
from __future__ import annotations

from copy import deepcopy

import cv2
import numpy as np

from . import boundary_partial_observations_v4_6 as BASE
from . import joint_ot_pnp as V3
from .interior_observation import NEIGHBORS


def _quality(observed):
    return (observed['inlier_count'], observed['inlier_fraction'],
            observed['inlier_span_fraction'], -observed['inlier_rms_px'],
            -abs(observed['angle_change_deg']))


def _rays(record, recovery):
    return ([(r, 'correction_record') for r in record.get('rays', [])]
            + [(r, 'expanded_boundary_recovery') for r in recovery.get('expanded_rays', [])]
            + [(a.get('remeasured_ray'), 'observed_line_recentering')
               for a in recovery.get('observed_line_recentering', [])])


def observe_shared_supported_b_lines(correction, geometry, states, fit, *,
                                    localization_model, nonself_scale_px,
                                    image_binding):
    """Return v4.6-compatible partial B normal/interval observation records.

Own supported factors are preserved. A strongest existing RGB-supported ray
per physical B--B edge is associated to an endpoint only if it is an original
fallback B and lacks its own factor for that edge. The other endpoint may
already be fully corrected; its complete observation/cost is unchanged. We
never use raw S/U/O/T/I coordinates or rays, nor corrected B as a new fallback.
"""
    result = BASE.observe_single_supported_b_lines(correction, geometry, states, fit,
        localization_model=localization_model, nonself_scale_px=nonself_scale_px,
        image_binding=image_binding)
    result = deepcopy(result)
    states = [V3._state(s) for s in states]
    hidden = geometry['initial_self_hidden']; roles = geometry['initial_roles']
    base = BASE._records_by_id(correction['records'])
    recovery = BASE._records_by_id(correction.get('boundary_recovery_records', []))
    fitted = BASE._records_by_id(fit)
    observations = {r['native_corner']: r for r in result['boundary_observations']}
    initial_nonS = [k for k in range(8) if not hidden[k]]
    # Model projections are guidance, not image measurements. Gate initial S
    # before indexing even these model coordinates; no S raw field is read.
    projection = np.asarray([V3._finite_array(geometry['initial_projection_xy'][k], (2,),
                                             'initial non-S model projection')
                             for k in initial_nonS])
    hull_index = cv2.convexHull(projection.astype(np.float32), returnPoints=False).ravel()
    hull = [initial_nonS[int(i)] for i in hull_index]
    hull_edges = {tuple(sorted((hull[i], hull[(i+1) % len(hull)]))) for i in range(len(hull))}
    eligible = {k for k in hull if not hidden[k] and states[k] == 'B' and roles[k] == 'B'}
    fallback = {k for k in eligible if k in base and k in fitted
                and not base[k].get('corrected')
                and fitted[k].get('source') == 'VISIBLE_BACKBONE_FALLBACK'}
    best = {}
    cfg = correction['configuration']
    for source in sorted(eligible):
        # State/initial geometry gates precede reading source-local ray fields.
        for ray, origin in _rays(base.get(source, {}), recovery.get(source, {})):
            observed = BASE._supported_ray(ray, source, cfg)
            if observed is None:
                continue
            mate = observed['native_neighbor']
            if mate not in eligible:
                continue
            edge = tuple(sorted((source, mate)))
            if edge not in hull_edges or mate not in NEIGHBORS[source]:
                continue
            candidate = dict(observed=observed, source=source, native_neighbor=mate,
                             record_origin=origin, physical_edge=edge)
            if edge not in best or _quality(observed) > _quality(best[edge]['observed']):
                best[edge] = candidate
    shared_audit = []
    for edge, candidate in sorted(best.items()):
        source = candidate['source']; observed = candidate['observed']
        audit = dict(physical_native_edge=list(edge), source_query_native_corner=source,
                     record_origin=candidate['record_origin'],
                     RGB_supported_sample_count=observed['rgb_supported_sample_count'],
                     added_factor_native_ids=[], existing_own_factor_native_ids=[],
                     complete_B_native_ids_preserved=[], no_new_RGB_queries=True)
        shared_audit.append(audit)
        for destination in edge:
            if destination not in fallback:
                audit['complete_B_native_ids_preserved'].append(destination)
                continue
            mate = edge[1] if destination == edge[0] else edge[0]
            existing = observations.get(destination)
            existing_edges = ([] if existing is None else
                              [l['provenance']['native_neighbor'] for l in existing['lines']])
            if mate in existing_edges:
                audit['existing_own_factor_native_ids'].append(destination)
                continue
            # Create an associated record, not a claim that RGB was queried
            # at destination. The same actual samples/normal/offset are kept.
            associated = deepcopy(observed['ray'])
            associated.update(native_corner=destination, native_neighbor=mate,
                              endpoints=[destination, mate], edge=destination*8+mate)
            proxy = dict(image_pixel_sha256=correction['image_pixel_sha256'],
                         configuration=cfg,
                         records=[dict(native_corner=destination, state='B', corrected=False, rays=[])],
                         boundary_recovery_records=[dict(native_corner=destination,
                             expanded_rays=[associated], observed_line_recentering=[])])
            # Reuse the exact v4.6 normal budget, broad tangent-envelope and
            # display construction for this receiving endpoint, with no IO.
            single = BASE.observe_single_supported_b_lines(proxy, geometry, states, fit,
                localization_model=localization_model, nonself_scale_px=nonself_scale_px,
                image_binding=image_binding)['boundary_observations']
            if len(single) != 1 or single[0]['native_corner'] != destination:
                raise RuntimeError('Validated shared edge failed the fallback endpoint contract')
            added = single[0]
            line = added['lines'][0]
            line['provenance'].update(shared_edge_propagated=True,
                physical_native_edge=list(edge), source_query_native_corner=source,
                receiver_native_corner=destination,
                RGB_queried_at_receiver_for_this_line=False,
                shared_normal_offset_and_samples_reused=True,
                correlated_shared_edge_reuse=True, shared_edge_variance_inflation=2.,
                record_origin=candidate['record_origin'])
            if existing is None:
                observations[destination] = added
            else:
                existing['lines'].append(line)
            audit['added_factor_native_ids'].append(destination)
        if audit['added_factor_native_ids']:
            # The matching own factor, when present, shares the same edge
            # information. Inflate it too; do not claim two independent lines.
            for endpoint in edge:
                if endpoint not in observations:
                    continue
                mate = edge[1] if endpoint == edge[0] else edge[0]
                for line in observations[endpoint]['lines']:
                    if line['provenance']['native_neighbor'] == mate:
                        line['provenance'].update(correlated_shared_edge_reuse=True,
                            shared_edge_variance_inflation=2., physical_native_edge=list(edge),
                            shared_RGB_independence_claimed=False)
    for k, record in sorted(observations.items()):
        if len(record['lines']) > 2:
            raise ValueError('More than two incident hull edge factors at a B corner')
        covariance = V3._finite_array(fitted[k]['fit_covariance_px2'], (2, 2), 'fixed original CAL fit Sigma')
        count = len(record['lines'])
        for line in record['lines']:
            n = np.asarray(line['normal_unit_xy'])
            shared = float(line['provenance'].get('shared_edge_variance_inflation', 1.))
            line['normal_variance_px2'] = count * shared * float(n @ covariance @ n)
            line['provenance'].update(variance_count_inflation=count,
                normal_variance_rule='line_count * shared_reuse_inflation * normal.T @ fixed_original_CAL_fit_Sigma @ normal',
                native_edge_association_is_conditional=True,
                shared_edge_noise_model_calibrated=False)
        n = np.asarray(record['lines'][0]['normal_unit_xy']);d=record['lines'][0]['offset_px']
        q = V3._finite_array(fitted[k]['xy'], (2,), 'fallback B reference')
        display = q - n * (n @ q + d)
        record['association_provenance'].update(normal_projected_display_xy=display.tolist(),
            shared_edge_propagation_is_native_association_hypothesis=True,
            no_new_receiver_RGB_query=True, shared_edge_correlation_not_fitted=True)
        result['records'][k].update(accepted_partial=True, lines=count,
            reason='own_or_shared_RGB_edge_normal_plus_broad_corner_tangent_range',
            propagated_line_count=sum(bool(l['provenance'].get('shared_edge_propagated')) for l in record['lines']),
            normal_projected_display_xy=display.tolist(), coordinate_read=True)
    result.update(schema='FALLBACK_B_SHARED_EDGE_PARTIAL_OBSERVATIONS_V4_7',
                  boundary_observations=[observations[k] for k in sorted(observations)],
                  shared_edge_records=shared_audit,
                  propagated_factor_count=sum(len(a['added_factor_native_ids']) for a in shared_audit),
                  shared_edge_variance_inflation=2.,
                  complete_corrected_B_costs_preserved=True,
                  shared_edge_independence_claimed=False,
                  shared_edge_noise_covariance_calibrated=False,
                  no_new_receiver_image_or_line_queries=True)
    return result


observe_shared_supported_B_lines = observe_shared_supported_b_lines
