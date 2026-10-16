"""Recover the original e inputs and output from immutable saved archives.

The later E_FROZEN/e_run reobservations are never read here. No inference,
pixel query, line search, fitting, or optimization occurs. The v4_15 input
archive preserves the pre-e observation and the e result separately.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np


VIS = Path('_docs/experiments/pallet_visibility_curriculum_20261013_v1/nonsevere_v1')
BOUNDARY = VIS/'statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1'
ARCHIVE = BOUNDARY/'fixed_wd_archived_tangent_eval_v4_15'
FINAL = BOUNDARY/'cached_native_edge_candidates_v4_17'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def pose_check(a, b, name):
    require(a['available'] == b['available'], name+'/availability')
    difference = 0.
    for key in ('R_cf', 'R_physical', 'centroid', 'cf_extents'):
        x, y = np.asarray(a[key], float), np.asarray(b[key], float)
        require(x.shape == y.shape and np.isfinite(x).all() and np.isfinite(y).all(), name+'/'+key+'/finite shape')
        d = float(np.max(np.abs(x-y)))
        require(d <= 1e-12, name+'/'+key+'/numeric pose changed')
        difference = max(difference, d)
    if 'physical_registration_Q' in a and 'physical_registration_Q' in b:
        d = float(np.max(np.abs(np.asarray(a['physical_registration_Q'])-b['physical_registration_Q'])))
        require(d <= 1e-12, name+'/registration changed')
        difference = max(difference, d)
    return difference


def compact_region(value):
    keys = ('native_corner', 'native_id', 'state', 'shape', 'center_xy', 'half_axes_px',
            'angle_deg', 'halfplanes', 'outside_canvas_sides', 'outside_canvas_side',
            'source', 'provenance', 'candidate_pose_inputs_used', 'GT_coordinates_used',
            'independent_precise_support', 'canvas_clipped')
    return {key: deepcopy(value[key]) for key in keys if key in value}


def actual_corrections(root, wanted):
    """The exact pre-e final RGB correction, including v4_3b recovery rays."""
    sources = []
    for part in ('CAL', 'TEST'):
        folder = root/BOUNDARY/'evaluation_v3'
        seal = read(folder/(part+'_RECOVERY_SEAL.json'))
        path = folder/(part+'_RECOVERY_PREDICTIONS.jsonl.gz')
        require(sha(path) == seal['predictions_sha256'], part+'/original recovery journal changed')
        sources.append((path, 'results.FULL', seal['predictions_sha256']))
    folder = root/VIS/'remainder_after_test49_v1/fresh_observations_v4_14'
    seal = read(folder/'OBSERVATION_SEAL.json')
    path = folder/'OBSERVATIONS.jsonl.gz'
    require(sha(path) == seal['observations_sha256'], 'Remainder original RGB observations changed')
    sources.append((path, 'correction', seal['observations_sha256']))
    result = {}
    for path, key, digest in sources:
        with gzip.open(path, 'rt') as stream:
            for line in stream:
                row = json.loads(line)
                if row['id'] in wanted:
                    correction = row['results']['FULL'] if key == 'results.FULL' else row['correction']
                    result[row['id']] = dict(correction=correction, image_sha256=row['image_sha256'],
                        binding=dict(path=str(path.relative_to(root)), sha256=digest, source_field=key,
                                     input_row_sha256=canonical(row)))
    require(set(result) == wanted, 'Original final RGB correction population differs')
    return result


def load_e_trace(source_root, ids, *, expected_final_poses=None):
    """Return actual archived e process, keyed by requested exact frame IDs.

    Optional expected_final_poses is {id: actual_pose} from the current E0
    prediction journal. When supplied, every requested output is numerically
    checked against it. Paths in the returned provenance remain repo-relative.
    """
    root, wanted = Path(source_root), set(ids)
    require(len(ids) == len(wanted), 'Duplicate requested trace IDs')
    ap, fp = root/ARCHIVE, root/FINAL
    seal = read(ap/'INPUT_SEAL.json')
    require(seal['rows'] == 238, 'Complete archived238 e inputs required')
    require(sha(ap/'INPUTS.jsonl.gz') == seal['input_sha256'], 'Archived e input bytes changed')
    require(sha(ap/'PROTOCOL.json') == seal['protocol_sha256'], 'Archived e input protocol changed')
    final_seal = read(fp/'PREDICTION_SEAL.json')
    require(final_seal['rows'] == 238, 'Complete original frozen e238 source required')
    require(sha(fp/'PREDICTIONS.jsonl.gz') == final_seal['predictions_sha256'], 'Original frozen e output bytes changed')
    corrections = actual_corrections(root, wanted)
    # Match per-frame files to the sealed whole journal; never trust a loose
    # same-ID JSON on its own.
    final_rows = {}
    with gzip.open(fp/'PREDICTIONS.jsonl.gz', 'rt') as stream:
        for line in stream:
            value = json.loads(line)
            if value['id'] in wanted:
                final_rows[value['id']] = value
    result = {}
    with gzip.open(ap/'INPUTS.jsonl.gz', 'rt') as stream:
        for line in stream:
            row = json.loads(line)
            ident = row['id']
            if ident not in wanted:
                continue
            require(ident in final_rows, ident+'/missing sealed e final')
            old = row['finish_input']['_original_source']
            baseline = row['baseline']
            saved = final_rows[ident]
            require(saved['image_sha256'] == row['image_sha256'], ident+'/image binding')
            require(saved['states'] == old['states'] == row['spec']['states'], ident+'/original states differ')
            require(saved['arms']['SAME_FROZEN_E'] == baseline, ident+'/whole e baseline archive differs')
            numeric_bridge = pose_check(baseline['actual_pose'], saved['arms']['SAME_FROZEN_E']['actual_pose'], ident+'/original e bridge')
            if expected_final_poses is not None:
                require(ident in expected_final_poses, ident+'/current E0 pose missing')
                numeric_bridge = max(numeric_bridge, pose_check(
                    baseline['actual_pose'], expected_final_poses[ident], ident+'/current E0 bridge'))
            geometry = old['initial_geometry']
            require(old['original_coarse_pose']['R_cf'] == geometry['R_cf'] and
                    old['original_coarse_pose']['centroid'] == geometry['centroid'],
                    ident+'/original coarse geometry differs')
            correction = corrections[ident]['correction']
            require(corrections[ident]['image_sha256'] == row['image_sha256'], ident+'/final correction image differs')
            by_measurement = {m['native_corner']: m for m in correction['visible_measurements']}
            for m in row['finish_input']['frozen_BI_fit']:
                require(m['native_corner'] in by_measurement, ident+'/actual final RGB measurement missing')
                observed = by_measurement[m['native_corner']]
                require(m['xy'] == observed['xy'] and m['source'] == observed['source'],
                        ident+'/final RGB correction and frozen e measurement differ')
            source_rays = []
            for rec in correction['records']:
                for ray in rec.get('rays', []):
                    source_rays.append((ray, 'actual_BI_RGB_correction_after_recovery', rec.get('native_corner')))
            # D/e uses typed partial line evidence after complete likelihood
            # revocation. Its original observed samples survive in line_record.
            for boundary in row['boundary_all']:
                for line in boundary.get('lines', []):
                    ray = line.get('provenance', {}).get('line_record')
                    if ray is not None:
                        source_rays.append((ray, 'D_e_partial_boundary_observation', boundary.get('native_corner')))
            rays, seen = [], set()
            for ray, stage, native_corner in source_rays:
                digest = canonical(ray)
                if digest in seen:
                    continue
                seen.add(digest)
                keys = ('native_corner', 'native_neighbor', 'endpoints', 'normal', 'offset',
                        'observed_points', 'observation_confidence', 'inlier_mask',
                        'supported', 'reason', 'measurement_source', 'guide_source',
                        'inlier_count', 'inlier_rms_px', 'whole_edge_search')
                rays.append(dict(**{k: deepcopy(ray[k]) for k in keys if k in ray},
                                 trace_stage=stage, trace_native_corner=native_corner,
                                 original_record_sha256=digest))
            visible = baseline.get('solution', {}).get('visible_observations', [])
            # The solver archives every original visible observation, including
            # explicitly revoked complete2D factors. Presence in that list does
            # not imply that a point remains a complete2D likelihood.
            explicit_revoked = baseline.get('solution', {}).get('boundary_factor_audit', {}).get(
                'explicit_complete_BI_revocation_ids')
            declared_revoked = {b['native_corner'] for b in row['boundary_all']
                                if b.get('explicit_complete_BI_likelihood_revocation') is True}
            if explicit_revoked is not None:
                require(set(explicit_revoked) == declared_revoked, ident+'/complete likelihood revocation archive differs')
            revoked = declared_revoked if explicit_revoked is None else set(explicit_revoked)
            complete = {m['native_corner'] for m in visible if isinstance(m, dict) and 'native_corner' in m} - revoked
            measurements = []
            for m in row['finish_input']['frozen_BI_fit']:
                value = deepcopy(m)
                value['used_as_complete_final_likelihood'] = m['native_corner'] in complete
                value['explicit_complete_2D_likelihood_revoked'] = m['native_corner'] in revoked
                value['e_likelihood_mode'] = ('partial_normal_and_tangent' if m['native_corner'] in revoked
                                              else 'complete_2D_observation')
                value['is_actual_image_observation'] = True
                value['trace_stage'] = 'frozen_BI_before_D_e_partial_revocation'
                measurements.append(value)
            mask = row.get('mask_binding')
            mask_path = None
            missing = []
            if mask is None:
                missing.append('Actual original mask binding is absent; no substitute mask generated.')
            elif not (root/mask['path']).is_file():
                missing.append('Bound original mask file is absent; no substitute mask generated.')
            else:
                require(sha(root/mask['path']) == mask['sha256'], ident+'/original mask bytes changed')
                mask_path = mask['path']
            solution = baseline.get('solution', {})
            hierarchy = baseline.get('hierarchical_correspondence_audit', {})
            selector = dict(accepted=baseline.get('accepted'), WD_family=baseline.get('WD_family'),
                method=baseline.get('method'), fallback_audit=baseline.get('fallback_audit'),
                hierarchy=deepcopy(hierarchy), objective=solution.get('objective'),
                status=solution.get('status'), rank=deepcopy(solution.get('rank')),
                ambiguity=deepcopy(baseline.get('ambiguity_audit')),
                saved_candidate_count=len(solution.get('candidates', [])),
                saved_feasible_count=len(solution.get('feasible_pose_samples', [])),
                complete_2D_likelihood_revocation_ids=sorted(revoked),
                initial_N3_pose_is_final_e=False)
            raw = deepcopy(old.get('original_raw_points'))
            if raw is None:
                missing.append('Original N3/SubPix native points absent; no reconstruction from final pose.')
            regions = dict(OT=[compact_region(d) for d in row['spec']['domains']],
                           U=[compact_region(d) for d in row['u_regions']],
                           partial_BI=deepcopy(row['boundary_all']))
            result[ident] = dict(id=ident, image_sha256=row['image_sha256'],
                initial_points=raw, initial_pose=deepcopy(old['original_coarse_pose']),
                states=deepcopy(old['states']), initial_roles=deepcopy(geometry['initial_roles']),
                state_records=deepcopy(old['state_observation']['records']),
                initial_self_hidden=deepcopy(geometry['initial_self_hidden']),
                K=deepcopy(geometry['K']), native_xyz=deepcopy(geometry['native_xyz']),
                raw_hw=deepcopy(geometry['image_hw']), measurements=measurements,
                final_complete_measurements=deepcopy([m for m in visible if m['native_corner'] in complete]),
                archived_visible_observations=deepcopy(visible), rays=rays, regions=regions,
                mask_path_or_none=mask_path,
                final_pose=deepcopy(baseline['actual_pose']), selected_candidate_audit=selector,
                provenance=dict(archive_path=str(ARCHIVE/'INPUTS.jsonl.gz'),
                    archive_sha256=seal['input_sha256'], input_row_sha256=canonical(row),
                    source_field='finish_input._original_source',
                    initial_pose_source='original archived N3/SubPix common-F coarse pose, before e',
                    initial_points_source='original_raw_points from the same pre-e observation',
                    actual_correction_binding=deepcopy(corrections[ident]['binding']),
                    actual_correction_matches_all_frozen_e_BI_xy_and_sources=True,
                    e_source=str(FINAL/'PREDICTIONS.jsonl.gz'), e_arm='SAME_FROZEN_E',
                    original_e_pose_numeric_bridge_max_abs=numeric_bridge,
                    current_E0_bridge_checked=expected_final_poses is not None,
                    mask_binding=deepcopy(mask),
                    state_mask_and_BI_are_original_e_inputs=True,
                    archived_after_e_but_source_observation_was_copied_unchanged=True,
                    post_e_E_FROZEN_reobservations_used=False,
                    S_input_points_are_not_inferred_or_filled=True,
                    missing_stages=missing,
                    no_new_NN_PnP_line_search_or_pixel_query=True))
    require(set(result) == wanted, 'Requested original e trace is incomplete: '+str(sorted(wanted-set(result))))
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root', type=Path, required=True)
    p.add_argument('--ids-file', type=Path, required=True)
    p.add_argument('--expected-e-inputs', type=Path)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    doc = read(a.ids_file)
    ids = doc if isinstance(doc, list) else doc['cohorts']['ALL5_IQR_T_R']['excluded_ids']
    expected = None
    if a.expected_e_inputs:
        expected = {f['id']: f['arms']['E_FROZEN']['coarse_pose'] for f in read(a.expected_e_inputs)['frames']}
    require(not a.output.exists(), 'Preserve previous e trace outputs')
    traces = load_e_trace(a.source_root, ids, expected_final_poses=expected)
    with a.output.open('x') as stream:
        json.dump(traces, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(rows=len(traces), masks=sum(v['mask_path_or_none'] is not None for v in traces.values()),
        missing=sum(bool(v['provenance']['missing_stages']) for v in traces.values()),
        e_numeric_bridge_max_abs=max(v['provenance']['original_e_pose_numeric_bridge_max_abs'] for v in traces.values()),
        new_NN_PnP_line_or_pixel_calls=0)))
