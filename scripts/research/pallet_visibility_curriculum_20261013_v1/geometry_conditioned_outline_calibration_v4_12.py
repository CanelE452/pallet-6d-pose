"""CAL-only outline mean grouped by known unordered cuboid dimensions.

This candidate changes only the signed mean in three controlled runtime
profiles. All profiles use the same frozen shared-global total normal scatter.
Group scatter is reported diagnostically, never substituted silently. A group
may contain one CAL session; that explicitly cannot validate independent
generalization. No image, semantic corner coordinate, TEST label or pose fit
is accessed by this module.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import shutil

import numpy as np

from . import common as C
from . import statewise_outline_noise_calibration_v4_10 as GLOBAL

OUTPUT = GLOBAL.OUTPUT.parent / 'CAL_geometry_outline_noise_v4_12'
MODEL_SCHEMA = 'CAL_GEOMETRY_OUTLINE_NOISE_MODEL_V4_12'
PROFILES = ('NO_BIAS', 'SHARED_MEAN', 'GEOMETRY_MEAN')
RECIPE = dict(
    schema='CAL_GEOMETRY_OUTLINE_NOISE_RECIPE_V4_12',
    population='all original CAL46, including native-incompatible; unsupported masks retained in inventory',
    source='existing CAL_FRAME_MASK_RESIDUALS; no new reference, RGB query or pose fit',
    sign='observed mask outside reference silhouette positive; inside negative',
    normalization='unchanged observed frozen-mask hull bounding-box diagonal',
    geometry_key='sorted positive known cf_extents, each formatted to 12 significant decimal digits',
    no_material_session_frame_or_native_compatibility_runtime_routing=True,
    minimum_group_supported_frames=10, minimum_group_CAL_sessions=1,
    single_session_group_independent_validation_claimed=False,
    mean_estimator='median of per-frame median signed normalized distances',
    diagnostic_group_scatter='1.4826 times frame-equal weighted MAD around group mean',
    shared_global_scatter='unchanged v4.10 CAL global scatter in every runtime profile',
    profiles=list(PROFILES),
    missing_or_insufficient_geometry_group='shared-global mean fallback',
    LOGO='exclude held-out session from group and shared-global means and scatter; missing group falls back global',
    correlated_boundary_samples_not_independent=True,
    reference_derived_not_independent_physical_truth=True,
    physical_probability_guarantee=False, future_geometry_coverage_guarantee=False,
    TEST_files_opened=False, image_reads=0, S_coordinate_reads=0, pose_fits=0,
    adoption='candidate only; no default or parameter choice made here')


def geometry_signature(known_extents):
    """WH-swap invariant key from registered size only, never pose or labels."""
    values = np.asarray(known_extents, dtype=float)
    if values.shape != (3,) or not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError('Three finite positive known cuboid dimensions required')
    return '|'.join(format(float(x), '.12g') for x in np.sort(values))


def _validate_records(records, geometry_by_id):
    records = list(records)
    # Partition is checked before any fitted residual coordinate is inspected.
    if any(r.get('partition') != 'CAL' for r in records):
        raise ValueError('CAL only; TEST records forbidden')
    ids = [r['id'] for r in records]
    if len(ids) != len(set(ids)) or set(ids) != set(geometry_by_id):
        raise ValueError('Unique exact CAL geometry/record ID match required')
    keys = {identity: geometry_signature(geometry_by_id[identity]) for identity in ids}
    for record in records:
        if not isinstance(record.get('session'), str) or not record['session']:
            raise ValueError('Explicit CAL session required')
        if not isinstance(record.get('supported'), bool):
            raise ValueError('Explicit mask support required')
        gap = float(record.get('unordered_registered_model_body_max_matching_gap_m', 0.))
        if not np.isfinite(gap) or gap > 1e-6 or gap < 0:
            raise ValueError('Known registered unordered model/reference shape mismatch')
        if not record['supported']:
            continue
        diagonal = float(record['mask_diagonal_px'])
        values = np.asarray(record['signed_distances_normalized'], dtype=float)
        if not np.isfinite(diagonal) or diagonal <= 0 or values.ndim != 1 or not len(values) or not np.isfinite(values).all():
            raise ValueError('Finite supported CAL signed residuals and mask diagonal required')
        if 'signed_distances_px' in record:
            pixels = np.asarray(record['signed_distances_px'], dtype=float)
            if pixels.shape != values.shape or not np.isfinite(pixels).all() or not np.allclose(pixels / diagonal, values, rtol=1e-12, atol=1e-12):
                raise ValueError('Frozen normalized/pixel signed residual mismatch')
    return records, keys


def _stats(records):
    bias = float(np.median([np.median(r['signed_distances_normalized']) for r in records]))
    errors, weights = [], []
    for record in records:
        residual = np.abs(np.asarray(record['signed_distances_normalized']) - bias)
        errors.extend(residual)
        weights.extend([1. / (len(records) * len(residual))] * len(residual))
    sigma = max(float(1.4826 * GLOBAL.OLD._quantile(errors, weights, .5)), GLOBAL.RECIPE['minimum_normalized_sigma'])
    return bias, sigma


def fit_geometry_outline_model(records, geometry_by_id, *, heldout_session=None, input_bindings=None):
    """Pure CAL fitter; group means and shared scatter exclude held-out data."""
    records, keys = _validate_records(records, geometry_by_id)
    if heldout_session is not None and heldout_session not in {r['session'] for r in records}:
        raise ValueError('Held-out session must belong to supplied CAL population')
    # The immutable shared fitter keeps its >=3 frames / >=2 sessions contract.
    global_model = GLOBAL.fit_outline_noise_model(records, heldout_session=heldout_session,
                                                  input_bindings=deepcopy(input_bindings or {}))
    train = [r for r in records if r['supported'] and r['session'] != heldout_session]
    groups = {}
    for key in sorted(set(keys.values())):
        members = [r for r in train if keys[r['id']] == key]
        sessions = sorted({r['session'] for r in members})
        available = len(members) >= RECIPE['minimum_group_supported_frames'] and len(sessions) >= RECIPE['minimum_group_CAL_sessions']
        group = dict(geometry_signature=key, known_sorted_extents=[float(x) for x in key.split('|')],
                     supported_train_frame_count=len(members), fit_ids=[r['id'] for r in members], fit_sessions=sessions,
                     group_mean_available=available, single_session_group=len(sessions) == 1,
                     independently_held_session_validation_available=len(sessions) > 1,
                     fallback_reason=None if available else 'fewer_than_10_supported_CAL_frames_or_missing_geometry',
                     group_scatter_runtime_used=False)
        if available:
            bias, sigma = _stats(members)
            group.update(bias_ratio=bias, diagnostic_total_normal_sigma_ratio=sigma)
        groups[key] = group
    model = dict(schema=MODEL_SCHEMA, fit_partition='CAL', recipe=deepcopy(RECIPE), heldout_session=heldout_session,
                 shared_global_model=global_model, geometry_groups=groups,
                 fit_ids=global_model['fit_ids'], fit_sessions=global_model['fit_sessions'],
                 fit_frame_count=global_model['fit_frame_count'], CAL_population_ids=[r['id'] for r in records],
                 unsupported_CAL_ids=global_model['unsupported_CAL_ids'],
                 excluded_heldout_ids=global_model['excluded_heldout_ids'],
                 runtime_profiles_share_identical_global_sigma=True, geometry_group_sigma_is_diagnostic_only=True,
                 input_bindings=deepcopy(input_bindings or {}))
    model['model_sha256'] = GLOBAL._digest(model)
    return model


def geometry_outline_parameters(model, known_extents, mask_diagonal_px, *, profile='GEOMETRY_MEAN'):
    """Pure runtime exporter, sharing sigma and changing only the chosen mean."""
    if model.get('schema') != MODEL_SCHEMA or model.get('fit_partition') != 'CAL':
        raise ValueError('CAL geometry outline model required')
    if GLOBAL._digest({k: v for k, v in model.items() if k != 'model_sha256'}) != model.get('model_sha256'):
        raise ValueError('Geometry model digest changed')
    if profile not in PROFILES:
        raise ValueError('Unknown fixed outline mean profile')
    global_parameters = GLOBAL.outline_noise_parameters(model['shared_global_model'], mask_diagonal_px)
    key = geometry_signature(known_extents)
    group = model['geometry_groups'].get(key)
    available = group is not None and group['group_mean_available']
    bias_ratio = 0. if profile == 'NO_BIAS' else model['shared_global_model']['bias_ratio']
    if profile == 'GEOMETRY_MEAN' and available:
        bias_ratio = group['bias_ratio']
    return dict(outward_mask_bias_px=float(mask_diagonal_px) * bias_ratio,
                total_normal_sigma_px=global_parameters['total_normal_sigma_px'],
                model_sha256=model['model_sha256'], shared_sigma_model_sha256=global_parameters['model_sha256'],
                profile=profile, geometry_signature=key, geometry_group_available=available,
                mean_source='zero' if profile == 'NO_BIAS' else 'geometry_group' if profile == 'GEOMETRY_MEAN' and available else 'shared_global',
                group_mean_fallback=profile == 'GEOMETRY_MEAN' and not available,
                fallback_reason=None if available else 'unknown_geometry' if group is None else group['fallback_reason'],
                single_session_geometry_group=bool(group and group['single_session_group']),
                runtime_sigma_source='shared_global_for_all_three_profiles',
                group_sigma_runtime_used=False, bias_applies_to='observed-minus-model signed residual',
                per_pixel_independence_claim=False, calibrated_physical_probability=False,
                image_reads=0, S_coordinate_reads=0, GT_runtime_reads=0)


def _evaluation(records, geometry_by_id, model):
    rows = []
    for record in records:
        if not record['supported']:
            rows.append(dict(id=record['id'], session=record['session'], supported=False, reason=record.get('reason')))
            continue
        values = np.asarray(record['signed_distances_px'])
        profiles = {}
        for profile in PROFILES:
            parameters = geometry_outline_parameters(model, geometry_by_id[record['id']], record['mask_diagonal_px'], profile=profile)
            centered = values - parameters['outward_mask_bias_px']
            profiles[profile] = dict(parameters=parameters, median_signed_px=float(np.median(centered)),
                                     mean_abs_px=float(np.mean(np.abs(centered))), P90_abs_px=float(np.quantile(np.abs(centered), .9)))
        rows.append(dict(id=record['id'], session=record['session'], supported=True, profiles=profiles))
    return rows


def _summary(rows):
    return {profile: {metric: GLOBAL._summary([r['profiles'][profile][metric] for r in rows if r['supported']])
                      for metric in ('median_signed_px', 'mean_abs_px', 'P90_abs_px')} for profile in PROFILES}


def _verify(bindings):
    for relative, digest in bindings.items():
        if C.sha(C.ROOT / relative) != digest:
            raise ValueError('Frozen CAL source/input changed: ' + relative)


def calibrate(output=OUTPUT):
    """Append-only fit from existing CAL residual journal, never reference reads."""
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('Preserve geometry CAL evidence ' + str(output))
    source_root = GLOBAL.OUTPUT
    journal = source_root / 'CAL_FRAME_MASK_RESIDUALS.jsonl.gz'
    old_seal = C.read(source_root / 'MODEL_SEAL.json')
    if C.sha(source_root / 'MODEL.json') != old_seal['model_sha256'] or C.sha(source_root / 'PROTOCOL.json') != old_seal['protocol_sha256']:
        raise ValueError('Frozen shared CAL model/protocol changed')
    _verify(old_seal['source_sha256'])
    # Do not reopen isolated labels or pixel masks from the earlier fit.
    cal_journal = GLOBAL.PARENT / 'CAL_PREDICTIONS.jsonl.gz'
    cal_seal_path = GLOBAL.PARENT / 'CAL_PREDICTION_SEAL.json'
    cal_seal = C.read(cal_seal_path)
    if C.sha(cal_journal) != cal_seal['predictions_sha256'] or cal_seal['rows'] != 46:
        raise ValueError('Exact frozen CAL46 inference journal required')
    records = list(C.rows(journal))
    rows = list(C.rows(cal_journal))
    if len(records) != 46 or len(rows) != 46 or any(r.get('partition') != 'CAL' for r in rows):
        raise ValueError('All original CAL46 only required')
    row_by_id = {r['id']: r for r in rows}
    geometry = {r['id']: r['runtime']['initial_geometry']['cf_extents'] for r in rows}
    _validate_records(records, geometry)
    for record in records:
        row = row_by_id[record['id']]
        if record['session'] != row['session'] or record['mask_binding']['image_sha256'] != row['image_sha256']:
            raise ValueError('Residual and known-geometry CAL inference binding mismatch')
        if record.get('S_or_U_local_coordinates_or_queries_read') != 0:
            raise ValueError('Existing outline evidence must have no semantic local queries')
    old_global = C.read(source_root / 'MODEL.json')
    # Pre-existing fold/global models are input evidence, not refitted targets.
    sessions = sorted({r['session'] for r in records})
    paths = [journal, source_root / 'MODEL.json', source_root / 'MODEL_SEAL.json', source_root / 'PROTOCOL.json',
             cal_journal, cal_seal_path, output.parent.parent / 'FIX_PROTOCOL_V1.json']
    paths += [source_root / 'folds' / session / 'MODEL.json' for session in sessions]
    bindings = {str(path.relative_to(C.ROOT)): C.sha(path) for path in paths}
    protected = C.read(output.parent.parent / 'FIX_PROTOCOL_V1.json')['protected_user_sha256']
    _verify(protected)
    sources = {str(Path(path).resolve().relative_to(C.ROOT)): C.sha(path)
               for path in (__file__, GLOBAL.__file__, GLOBAL.OLD.__file__, C.__file__)}
    output.mkdir(parents=True)
    C.write(output / 'PROTOCOL.json', dict(schema='CAL_GEOMETRY_OUTLINE_PROTOCOL_V4_12', recipe=RECIPE,
             CAL=46, TEST=0, source_sha256=sources, input_sha256=bindings, protected_user_sha256=protected,
             shared_global_model_sha256=C.sha(source_root / 'MODEL.json'), group_scatter_diagnostic_only=True,
             no_new_RGB_pose_or_label_reads=True, model_candidate_not_default=True))
    for relative in sources:
        dest = output / 'code_snapshot' / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(C.ROOT / relative, dest)
    model = fit_geometry_outline_model(records, geometry, input_bindings=bindings)
    for key in ('bias_ratio', 'total_normal_sigma_ratio', 'fit_ids', 'fit_sessions'):
        if model['shared_global_model'][key] != old_global[key]:
            raise ValueError('New shared model must preserve frozen global statistics exactly')
    C.write(output / 'MODEL.json', model)
    full = _evaluation(records, geometry, model)
    folds = []
    for session in sessions:
        fold = fit_geometry_outline_model(records, geometry, heldout_session=session, input_bindings=bindings)
        frozen_fold = C.read(source_root / 'folds' / session / 'MODEL.json')
        for key in ('bias_ratio', 'total_normal_sigma_ratio', 'fit_ids', 'fit_sessions'):
            if fold['shared_global_model'][key] != frozen_fold[key]:
                raise ValueError('LOGO shared statistics changed')
        heldout = [r for r in records if r['session'] == session]
        evaluation = _evaluation(heldout, geometry, fold)
        C.write(output / 'folds' / session / 'MODEL.json', fold)
        folds.append(dict(heldout_session=session, train_ids=fold['fit_ids'], heldout_ids=[r['id'] for r in heldout],
                          model=fold, rows=evaluation, summary=_summary(evaluation)))
    logo = [row for fold in folds for row in fold['rows']]
    report = dict(schema='CAL_GEOMETRY_OUTLINE_LOGO_REPORT_V4_12', CAL=46, supported=sum(r['supported'] for r in records),
                  folds=folds, full_CAL_in_sample_summary=_summary(full), CAL_LOGO_summary=_summary(logo),
                  single_wood_session_independent_geometry_validation=False, all_native_incompatible_CAL_retained=True,
                  conditional_upstream_CAL_fitted=True, TEST_files_opened=False, no_pose_accuracy_claim=True,
                  no_profile_selected_or_default_changed=True)
    C.write(output / 'CAL_LOGO_REPORT.json', report)
    C.write(output / 'FULL_CAL_FIT_DIAGNOSTICS.json', dict(rows=full, summary=_summary(full)))
    _verify({**sources, **bindings, **protected})
    files = [output / 'MODEL.json', output / 'CAL_LOGO_REPORT.json', output / 'FULL_CAL_FIT_DIAGNOSTICS.json']
    files += [output / 'folds' / session / 'MODEL.json' for session in sessions]
    C.write(output / 'MODEL_SEAL.json', dict(schema='CAL_GEOMETRY_OUTLINE_MODEL_SEAL_V4_12',
            model_sha256=C.sha(output / 'MODEL.json'), internal_model_sha256=model['model_sha256'],
            LOGO_sha256=C.sha(output / 'CAL_LOGO_REPORT.json'), protocol_sha256=C.sha(output / 'PROTOCOL.json'),
            output_sha256={str(p.relative_to(output)): C.sha(p) for p in files}, source_sha256=sources,
            input_sha256=bindings, protected_user_sha256=protected, CAL_only=True, TEST_files_opened=False,
            pose_solves=0, RGB_forward=0, S_coordinate_reads=0, runtime_group_sigma_used=False, model_adopted=False))
    print(C.finite(dict(output=str(output), supported=report['supported'], groups=model['geometry_groups'],
                       global_sigma_ratio=model['shared_global_model']['total_normal_sigma_ratio'],
                       LOGO_summary=report['CAL_LOGO_summary'])), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    calibrate(parser.parse_args().output)
