"""CAL-only total angular scale for authenticated, named RGB/native edges.

The cached v15 control and any future new observer must have separate models.
Preparation authenticates/deduplicates prediction-only lines and seals sources
before CAL reference decoding. No image, predictor or pose optimizer runs here.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
from itertools import islice
import json
from pathlib import Path
import shutil

import numpy as np

from . import common as C
from . import statewise_evaluation_v4 as E
from . import evaluate_fixed_wd_archived_tangent_v4_15 as ARCHIVE
from . import statewise_uncertainty_calibration_v4 as ALIGN
from . import hierarchical_native_candidate_score_v4_17 as AUTH
from . import projected_native_edge_factor_v4_17 as FACTOR


OUTPUT = ARCHIVE.OUTPUT.parent / 'CAL_native_edge_angle_v4_17'
LABELS = E.COHORT / 'OOF_pose_v1/ISOLATED_CAL_SCORING.json'
SCHEMA = 'CAL_NATIVE_EDGE_TOTAL_SINE_ANGLE_MODEL_V4_17'
RECIPE = dict(
    fit_partition='CAL', inventory='all original CAL46; fixed-native compatible CAL44 only',
    original_WD_geometry_only=True, reference_phase='fixed 3D ALIGN._reference_alignment; no pose-error choice',
    authentication='same caller authentication and physical-edge deduplication as runtime',
    observation_generation='immutable v15 boundary_all cached RGB lines; not a new observer',
    residual='cross(canonical observed unit normal, projected named-edge unit normal)',
    support_anchor='exact authenticated TLS inlier-support interval midpoint, never a precise KP',
    weighting='equal sessions; equal supported frames within session; equal deduplicated edges within frame',
    scale_estimator='weighted median absolute total sine error / 0.67449; zero bias',
    median_convention='smallest sorted value whose cumulative weight reaches one half',
    normal_angle_correlation=0., correlation_is_engineering_approximation=True,
    per_edge_variance='CAL sigma_sine^2 + (1 pixel / actual inlier span pixels)^2',
    conditional_pixel_floor_px=1., outliers_trimmed=False,
    model_noise_not_conditional_TLS_RMS=True, cached_search_window_saturation_not_independently_resolved=True,
    future_probability_or_coverage_guarantee=False, independent_edge_or_pixel_likelihood_claim=False,
    reference_is_independent_metrology=False, runtime_frame_or_session_routing=False,
    TEST_or_remaining_labels_fitted=False, pose_fits=0, image_or_raw_S_coordinate_reads=0)


def _digest(value):
    data = json.dumps(C.finite(value), sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(data.encode()).hexdigest()


def _number(value, name, *, positive=False):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(name + ' must be a finite scalar')
    value = float(value)
    if not np.isfinite(value) or (positive and value <= 0):
        raise ValueError(name + ' must be finite' + (' and positive' if positive else ''))
    return value


def _weighted_median(values, weights):
    values, weights = np.asarray(values, float), np.asarray(weights, float)
    order = np.argsort(values, kind='stable')
    return float(values[order[min(int(np.searchsorted(np.cumsum(weights[order]), weights.sum() * .5)), len(order)-1)]])


def fit_native_edge_angle_model(records, *, fit_ids=None, inventory_ids=None,
                                heldout_session=None, input_bindings=None):
    """Pure session/frame-equal CAL fitter; excluded rows' errors are not read."""
    records = list(records)
    if any(r.get('partition') != 'CAL' for r in records):
        raise ValueError('CAL only; TEST and remaining reference inputs are forbidden')
    ids = [r['id'] for r in records]
    if len(set(ids)) != len(ids):
        raise ValueError('Unique CAL frame IDs required')
    inventory_ids = list(ids if inventory_ids is None else inventory_ids)
    if len(set(inventory_ids)) != len(inventory_ids) or set(ids) != set(inventory_ids):
        raise ValueError('Complete declared CAL inventory required')
    allowed = set(ids if fit_ids is None else fit_ids)
    if not allowed <= set(ids):
        raise ValueError('Fit IDs must belong to the declared CAL inventory')
    train, eligible = [], []
    for row in records:
        # Structural gates precede reading any held-out/incompatible line error.
        if row['id'] not in allowed or row.get('compatible') is not True or row['session'] == heldout_session:
            continue
        eligible.append(row['id'])
        lines = []
        for edge in row['edges']:
            error = _number(edge['signed_angular_sine_error'], 'sine error')
            if abs(error) > 1. + 1e-12:
                raise ValueError('Sine error must lie in [-1,1]')
            span = _number(edge['actual_inlier_span_px'], 'actual observed inlier span', positive=True)
            lines.append(dict(error=error, span=span))
        if lines:
            train.append(dict(id=row['id'], session=row['session'], lines=lines))
    if not train:
        raise ValueError('At least one supported compatible training CAL frame required')
    sessions = sorted({r['session'] for r in train})
    frame_counts = Counter(r['session'] for r in train)
    errors, weights = [], []
    for row in train:
        weight = 1. / (len(sessions) * frame_counts[row['session']] * len(row['lines']))
        errors.extend(abs(e['error']) for e in row['lines'])
        weights.extend([weight] * len(row['lines']))
    median_abs = _weighted_median(errors, weights)
    model = dict(schema=SCHEMA, fit_partition='CAL', recipe=deepcopy(RECIPE),
        heldout_session=heldout_session, inventory_ids=inventory_ids,
        inventory_frame_count=len(records), compatible_inventory_count=sum(r.get('compatible') is True for r in records),
        compatible_training_ids=eligible, fit_ids=[r['id'] for r in train], fit_sessions=sessions,
        fit_frame_count=len(train), fit_edge_count=len(errors),
        unsupported_training_ids=[fid for fid in eligible if fid not in {r['id'] for r in train}],
        excluded_heldout_ids=[r['id'] for r in records if r['session'] == heldout_session],
        incompatible_inventory_ids=[r['id'] for r in records if r.get('compatible') is not True],
        sigma_sine=float(median_abs / .67449), weighted_median_absolute_sine=median_abs,
        session_weights={s:1./len(sessions) for s in sessions},
        fit_weight_sum=float(sum(weights)), normal_angle_correlation=0.,
        angular_bias_correction=0., input_bindings=deepcopy(input_bindings or {}),
        independent_generalization_or_probability_claim=False)
    model['model_digest_sha256'] = _digest(model)
    return model


def angular_variance_for_edge(model, actual_inlier_span_px):
    """Final per-edge variance; do not feed this floor-inclusive value to AUTH."""
    if model.get('schema') != SCHEMA or model.get('fit_partition') != 'CAL' or model.get('recipe') != RECIPE:
        raise ValueError('Declared CAL native-edge angular model required')
    if _digest({k:v for k,v in model.items() if k != 'model_digest_sha256'}) != model.get('model_digest_sha256'):
        raise ValueError('CAL angular model digest changed')
    sigma = _number(model['sigma_sine'], 'CAL total sine sigma')
    if sigma < 0:
        raise ValueError('Nonnegative CAL sine scale required')
    span = _number(actual_inlier_span_px, 'actual observed inlier span', positive=True)
    floor = (RECIPE['conditional_pixel_floor_px'] / span) ** 2
    return dict(angular_variance_rad2=sigma**2 + floor, CAL_total_sine_variance=sigma**2,
        conditional_pixel_span_variance=floor, normal_angle_correlation=0.,
        model_digest_sha256=model['model_digest_sha256'], fit_partition='CAL',
        sine_is_small_angle_scale_not_radian_coverage=True, probability_guarantee=False,
        covariance_chart='canonical observed normal; exact authenticated support anchor')


def authentication_angular_parameters(model):
    """AUTH input contains only CAL variance; AUTH adds its per-line floor once."""
    # Reuse strict digest/recipe validation; the temporary span does not enter
    # the returned component and creates no runtime observation.
    validated = angular_variance_for_edge(model, 1.)
    return dict(angular_variance_rad2=validated['CAL_total_sine_variance'],
        normal_angle_correlation=0., angular_variance_provenance=dict(
            fit_partition='CAL', model_digest_sha256=model['model_digest_sha256'],
            component='CAL total sine variance only; AUTH adds 1px/span floor once',
            observation_generation=RECIPE['observation_generation'], probability_guarantee=False))


def _verify_preparation(output):
    protocol = C.read(output/'PROTOCOL.json'); seal = C.read(output/'PREPARATION_SEAL.json')
    if C.sha(output/'PROTOCOL.json') != seal['protocol_sha256'] or C.sha(output/'CAL_AUTHENTICATED_INPUTS.jsonl.gz') != seal['CAL_inputs_sha256']:
        raise ValueError('Prepared CAL sources/inputs changed')
    for key in ('source_sha256', 'input_sha256', 'protected_user_sha256'):
        E._verify_sources(protocol[key])
    return protocol


def prepare(output=OUTPUT):
    """Prediction-only authentication and source snapshots, before label decode."""
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('Preserve every earlier calibration preparation')
    parent = C.read(ARCHIVE.OUTPUT/'PROTOCOL.json'); seal = C.read(ARCHIVE.OUTPUT/'INPUT_SEAL.json')
    journal = ARCHIVE.OUTPUT/'INPUTS.jsonl.gz'
    if seal['rows'] != 238 or C.sha(journal) != seal['input_sha256'] or C.sha(ARCHIVE.OUTPUT/'PROTOCOL.json') != seal['protocol_sha256']:
        raise ValueError('Immutable complete238 prediction-input archive required')
    for key in ('source_sha256', 'input_sha256', 'protected_user_sha256'):
        E._verify_sources(parent[key])
    # The sealed archive orders CAL46 first. Do not decode TEST/R143 rows.
    rows = list(islice(C.rows(journal), 46))
    if len(rows) != 46 or any(r['partition'] != 'CAL' for r in rows) or [r['id'] for r in rows] != seal['ordered_ids'][:46]:
        raise ValueError('Exact first CAL46 inventory required; no other partitions decoded')
    sources = dict(parent['source_sha256'])
    for p in (__file__, C.__file__, E.__file__, ALIGN.__file__, AUTH.__file__, FACTOR.__file__):
        p = Path(p); sources[str(p.relative_to(C.ROOT))] = C.sha(p)
    inputs = {str(p.relative_to(C.ROOT)):C.sha(p) for p in (journal, ARCHIVE.OUTPUT/'PROTOCOL.json', ARCHIVE.OUTPUT/'INPUT_SEAL.json', LABELS)}
    output.mkdir(parents=True)
    protocol = dict(schema='CAL_NATIVE_EDGE_ANGLE_PREFIT_PROTOCOL_V17', recipe=RECIPE,
        prepared_before_CAL_reference_decode_UTC=datetime.now(timezone.utc).isoformat(),
        ids=[r['id'] for r in rows], rows=46, source_sha256=sources, input_sha256=inputs,
        source_binding_scope='conservative archived source union plus explicitly used fitter/auth/kernel; not all union modules execute',
        protected_user_sha256=parent['protected_user_sha256'], CAL_label_path=str(LABELS.relative_to(C.ROOT)),
        TEST_or_remaining_label_paths=False, other_partition_prediction_rows_decoded=0,
        new_pose_fits=0, new_RGB_or_model_forwards=0)
    C.write(output/'PROTOCOL.json', protocol)
    for relative, digest in sources.items():
        p = output/'code_snapshot'/relative; p.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(C.ROOT/relative, p)
        if C.sha(p) != digest:
            raise ValueError('Prepared source snapshot mismatch')
    prepared = []
    for row in rows:
        family = row['stage1_family_seed_bank']['ORIGINAL_WD']['family']
        initial = row['finish_input']['_original_source']['initial_geometry']
        binding = dict(sha256=row['image_sha256'], image_hw=row['spec']['image_hw'])
        auth = AUTH.authenticate_supported_native_edges(family['native_xyz'], row['spec']['states'],
            row['spec']['visible'], row['boundary_all'], row['spec']['image_hw'],
            initial_self_hidden=initial['initial_self_hidden'], image_binding=binding,
            angular_variance_rad2=0., angular_variance_provenance=dict(fit_partition='CAL',
                purpose='geometry authentication only; placeholder not fitted angular uncertainty'),
            normal_angle_correlation=0., config=row['spec']['config'])
        prepared.append(dict(id=row['id'], partition='CAL', session=row['session'],
            native_xyz=family['native_xyz'], physical_registration_Q=family['physical_registration_Q'],
            K=row['spec']['K'], image_hw=row['spec']['image_hw'], states=row['spec']['states'],
            image_binding=binding, authenticated_edges=auth['edges'], authentication_audit=auth))
    C.save_rows(output/'CAL_AUTHENTICATED_INPUTS.jsonl.gz', prepared)
    for value in (sources, inputs, protocol['protected_user_sha256']): E._verify_sources(value)
    C.write(output/'PREPARATION_SEAL.json', dict(protocol_sha256=C.sha(output/'PROTOCOL.json'),
        CAL_inputs_sha256=C.sha(output/'CAL_AUTHENTICATED_INPUTS.jsonl.gz'), rows=46,
        source_sha256=sources, before_CAL_reference_decode=True, no_TEST_or_remaining_label_decode=True))
    print(dict(prepared_CAL=46, authenticated_edges=sum(len(r['authenticated_edges']) for r in prepared),
        before_CAL_reference_decode=True), flush=True)


def fit(output=OUTPUT):
    """Open only the declared CAL reference after the preparation seal exists."""
    output = Path(output).resolve(); protocol = _verify_preparation(output)
    if (output/'MODEL.json').exists(): raise FileExistsError('Preserve fitted CAL model')
    rows = list(C.rows(output/'CAL_AUTHENTICATED_INPUTS.jsonl.gz'))
    labels = C.read(C.ROOT/protocol['CAL_label_path'])['rows']; records = []
    for row in rows:
        label = labels[row['id']]
        if label['metadata'].get('partition') != 'CAL': raise ValueError('CAL reference metadata required')
        alignment = ALIGN._reference_alignment({'runtime':{'initial_geometry':row}}, label)
        record = dict(id=row['id'], partition='CAL', session=row['session'], compatible=bool(alignment['compatible']),
            fixed_3D_phase=int(alignment['phase']), geometry_gap_m=float(alignment['geometry_gap_m']),
            authenticated_edge_count=len(row['authenticated_edges']), edges=[])
        if alignment['compatible']:
            K=np.asarray(row['K']);xyz=np.asarray(row['native_xyz']);R=alignment['R_reference_native'];t=alignment['t_reference']
            q=C.project(xyz,R,t,K);depth=(xyz@R.T+t)[:,2]
            for edge in row['authenticated_edges']:
                first,second,observed,anchor,cov=FACTOR._parameters(edge)
                direction=q[second]-q[first];length=float(np.linalg.norm(direction))
                if not np.isfinite(q[[first,second]]).all() or np.min(depth[[first,second]]) <= 1e-8 or length <= 1e-8:
                    raise ValueError('Compatible CAL named reference edge collapsed; no silent trimming')
                normal=np.array([-direction[1],direction[0]])/length
                sine=float(observed[0]*normal[1]-observed[1]*normal[0])
                interval=edge['actual_support_tangent_interval_px'];span=float(interval[1]-interval[0])
                record['edges'].append(dict(physical_native_edge=edge['physical_native_edge'], native_corner=first,
                    native_neighbor=second, signed_angular_sine_error=sine,
                    unoriented_angle_deg=float(np.degrees(np.arcsin(np.clip(abs(sine),0.,1.)))),
                    actual_inlier_span_px=span, support_anchor_xy=anchor.tolist(), canonical_observed_normal=observed.tolist(),
                    observed_line_offset_px=edge['offset_px'],
                    canonical_observed_offset_px=float(-observed@anchor), reference_edge_length_px=length,
                    old_normal_variance_px2=edge['normal_variance_px2'],
                    actual_inlier_points_present=edge['provenance']['actual_inlier_points_present'],
                    source_edge=deepcopy(edge)))
        records.append(record)
    if len(records)!=46 or sum(r['compatible'] for r in records)!=44:
        raise ValueError('Expected all CAL46 inventory / fixed-native compatible44, without error filtering')
    model=fit_native_edge_angle_model(records,input_bindings=protocol['input_sha256'])
    folds=[]
    for session in sorted({r['session'] for r in records}):
        local=fit_native_edge_angle_model(records,heldout_session=session,input_bindings=protocol['input_sha256'])
        held=[r for r in records if r['session']==session and r['compatible']]
        ratios=[abs(e['signed_angular_sine_error'])/np.sqrt(angular_variance_for_edge(local,e['actual_inlier_span_px'])['angular_variance_rad2']) for r in held for e in r['edges']]
        folds.append(dict(heldout_session=session,model=local,heldout_ids=[r['id'] for r in held],
            heldout_supported_edge_count=len(ratios),descriptive_fraction_le2=None if not ratios else float(np.mean(np.asarray(ratios)<=2)),
            descriptive_fraction_le3=None if not ratios else float(np.mean(np.asarray(ratios)<=3)),
            independent_probability_or_native_identity_guarantee=False))
    C.write(output/'MODEL.json',model)
    C.save_rows(output/'CAL_ANGULAR_ERRORS.jsonl.gz',records)
    C.write(output/'CAL_LOGO_REPORT.json',dict(inventory_frame_count=46,compatible_frame_count=44,
        supported_fit_frames=model['fit_frame_count'],deduplicated_fit_edges=model['fit_edge_count'],
        LOGO_folds=folds,no_error_or_geometry_size_trimming=True,no_TEST_or_remaining_labels_read=True,
        new_observer_model_not_substitutable_without_new_fit_and_seal=True))
    _verify_preparation(output)
    C.write(output/'MODEL_SEAL.json',dict(model_sha256=C.sha(output/'MODEL.json'),
        CAL_errors_sha256=C.sha(output/'CAL_ANGULAR_ERRORS.jsonl.gz'),LOGO_sha256=C.sha(output/'CAL_LOGO_REPORT.json'),
        preparation_seal_sha256=C.sha(output/'PREPARATION_SEAL.json'),source_sha256=protocol['source_sha256'],
        input_sha256=protocol['input_sha256'],fit_partition='CAL',CAL_inventory=46,compatible_inventory=44,
        no_new_pose_or_image_fit=True,no_default_adoption=True))
    print(dict(sigma_sine=model['sigma_sine'],CAL46=46,compatible44=44,
        supported_frames=model['fit_frame_count'],fit_edges=model['fit_edge_count'],LOGO=len(folds)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('prepare','fit'))
    parser.add_argument('--output',type=Path,default=OUTPUT)
    args=parser.parse_args()
    (prepare if args.mode=='prepare' else fit)(args.output)
