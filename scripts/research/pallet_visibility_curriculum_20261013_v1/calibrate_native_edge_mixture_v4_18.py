"""CAL-only total normal/sine noise for all actual-image NMS alternatives.

This is a separate model for the v17c image-lattice observer.  Old cached
window-line calibration is never substituted.  The pose kernel subtracts a
fixed peak for convenience; noise fitting deliberately uses the uncentered
mixture pseudo-likelihood, including covariance volume.  No image forward,
keypoint correction or pose optimization is performed here.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from itertools import islice
from pathlib import Path
import shutil
import sys

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

from . import common as C
from . import statewise_evaluation_v4 as E
from . import evaluate_fixed_wd_archived_tangent_v4_15 as ARCHIVE
from . import statewise_uncertainty_calibration_v4 as ALIGN
from . import calibrate_native_edge_angle_v4_17 as OLD_CAL
from . import native_edge_problem_v4_17 as N
from . import image_chain_mixture_problem_v4_18 as MIX
from . import projected_native_edge_factor_v4_17 as FACTOR


OBSERVATIONS = ARCHIVE.OUTPUT.parent / 'native_edge_observations_v4_17b'
OUTPUT = ARCHIVE.OUTPUT.parent / 'CAL_native_edge_mixture_v4_18'
LABELS = OLD_CAL.LABELS
SCHEMA = 'CAL_ACTUAL_NMS_NATIVE_EDGE_MIXTURE_NOISE_V4_18'
RECIPE = dict(
    fit_partition='CAL', inventory='all original CAL46; fixed-native compatible44 only',
    reference_phase='fixed geometry-only ALIGN._reference_alignment; no pose-error choice',
    observation_generation='v17c actual image-lattice NMS chains in sealed v17b CAL archive',
    applicability='ORIGINAL boundary_all old-partial physical edges only; same runtime authentication',
    group='one physical native edge; deduplicate identical actual inlier pixels',
    component_prior='uniform over every authenticated actual-pixel alternative; no best-RMS/GT selection',
    support_anchor='actual unweighted inlier centroid projected onto saved weighted-TLS line; same runtime gauge',
    residual='normal offset at support anchor and unoriented angular sine',
    normal_floor='fn=max(1 pixel, actual inlier RMS); variance=fn^2',
    angular_floor='2*fn^2/(actual inlier span^2+2*fn^2)',
    normal_angle_correlation=0., correlation_is_engineering_approximation=True,
    total_covariance='diag(CAL sigma_normal^2+normal_floor, CAL sigma_sine^2+angular_floor)',
    bias_correction=0., scale_components_nonnegative=True,
    weighting='equal supported sessions, equal supported frames within session, equal physical edges within frame',
    estimator='uncentered radial-pseudo-Huber mixture NLL with covariance log volume',
    runtime_peak_centering_used_for_scale_fit=False, huber_scale=3.,
    optimizer='L-BFGS-B on nonnegative scaled variance components',
    optimizer_scaled_variance_starts=[[0., 0.], [1., 1.], [4., 4.]],
    optimizer_maximum_iterations=1000, optimizer_ftol=1e-12, optimizer_gtol=1e-8,
    optimizer_global_minimum_guaranteed=False, error_or_geometry_size_trimming=False,
    sample_count_precision_boost=False, old_reliability_used_as_extra_calibration_weight=False,
    independent_component_or_pixel_likelihood_claim=False,
    native_identity_or_future_probability_guarantee=False,
    reference_is_independent_metrology=False,
    TEST_or_remaining_labels_fitted=False, runtime_frame_or_session_routing=False,
    pose_fits=0, image_or_raw_S_coordinate_reads=0)


def _digest(value):
    return OLD_CAL._digest(value)


def _finite(value, name, *, positive=False):
    value = OLD_CAL._number(value, name, positive=positive)
    return value


def conditional_covariance_floors(actual_inlier_span_px, inlier_rms_px):
    """Exactly the v18 runtime floors, with no precision gain from count."""
    span = _finite(actual_inlier_span_px, 'actual inlier span', positive=True)
    rms = _finite(inlier_rms_px, 'actual inlier RMS')
    if rms < 0:
        raise ValueError('Nonnegative actual inlier RMS required')
    fn = max(1., rms)
    return np.array([fn*fn, 2.*fn*fn/(span*span+2.*fn*fn)])


def _training_groups(records, *, fit_ids=None, inventory_ids=None, heldout_session=None):
    records = list(records)
    if any(r.get('partition') != 'CAL' for r in records):
        raise ValueError('CAL only; TEST and remaining reference inputs are forbidden')
    ids = [r['id'] for r in records]
    inventory = list(ids if inventory_ids is None else inventory_ids)
    if len(set(ids)) != len(ids) or len(set(inventory)) != len(inventory) or set(ids) != set(inventory):
        raise ValueError('Complete unique declared CAL inventory required')
    allowed = set(ids if fit_ids is None else fit_ids)
    if not allowed <= set(ids):
        raise ValueError('Fit IDs must belong to the CAL inventory')
    train, eligible = [], []
    for row in records:
        # Exclusion is structural and precedes reading any reference error.
        if row['id'] not in allowed or row.get('compatible') is not True or row['session'] == heldout_session:
            continue
        eligible.append(row['id']); groups = []
        for group in row['groups']:
            components = []
            for component in group['components']:
                error = np.array([_finite(component['normal_error_px'], 'normal error'),
                    _finite(component['angular_sine_error'], 'angular sine error')])
                if abs(error[1]) > 1.+1e-12:
                    raise ValueError('Angular sine must lie in [-1,1]')
                floor = conditional_covariance_floors(component['actual_inlier_span_px'], component['normal_floor_px'])
                supplied = np.asarray(component['conditional_variance_floor'], float)
                if supplied.shape != (2,) or not np.allclose(floor, supplied, atol=1e-14, rtol=1e-12):
                    raise ValueError('CAL/runtime conditional covariance floors differ')
                components.append(dict(error=error, floor=floor))
            if not components:
                raise ValueError('An applicable physical edge requires all nonempty alternatives')
            groups.append(components)
        if groups:
            train.append(dict(id=row['id'], session=row['session'], groups=groups))
    if not train:
        raise ValueError('At least one applicable compatible CAL training frame required')
    sessions = sorted({r['session'] for r in train}); counts = Counter(r['session'] for r in train)
    groups, weights = [], []
    for row in train:
        weight = 1./(len(sessions)*counts[row['session']]*len(row['groups']))
        groups.extend(row['groups']); weights.extend([weight]*len(row['groups']))
    return records, inventory, eligible, train, sessions, groups, np.asarray(weights)


def mixture_noise_nll_and_gradient(variance_components, groups, weights, *, huber_scale=3.):
    """Uncentered -2 log mixture density; covariance volume is essential.

    The common two-dimensional robust density normalizer is independent of
    covariance and omitted.  In particular, the pose kernel's covariance-
    dependent peak subtraction must not be included in this estimator.
    """
    variance = np.asarray(variance_components, float)
    weights = np.asarray(weights, float); delta = float(huber_scale)
    if (variance.shape != (2,) or not np.isfinite(variance).all() or np.any(variance < 0)
            or weights.shape != (len(groups),) or not np.isfinite(weights).all()
            or np.any(weights <= 0) or not np.isfinite(delta) or delta <= 0):
        raise ValueError('Nonnegative variance components and positive finite group weights required')
    total, gradient = 0., np.zeros(2)
    for components, weight in zip(groups, weights):
        errors = np.asarray([v['error'] for v in components], float)
        floors = np.asarray([v['floor'] for v in components], float)
        cov = variance + floors
        if not len(components) or np.any(cov <= 0) or not np.isfinite(cov).all():
            raise ValueError('Finite positive total component covariance required')
        squared = errors*errors/cov
        denominator = np.sqrt(1.+squared.sum(axis=1)/(delta*delta))
        rho = 2.*delta*delta*(denominator-1.)
        logits = -.5*np.log(cov).sum(axis=1)-.5*rho-np.log(len(components))
        log_density = float(logsumexp(logits))
        posterior = np.exp(logits-log_density)
        derivative = -.5/cov + .5*errors*errors/(cov*cov*denominator[:, None])
        total += -2.*weight*log_density
        gradient += -2.*weight*(posterior@derivative)
    return float(total), gradient


def fit_native_edge_mixture_model(records, *, fit_ids=None, inventory_ids=None,
                                  heldout_session=None, input_bindings=None):
    records, inventory, eligible, train, sessions, groups, weights = _training_groups(
        records, fit_ids=fit_ids, inventory_ids=inventory_ids, heldout_session=heldout_session)
    # These training-only scales condition the two numerical variance axes;
    # they do not choose a mixture component or an observation threshold.
    component_errors, component_floors, component_weights = [], [], []
    for group, weight in zip(groups, weights):
        component_errors.extend(np.abs([v['error'] for v in group]))
        component_floors.extend([v['floor'] for v in group])
        component_weights.extend([weight/len(group)]*len(group))
    errors, floors = np.asarray(component_errors), np.asarray(component_floors)
    scales = np.array([max(OLD_CAL._weighted_median(errors[:, j], component_weights)/.67449,
        np.sqrt(OLD_CAL._weighted_median(floors[:, j], component_weights)))**2 for j in range(2)])
    def objective(scaled):
        value, gradient = mixture_noise_nll_and_gradient(np.asarray(scaled)*scales, groups, weights,
            huber_scale=RECIPE['huber_scale'])
        return value, gradient*scales
    searches = []
    for start in RECIPE['optimizer_scaled_variance_starts']:
        result = minimize(objective, np.array(start), method='L-BFGS-B', jac=True,
            bounds=[(0., None), (0., None)], options=dict(maxiter=RECIPE['optimizer_maximum_iterations'],
            ftol=RECIPE['optimizer_ftol'], gtol=RECIPE['optimizer_gtol']))
        variance = np.maximum(np.asarray(result.x)*scales, 0.)
        value, gradient = mixture_noise_nll_and_gradient(variance, groups, weights,
            huber_scale=RECIPE['huber_scale'])
        if not np.isfinite([*variance, value, *gradient]).all():
            raise ValueError('No nonfinite noise-fit result may be selected')
        searches.append(dict(scaled_start=start, variance_components=variance.tolist(),
            training_mixture_nll=value, success=bool(result.success), message=str(result.message),
            iterations=int(result.nit), gradient_wrt_variance=gradient.tolist()))
    winner = min(range(len(searches)), key=lambda j:(searches[j]['training_mixture_nll'], j))
    variance = np.asarray(searches[winner]['variance_components'])
    model = dict(schema=SCHEMA, fit_partition='CAL', recipe=deepcopy(RECIPE),
        heldout_session=heldout_session, inventory_ids=inventory, inventory_frame_count=len(records),
        compatible_inventory_count=sum(r.get('compatible') is True for r in records),
        compatible_training_ids=eligible, fit_ids=[r['id'] for r in train], fit_sessions=sessions,
        fit_frame_count=len(train), fit_physical_edge_count=len(groups),
        fit_alternative_count=sum(len(g) for g in groups),
        unsupported_training_ids=[fid for fid in eligible if fid not in {r['id'] for r in train}],
        excluded_heldout_ids=[r['id'] for r in records if r['session'] == heldout_session],
        incompatible_inventory_ids=[r['id'] for r in records if r.get('compatible') is not True],
        sigma_normal_px=float(np.sqrt(variance[0])), sigma_sine=float(np.sqrt(variance[1])),
        normal_variance_component_px2=float(variance[0]), angular_variance_component_sine2=float(variance[1]),
        normal_angle_correlation=0., normal_bias_correction_px=0., angular_bias_correction=0.,
        fit_weight_sum=float(weights.sum()), session_weights={s:1./len(sessions) for s in sessions},
        optimizer_conditioning_variance_scales=scales.tolist(), optimizer_searches=searches,
        selected_optimizer_start_index=winner, training_mixture_nll=searches[winner]['training_mixture_nll'],
        input_bindings=deepcopy(input_bindings or {}),
        independent_generalization_or_probability_claim=False, model_is_for_new_image_NMS_only=True)
    model['model_digest_sha256'] = _digest(model)
    return model


def mixture_noise_parameters(model):
    """Runtime exports CAL variance components; the factory adds floors once."""
    if model.get('schema') != SCHEMA or model.get('fit_partition') != 'CAL' or model.get('recipe') != RECIPE:
        raise ValueError('Separate declared CAL actual-NMS mixture model required')
    if _digest({k:v for k,v in model.items() if k != 'model_digest_sha256'}) != model.get('model_digest_sha256'):
        raise ValueError('CAL mixture model digest changed')
    vn = _finite(model['normal_variance_component_px2'], 'CAL normal variance')
    va = _finite(model['angular_variance_component_sine2'], 'CAL sine variance')
    if min(vn, va) < 0:
        raise ValueError('Nonnegative CAL variance components required')
    return dict(normal_variance_component_px2=vn, angular_variance_component_sine2=va,
        normal_angle_correlation=0., noise_provenance=dict(fit_partition='CAL',
            model_digest_sha256=model['model_digest_sha256'], observation_generation=RECIPE['observation_generation'],
            observation_family=MIX.NOISE_OBSERVATION_FAMILY,
            component='CAL total variance components only; runtime adds exact per-chain floors once',
            probability_or_native_identity_guarantee=False))


def _verify_preparation(output):
    protocol = C.read(output/'PROTOCOL.json'); seal = C.read(output/'PREPARATION_SEAL.json')
    if C.sha(output/'PROTOCOL.json') != seal['protocol_sha256'] or C.sha(output/'CAL_AUTHENTICATED_INPUTS.jsonl.gz') != seal['CAL_inputs_sha256']:
        raise ValueError('Prepared CAL source/input seal changed')
    for key in ('source_sha256', 'input_sha256', 'protected_user_sha256'):
        E._verify_sources(protocol[key])
    return protocol


def _observation_inventory(observation):
    pairs, alternatives = set(), 0
    for ray in observation['edge_records']:
        # S/grazing skip records intentionally have no point/chain fields.
        if ray.get('supported') is not True:
            continue
        chains = ray.get('chains', ())
        if chains:
            pairs.add(tuple(sorted((ray['native_corner'], ray['native_neighbor']))))
            alternatives += len(chains)
    return len(pairs), alternatives


def prepare(output=OUTPUT):
    """Seal CAL inputs and actual authentication before any reference decode."""
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('Preserve every earlier calibration preparation')
    obs_protocol = C.read(OBSERVATIONS/'PROTOCOL.json')
    obs_seal = C.read(OBSERVATIONS/'CAL_OBSERVATION_SEAL.json')
    obs_journal = OBSERVATIONS/'CAL_OBSERVATIONS.jsonl.gz'
    if (obs_seal['rows'] != 46 or C.sha(obs_journal) != obs_seal['observations_sha256']
            or C.sha(OBSERVATIONS/'PROTOCOL.json') != obs_seal['protocol_sha256']):
        raise ValueError('Sealed actual-NMS CAL46 observations required')
    parent = C.read(ARCHIVE.OUTPUT/'PROTOCOL.json'); parent_seal = C.read(ARCHIVE.OUTPUT/'INPUT_SEAL.json')
    old_journal = ARCHIVE.OUTPUT/'INPUTS.jsonl.gz'
    if C.sha(old_journal) != parent_seal['input_sha256'] or C.sha(ARCHIVE.OUTPUT/'PROTOCOL.json') != parent_seal['protocol_sha256']:
        raise ValueError('Frozen old partial-factor archive changed')
    for source in (obs_protocol, parent):
        for key in ('source_sha256', 'input_sha256', 'protected_user_sha256'):
            E._verify_sources(source[key])
    old_rows = list(islice(C.rows(old_journal), 46)); rows = list(C.rows(obs_journal))
    ids = [r['id'] for r in rows]
    if (len(rows) != 46 or any(r['partition'] != 'CAL' for r in rows+old_rows)
            or ids != obs_seal['ordered_ids'] or ids != [r['id'] for r in old_rows]
            or ids != parent_seal['ordered_ids'][:46]):
        raise ValueError('Exact same complete CAL46 inventory required; no TEST/R rows decoded')
    sources = {**parent['source_sha256'], **obs_protocol['source_sha256']}
    for module in (sys.modules[__name__], C, E, ARCHIVE, ALIGN, OLD_CAL, N, MIX, MIX.M, FACTOR):
        path = Path(module.__file__).resolve(); sources[str(path.relative_to(C.ROOT))] = C.sha(path)
    paths = (old_journal, ARCHIVE.OUTPUT/'PROTOCOL.json', ARCHIVE.OUTPUT/'INPUT_SEAL.json',
        obs_journal, OBSERVATIONS/'PROTOCOL.json', OBSERVATIONS/'CAL_OBSERVATION_SEAL.json', LABELS)
    inputs = {str(path.relative_to(C.ROOT)):C.sha(path) for path in paths}
    output.mkdir(parents=True)
    protocol = dict(schema='CAL_ACTUAL_NMS_MIXTURE_NOISE_PREFIT_PROTOCOL_V18', recipe=RECIPE,
        prepared_before_CAL_reference_decode_UTC=datetime.now(timezone.utc).isoformat(),
        ids=ids, rows=46, source_sha256=sources, input_sha256=inputs,
        protected_user_sha256=parent['protected_user_sha256'], CAL_label_path=str(LABELS.relative_to(C.ROOT)),
        source_binding_scope='conservative archived union plus actual auth/fitter; not all union files execute',
        other_partition_prediction_rows_decoded=0, TEST_or_remaining_label_paths=False,
        new_pose_fits=0, new_RGB_or_model_forwards=0)
    C.write(output/'PROTOCOL.json', protocol)
    for relative, digest in sources.items():
        target = output/'code_snapshot'/relative; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(C.ROOT/relative, target)
        if C.sha(target) != digest:
            raise ValueError('Source snapshot changed')
    prepared = []
    for old, row in zip(old_rows, rows):
        if (row['states'] != old['spec']['states'] or row['visible_measurements'] != old['spec']['visible']
                or row['domains'] != old['spec']['domains'] or row['u_regions'] != old['u_regions']
                or row['initial_geometry'] != old['finish_input']['_original_source']['initial_geometry']):
            raise ValueError('Frozen production geometry/q/cov/state/regions changed')
        spec = old['spec']; family = old['stage1_family_seed_bank']['ORIGINAL_WD']['family']
        cfg = {**spec['config'], 'prior_weight':0.}
        coarse = deepcopy(family['coarse_pose'])
        # Family field names are bound explicitly rather than inferred from GT.
        base = N.prepare_native_edge_problem(family['native_xyz'], spec['K'], spec['states'], spec['visible'],
            spec['domains'], spec['image_hw'], coarse, config=cfg,
            boundary_observations=old['boundary_all'], u_regions=old['u_regions'],
            edge_bundle=dict(edges=[], replaces_normal_factor_refs=[]))
        binding = row['chain_observation']['image_binding']
        authenticated = MIX.authenticate_image_chain_groups(base, row['chain_observation'],
            initial_self_hidden=row['initial_geometry']['initial_self_hidden'], image_binding=binding,
            normal_variance_component_px2=0., angular_variance_component_sine2=0.,
            noise_provenance=dict(fit_partition='CAL', observation_family=MIX.NOISE_OBSERVATION_FAMILY,
                purpose='prefit authentication; no fitted CAL noise yet'),
            normal_angle_correlation=0.)
        physical_count, alternative_count = _observation_inventory(row['chain_observation'])
        prepared.append(dict(id=row['id'], partition='CAL', session=row['session'],
            initial_geometry=deepcopy(row['initial_geometry']),
            all_NMS_physical_edges=physical_count, all_NMS_alternatives=alternative_count,
            authenticated_group_bundle=authenticated))
    C.save_rows(output/'CAL_AUTHENTICATED_INPUTS.jsonl.gz', prepared)
    _verify_maps = (sources, inputs, protocol['protected_user_sha256'])
    for value in _verify_maps:
        E._verify_sources(value)
    C.write(output/'PREPARATION_SEAL.json', dict(protocol_sha256=C.sha(output/'PROTOCOL.json'),
        CAL_inputs_sha256=C.sha(output/'CAL_AUTHENTICATED_INPUTS.jsonl.gz'), rows=46,
        source_sha256=sources, before_CAL_reference_decode=True, no_TEST_or_remaining_label_decode=True,
        all_NMS_edge_count=sum(r['all_NMS_physical_edges'] for r in prepared),
        runtime_applicable_group_count=sum(len(r['authenticated_group_bundle']['groups']) for r in prepared)))
    print(dict(prepared_CAL=46, applicable_groups=sum(len(r['authenticated_group_bundle']['groups']) for r in prepared),
        before_CAL_reference_decode=True), flush=True)


def fit(output=OUTPUT):
    output = Path(output).resolve(); protocol = _verify_preparation(output)
    if (output/'MODEL.json').exists():
        raise FileExistsError('Preserve fitted CAL model')
    prepared = list(C.rows(output/'CAL_AUTHENTICATED_INPUTS.jsonl.gz'))
    labels = C.read(C.ROOT/protocol['CAL_label_path'])['rows']; records = []
    for row in prepared:
        label = labels[row['id']]
        if label['metadata'].get('partition') != 'CAL':
            raise ValueError('CAL reference metadata required')
        alignment = ALIGN._reference_alignment({'runtime':{'initial_geometry':row['initial_geometry']}}, label)
        record = dict(id=row['id'], partition='CAL', session=row['session'], compatible=bool(alignment['compatible']),
            fixed_3D_phase=int(alignment['phase']), geometry_gap_m=float(alignment['geometry_gap_m']),
            all_NMS_physical_edges=row['all_NMS_physical_edges'], all_NMS_alternatives=row['all_NMS_alternatives'],
            runtime_applicable_group_count=len(row['authenticated_group_bundle']['groups']), groups=[])
        if alignment['compatible']:
            geometry = row['initial_geometry']; xyz = np.asarray(geometry['native_xyz']); K = np.asarray(geometry['K'])
            R, t = alignment['R_reference_native'], alignment['t_reference']
            scale = max(float(np.linalg.norm(np.ptp(xyz, axis=0))), 1e-3)
            p = N.V._pose_vector(R, t, scale)
            for group in row['authenticated_group_bundle']['groups']:
                components = []
                for edge in group['components']:
                    _, _, audit = FACTOR.projected_edge_residual_and_jacobian(p, xyz, K, scale, edge)
                    floor = conditional_covariance_floors(edge['actual_inlier_span_px'], edge['normal_floor_px'])
                    supplied = [edge['normal_variance_px2'], edge['angular_variance_rad2']]
                    if not np.allclose(floor, supplied, atol=1e-14, rtol=1e-12):
                        raise ValueError('Authenticated runtime floors differ from fitted floors')
                    components.append(dict(normal_error_px=audit['normal_residual_px'],
                        angular_sine_error=audit['angular_sine_residual'],
                        actual_inlier_span_px=edge['actual_inlier_span_px'], normal_floor_px=edge['normal_floor_px'],
                        conditional_variance_floor=floor.tolist(), source_component=deepcopy(edge)))
                record['groups'].append(dict(physical_native_edge=[group['native_corner'],group['native_neighbor']],
                    components=components, old_reliability_preserved_but_not_extra_CAL_weight=group['reliability_weight']))
        records.append(record)
    if len(records) != 46 or sum(r['compatible'] for r in records) != 44:
        raise ValueError('Expected exact CAL46 / fixed-native compatible44 inventory')
    model = fit_native_edge_mixture_model(records, input_bindings=protocol['input_sha256'])
    folds = []
    for session in sorted({r['session'] for r in records}):
        local = fit_native_edge_mixture_model(records, heldout_session=session, input_bindings=protocol['input_sha256'])
        held = [r for r in records if r['session'] == session and r['compatible'] and r['groups']]
        held_groups = [[dict(error=np.array([c['normal_error_px'], c['angular_sine_error']]),
            floor=np.asarray(c['conditional_variance_floor'])) for c in g['components']] for r in held for g in r['groups']]
        held_nll = None
        if held_groups:
            held_nll = mixture_noise_nll_and_gradient([local['normal_variance_component_px2'],
                local['angular_variance_component_sine2']], held_groups, np.full(len(held_groups),1./len(held_groups)))[0]
        folds.append(dict(heldout_session=session, model=local,
            heldout_ids=[r['id'] for r in records if r['session'] == session],
            heldout_supported_frames=len(held), heldout_applicable_edge_count=len(held_groups),
            descriptive_heldout_equal_edge_mixture_nll=held_nll,
            independent_probability_or_native_identity_guarantee=False))
    C.write(output/'MODEL.json', model)
    C.save_rows(output/'CAL_MIXTURE_ERRORS.jsonl.gz', records)
    C.write(output/'CAL_LOGO_REPORT.json', dict(inventory_frame_count=46, compatible_frame_count=44,
        all_NMS_physical_edges=sum(r['all_NMS_physical_edges'] for r in records),
        all_NMS_alternatives=sum(r['all_NMS_alternatives'] for r in records), LOGO_folds=folds,
        supported_fit_frames=model['fit_frame_count'], applicable_fit_edges=model['fit_physical_edge_count'],
        fit_alternatives=model['fit_alternative_count'], no_error_or_geometry_size_trimming=True,
        no_TEST_or_remaining_labels_read=True, old_cached_window_line_model_used=False,
        calibrated_pose_probability_or_default_adoption=False))
    _verify_preparation(output)
    C.write(output/'MODEL_SEAL.json', dict(model_sha256=C.sha(output/'MODEL.json'),
        CAL_errors_sha256=C.sha(output/'CAL_MIXTURE_ERRORS.jsonl.gz'), LOGO_sha256=C.sha(output/'CAL_LOGO_REPORT.json'),
        preparation_seal_sha256=C.sha(output/'PREPARATION_SEAL.json'), source_sha256=protocol['source_sha256'],
        input_sha256=protocol['input_sha256'], fit_partition='CAL', CAL_inventory=46, compatible_inventory=44,
        new_pose_or_image_fits=0, default_adoption=False))
    print(dict(sigma_normal_px=model['sigma_normal_px'], sigma_sine=model['sigma_sine'],
        fit_frames=model['fit_frame_count'], physical_edges=model['fit_physical_edge_count'],
        alternatives=model['fit_alternative_count'], LOGO=len(folds)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'fit'))
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    (prepare if args.mode == 'prepare' else fit)(args.output)
