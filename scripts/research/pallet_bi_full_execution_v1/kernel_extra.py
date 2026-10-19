"""Completed-action union scored on each matched full-normal core's own packet.

No pose solver, model fit, image reader or evaluation-reference reader is called.
Original endpoints and scalar features remain explicit inputs. The public
five-argument union_frame returns a feature row; batch_features additionally
archives individual new costs and all visited extra endpoints.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from pathlib import Path
import time

import cv2
import numpy as np

from . import s1_union as U
from . import fullnormal_backend as B
from . import fullnormal_batch as CORE
from . import qcov as Q
from . import extra_sources as SOURCES
from scripts.research.pallet_statewise_end_to_end_recovery_v1 import solver as S
from scripts.research.pallet_boundary_action_repair_v1 import action_data as D

ACTIVE_MODES = ('fixed', 'A', 'AB')
ABSENT_INDICES = (2, 4, 28, 30)
CONTEXT_SCHEMA = 'FULL_NORMAL_OWN_PACKET_FEATURE_CONTEXT_V1'
PARALLEL_SCHEMAS = ('BI_FULL_NORMAL_PARALLEL_EXECUTION_V1', 'FULL_NORMAL_ORDERED_DISPATCH_V1',
                    'FULL_NORMAL_FOUR_CPU_DISPATCH_V1')
PF_METHODS = frozenset(('POSEFIX_SUBPIX_PF_ANCHOR', 'POSEFIX_SUBPIX_YOLO_ANCHOR'))


def extra_inputs_for_frames(ids, extra_rows_path=None, extra_seal_path=None,
                            extra_methods=None, *, extra_inputs=None,
                            extra_inputs_manifest=None):
    """Verify ordered independent archives, preserving every selected raw row.

    New manifests contain {extra_inputs:[{path,seal,methods?},...]}. Paths may
    be relative to the manifest. List order is significant; PF archives must
    precede material/render archives. The legacy single-path API is unchanged.
    Every parent must represent every input frame, including unavailable rows.
    """
    if len(set(ids)) != len(ids):
        raise ValueError('Unique ordered core frame IDs required')
    multiple = extra_inputs is not None or extra_inputs_manifest is not None
    manifest_binding = None
    if multiple:
        if extra_rows_path is not None or extra_seal_path is not None or extra_methods is not None:
            raise ValueError('Use either legacy single source or explicit ordered extra inputs')
        if extra_inputs is not None and extra_inputs_manifest is not None:
            raise ValueError('Use one multi-input declaration mechanism')
        if extra_inputs_manifest is not None:
            manifest_path = Path(extra_inputs_manifest).resolve()
            value = U.read(manifest_path)
            if not isinstance(value, dict) or set(value) != {'extra_inputs'}:
                raise ValueError('Manifest requires only the ordered extra_inputs list')
            extra_inputs = value['extra_inputs']; base = manifest_path.parent
            manifest_binding = dict(path=str(manifest_path), sha256=U.sha(manifest_path))
        else:
            base = Path.cwd()
        if not isinstance(extra_inputs, list) or not extra_inputs:
            raise ValueError('At least one explicit independent extra input required')
        specs = []
        for item in extra_inputs:
            if not isinstance(item, dict) or set(item) - {'path', 'seal', 'methods'} or not item.get('path') or not item.get('seal'):
                raise ValueError('Each extra input requires path/seal and optional methods')
            path, seal = (Path(item[key]) for key in ('path', 'seal'))
            specs.append(dict(path=str((base/path).resolve()), seal=str((base/seal).resolve()), methods=item.get('methods')))
    else:
        if extra_rows_path is None:
            raise ValueError('An extra archive or ordered extra-input declaration is required')
        specs = [dict(path=str(extra_rows_path), seal=extra_seal_path, methods=extra_methods)]
    extras = {fid: [] for fid in ids}; frame_parents = {fid: [] for fid in ids}
    bindings = []; pf_parents = []; other_parents = []
    for index, item in enumerate(specs):
        methods = item['methods']
        if methods is not None:
            if not isinstance(methods, (list, tuple, set)) or not methods or any(not isinstance(m, str) or not m for m in methods):
                raise ValueError('Nonempty explicit method names required')
            methods = set(methods)
        verified = SOURCES.verify(item['path'], item['seal'])
        selected = {fid: [] for fid in ids}; actual_methods = set()
        for row in U.rows(item['path']):
            if methods is not None and row['method'] not in methods:
                continue
            if row['id'] not in selected:
                raise ValueError('Extra frame is absent from exact core inputs')
            selected[row['id']].append(row); actual_methods.add(row['method'])
        if any(not values for values in selected.values()):
            raise ValueError('Every independent parent must explicitly cover every core frame, including unavailable actions')
        if actual_methods & PF_METHODS: pf_parents.append(index)
        if actual_methods - PF_METHODS: other_parents.append(index)
        binding = dict(input_index=index, **verified,
            admitted_methods=sorted(methods) if methods is not None else None,
            actual_selected_methods=sorted(actual_methods))
        bindings.append(binding)
        for fid, values in selected.items():
            # Values remain the exact decoded producer payloads. Provenance
            # lives beside them rather than changing the parent row objects.
            extras[fid].extend(values)
            frame_parents[fid].append(dict(input_index=index,
                parent_binding_sha256=U.payload_sha(binding),
                selected_rows_payload_sha256=U.payload_sha(values), selected_rows=len(values)))
    if multiple and pf_parents and other_parents and max(pf_parents) >= min(other_parents):
        raise ValueError('PF archives must precede separate material/render archives')
    for fid in ids:
        cids = [row.get('candidate_id', row['method']) for row in extras[fid]]
        if len(cids) != len(set(cids)):
            raise ValueError('Independent extra archives contain colliding candidate IDs')
    composite = dict(schema='ORDERED_INDEPENDENT_COMPLETE_ACTION_INPUTS_V1',
        ordered_inputs=bindings, declaration_manifest=manifest_binding)
    return extras, frame_parents, composite, multiple


def parallel_lock(core_lock):
    """Recognize three explicit schedules and authenticate their row helper."""
    if core_lock.get('schema') not in PARALLEL_SCHEMAS:
        return False
    from . import fullnormal_parallel as PARALLEL
    helper_sha = U.sha(PARALLEL.__file__)
    if core_lock.get('driver_sha256') != helper_sha:
        raise ValueError('Parallel producer driver/row-helper source differs')
    if core_lock['schema'] in ('FULL_NORMAL_ORDERED_DISPATCH_V1', 'FULL_NORMAL_FOUR_CPU_DISPATCH_V1'):
        from . import fullnormal_dispatch as DISPATCH
        if core_lock.get('row_helper_sha256') != helper_sha:
            raise ValueError('Ordered dispatch row-helper source differs')
        if core_lock['schema'] == 'FULL_NORMAL_ORDERED_DISPATCH_V1':
            if core_lock.get('orchestration_sha256') != U.sha(DISPATCH.__file__):
                raise ValueError('Ordered dispatch orchestration source differs')
        else:
            from . import fullnormal_dispatch_four as FOUR
            d3_sha = U.sha(DISPATCH.__file__)
            if core_lock.get('orchestration_sha256') != U.sha(FOUR.__file__):
                raise ValueError('Four-worker dispatch orchestration source differs')
            if core_lock.get('preserved_D3_validation_sha256') != d3_sha or FOUR.D3_EXPECTED_SHA256 != d3_sha:
                raise ValueError('Four-worker preserved D3 validation source differs')
            resource = core_lock.get('resource_policy', {})
            if any(type(value) is not int or value != expected for value, expected in (
                    (core_lock.get('workers'), 4), (core_lock.get('dispatch_window'), 4),
                    (resource.get('CPU_workers'), 4), (resource.get('BLAS_threads_per_worker'), 1))):
                raise ValueError('Four-worker dispatch requires workers/window/CPU=4 and BLAS=1')
    return True


class _BoundFeature(dict):
    """An in-process receipt issued only after validating a sealed core row."""

    def __init__(self, feature, binding):
        super().__init__(feature)
        self.binding = binding


def source_binding():
    result = dict(CORE.source_binding())
    for module in (U, D, SOURCES):
        result[str(Path(module.__file__).resolve())] = U.sha(module.__file__)
    result[str(Path(__file__).resolve())] = U.sha(__file__)
    return result


def _context(frame, result, feature, config):
    return dict(schema=CONTEXT_SCHEMA, id=frame['id'],
        packet_sha256=U.payload_sha(result['packet']),
        families_sha256=U.payload_sha(result['families']),
        feature_payload_sha256=U.payload_sha(dict(feature)),
        feature_configuration_sha256=U.payload_sha(config),
        core_feature_sources=CORE.source_binding())


def bound_core_feature(frame, result, feature, config, core_lock, frame_seal,
                       result_path, feature_path, *, context_payload=None):
    """Validate immutable core provenance before allowing old cost reuse.

    The core lock binds the feature configuration and original source closure;
    its row seal binds both actual RESULT and FEATURE bytes. The source closure
    is checked now, rather than accepting a new adapter's claimed old receipt.
    """
    parallel = parallel_lock(core_lock)
    sources = core_lock.get('numerical_kernel_source_binding') if parallel else core_lock.get('source_binding')
    if sources != CORE.source_binding():
        raise ValueError('Core feature source closure differs')
    if parallel:
        if context_payload is None or context_payload.get('feature_configuration') != config or context_payload.get('source') != sources:
            raise ValueError('Parallel core requires its exact feature-configuration cache payload')
        if Q.P.digest(context_payload) != frame_seal['cache_payload_sha256']:
            raise ValueError('Parallel core exact old feature context digest differs')
        expected_frame = {key: frame[key] for key in ('K', 'native_xyz', 'raw_hw', 'initial_pose', 'image_sha256')}
        if context_payload.get('frame') != expected_frame:
            raise ValueError('Parallel core cached frame/camera differs')
    elif core_lock.get('feature_configuration') != config:
        raise ValueError('Core feature configuration differs')
    if not frame_seal.get('complete') or frame_seal['id'] != frame['id']:
        raise ValueError('Incomplete or mismatched core frame seal')
    for path in (Path(result_path), Path(feature_path)):
        if frame_seal['artifacts'].get(path.name) != U.sha(path):
            raise ValueError('Core artifact seal differs: ' + str(path))
    if Q.read(result_path) != result or Q.read(feature_path) != feature:
        raise ValueError('Core decoded artifact differs')
    if frame_seal['source_binding'] != sources:
        raise ValueError('Core row source closure differs from producer lock')
    if not parallel and not core_lock.get('new_matched_full_normal_backend'):
        raise ValueError('Matched full-normal core required')
    return _BoundFeature(feature, _context(frame, result, feature, config))


def _parallel_inputs(core_lock, inputs, domain, prior_run):
    """Authenticate a parallel producer's immutable packet/mask context once."""
    from . import fullnormal_parallel as PARALLEL
    from scripts.research.pallet_visibility_curriculum_20261013_v1 import observe_remainder_rgb_v4_14 as LEGACY
    if not parallel_lock(core_lock):
        raise ValueError('Explicit recognized parallel core schema required')
    if prior_run is None:
        raise ValueError('Parallel own-kernel reuse requires --prior-run for the original P0 mask binding')
    p0_path = Path(prior_run)/'P0_INPUT_CONTRACT.json'
    if U.sha(p0_path) != core_lock['prior_P0_sha256']:
        raise ValueError('Parallel original P0 input contract differs')
    if core_lock['driver_sha256'] != U.sha(PARALLEL.__file__):
        raise ValueError('Parallel producer driver source differs')
    ordered = core_lock['static_ordered_row_ids']
    if ordered[domain] != [frame['id'] for frame in inputs]:
        raise ValueError('Parallel core full ordered denominator differs')
    packet_seal = Path(core_lock['packet_seal'])
    if U.sha(packet_seal) != core_lock['packet_seal_sha256']:
        raise ValueError('Parallel requested packet manifest differs')
    expected, receipt = PARALLEL.verify_packet_bindings(core_lock['packet_root'], packet_seal,
        {key: [None]*len(values) for key, values in ordered.items()})
    # Only the original solver-configuration JSON is read. No calibration
    # model, image, native observation or numerical evaluation reference opens.
    original_config = B.configuration(U.read(LEGACY.LEGACY_PROTOCOL)['solver_configuration'])
    binding = dict(P0_path=str(p0_path), P0_sha256=U.sha(p0_path),
        legacy_solver_protocol_sha256=U.sha(LEGACY.LEGACY_PROTOCOL),
        requested_packet_manifest_sha256=U.sha(packet_seal), packet_validation=receipt,
        driver_source_sha256=U.sha(PARALLEL.__file__))
    return U.read(p0_path), original_config, expected, binding


def _parallel_context(frame, result, config, core_lock, domain, ordinal, core_arm,
                      p0, original_config, expected_packets):
    packet_path = Path(core_lock['packet_root'])/'packets'/domain/f'{ordinal:04d}.json.gz'
    if U.sha(packet_path) != expected_packets[(domain, ordinal)]:
        raise ValueError('Parallel requested row packet changed')
    packet = Q.read(packet_path)
    if packet['id'] != frame['id']:
        raise ValueError('Parallel requested row packet ID differs')
    inventory = p0[domain]['inventory'][ordinal]
    mask_path = Path(inventory['observation_folder'])/'MASK.npz'
    spec = result['packet']['solver_input']
    observation = dict(_original_source=dict(ot_domains=dict(domains=spec['domains'])),
        u_region_observation=dict(u_regions=spec['u_regions']))
    return CORE.cache_payload(frame, packet['selections'][core_arm], observation,
        U.sha(mask_path), original_config, config)


def _valid(pose, family):
    try:
        return U.pose_valid(pose, family)
    except (KeyError, TypeError, ValueError, OverflowError):
        return False, 'NONFINITE_OR_SHAPE'


def _raw(candidate):
    return dict(candidate_id=candidate['candidate_id'], source='COMPLETED_ACTION',
        R_physical=deepcopy(candidate['actual_pose']['R_physical']),
        centroid=deepcopy(candidate['actual_pose']['centroid']))


def _mask_absent(candidate):
    for index in ABSENT_INDICES:
        candidate['features'][index] = 0.
        candidate['features'][index + 31] = 0.


def _base_structure(frame, result, old_feature, models):
    """Check ID, actual old bank, all visited aliases and absent-cost contract."""
    D.validate_frame(old_feature)
    if frame['id'] != result['id'] or frame['id'] != old_feature['id'] or result['packet']['id'] != frame['id']:
        raise ValueError('Core/frame identity differs')
    spec = result['packet']['solver_input']
    if spec['K'] != frame['K'] or spec['xyz'] != frame['native_xyz'] or spec['image_hw'] != frame['raw_hw']:
        raise ValueError('Actual core packet camera/native geometry/image extent differs')
    if old_feature['fallback'] != frame['initial_pose'] or result['fallback'] != frame['initial_pose']:
        raise ValueError('Core/frame native fallback differs')
    if old_feature['e0_regenerated'] != result['e0_regenerated']:
        raise ValueError('Core independently executed E0 differs')
    if old_feature.get('feature_names') != list(S.FEATURE_NAMES):
        raise ValueError('Original 31-feature contract required')
    families = {name: value['family'] for name, value in result['packet']['original_e_stage1_family_seed_bank'].items()}
    if families != result['families']:
        raise ValueError('Actual packet and result native families differ')
    scale = max(model.ctx.scale for model in models.values())
    if old_feature['object_scale_m'] != scale:
        raise ValueError('Core native object scale differs')
    old = old_feature['candidates']
    for candidate in old:
        valid, _ = _valid(candidate['actual_pose'], families['ORIGINAL_WD'])
        if not valid or candidate['feature_context_family'] not in models:
            raise ValueError('Old feature has invalid pose or context')
        if any(candidate['features'][i] != 0. or candidate['features'][i + 31] != 0. for i in ABSENT_INDICES):
            raise ValueError('Normal core B/edge features must be explicitly absent')
    # Every proper positive endpoint must still have its original source alias.
    # Invalid endpoints remain in the immutable RESULT journal, not the H1 bank.
    for endpoint in result['candidates']:
        pose = dict(available=True, R_physical=endpoint['R_physical'], centroid=endpoint['centroid'])
        if not _valid(pose, families['ORIGINAL_WD'])[0]:
            continue
        match = next((c for c in old if D.same_pose(c['actual_pose'], pose, scale)), None)
        if match is None or endpoint['candidate_id'] not in match.get('source_aliases', []):
            raise ValueError('Old endpoint or alias missing from core feature bank')
    for candidate in old:
        sources = [e for e in result['candidates'] if D.same_pose(candidate['actual_pose'],
            dict(available=True, R_physical=e['R_physical'], centroid=e['centroid']), scale)]
        if not sources:
            raise ValueError('Old feature pose is absent from actual core endpoints')
        baseline = {e['source'] for e in sources if e['source'] in ('N0', 'E0')}
        if set(candidate.get('source_baseline_aliases', [])) != baseline:
            raise ValueError('Old baseline aliases differ from actual core endpoints')
    return scale


def _fresh_audit(frame, result, old_feature, models):
    """Bare in-memory calls prove old numerical context by a full fresh audit.

    Production calls instead use bound_core_feature's exact source/config/byte
    receipt. A fresh audit is reported as such and counts every old cost call.
    """
    bank = [_raw(c) for c in old_feature['candidates']]
    scores = [S.candidate_scores(c, models) for c in bank]
    inp = dict(frame, packet=result['packet'], families=result['families'], fallback=frame['initial_pose'])
    fresh = S.features_for_bank(inp, models, bank, scores)
    max_error = 0.
    for original, rebuilt in zip(old_feature['candidates'], fresh):
        _mask_absent(rebuilt)
        if original['feature_context_family'] != rebuilt['feature_context_family']:
            raise ValueError('Old selected feature context differs from own packet')
        a, b = np.asarray(original['features']), np.asarray(rebuilt['features'])
        if not np.array_equal(a[31:], b[31:]):
            raise ValueError('Old feature validity differs from own packet')
        error = float(np.max(np.abs(a - b)))
        if error > U.FEATURE_ATOL:
            raise ValueError('Old features differ from own packet/configuration')
        max_error = max(max_error, error)
    return scores, max_error


def _minima(scores, models, modes=ACTIVE_MODES):
    values = {}
    for mode in modes:
        finite = [s[family]['scores'][mode] for s in scores for family in models
            if S._finite_feature(s[family]['scores'][mode]) is not None]
        values[mode] = min(finite) if finite else None
    return values


def _public_scores(scores):
    saved = deepcopy(scores)
    for score in saved:
        for family in score.values():
            family['available_modes'] = {mode: mode in ACTIVE_MODES for mode in U.MODES}
            for mode in ('B', 'edge'):
                family['scores'][mode] = None
                family['feasible'][mode] = None
            family['missing_modes_reason'] = 'Normal-only packet has no B profile or edge profile; AB is A'
    return U.finite(saved)


def _union(frame, result, old_feature, extra_rows, old_config):
    models = B.compile_models(result['packet'], old_config)
    scale = _base_structure(frame, result, old_feature, models)
    bound = isinstance(old_feature, _BoundFeature) and old_feature.binding == _context(frame, result, old_feature, old_config)
    old_scores, parity_error = (None, None) if bound else _fresh_audit(frame, result, old_feature, models)
    old_cost_calls = 0 if bound else len(old_feature['candidates'])
    old = deepcopy(old_feature['candidates'])
    old_ids = {c['candidate_id'] for c in old}
    additions, rejected, duplicates, seen, methods = [], [], [], set(), set()
    for row in extra_rows:
        if row['id'] != frame['id']:
            raise ValueError('Extra action frame ID differs')
        method = row['method']; cid = row.get('candidate_id', method)
        if not isinstance(cid, str) or not cid or cid in seen or cid in old_ids or cid in ('N0', 'E0', 'E0_REGENERATED'):
            raise ValueError('Unique non-baseline extra candidate IDs required')
        seen.add(cid); methods.add(method)
        if row.get('image_sha256') not in (None, frame['image_sha256']):
            raise ValueError('Extra/core RGB identity differs')
        if 'K' in row and row['K'] != frame['K']:
            raise ValueError('Extra/core camera differs')
        pose = row.get('actual_pose')
        if not row.get('producer_available', bool(pose and pose.get('available'))):
            valid, reason = False, 'PRODUCER_UNAVAILABLE'
        else:
            valid, reason = _valid(pose, result['families']['ORIGINAL_WD'])
            if reason == 'ORIGINAL_F_UNAVAILABLE':
                reason = 'PRODUCER_UNAVAILABLE'
        if not valid:
            rejected.append(dict(candidate_id=cid, method=method, reason=reason)); continue
        match = next((c for c in old + additions if D.same_pose(c['actual_pose'], pose, scale)), None)
        if match is not None:
            match['source_aliases'] = sorted(set(match.get('source_aliases', [])) | {cid, method})
            match['complete_action_aliases'] = sorted(set(match.get('complete_action_aliases', [])) | {cid})
            match['complete_action_stage_aliases'] = sorted(set(match.get('complete_action_stage_aliases', [])) | {method})
            # The independent estimator is a valid complete action even when
            # an identical old endpoint lacked a joint-constraint certificate.
            match['eligible'] = True
            duplicates.append(dict(extra_candidate_id=cid, method=method, candidate_id=match['candidate_id']))
        else:
            additions.append(dict(candidate_id=cid, actual_pose=deepcopy(pose), source_aliases=sorted({cid, method}),
                source_baseline_aliases=[], complete_action_aliases=[cid], complete_action_stage_aliases=[method],
                eligible=True, object_scale_m=scale))
    recovered, spread = U.recover_minima(old)
    minima = {m: recovered[m] for m in ACTIVE_MODES}
    missing = [m for m in ACTIVE_MODES if minima[m] is None]
    reconstruction_calls = 0
    if additions and missing:
        if old_scores is None:
            old_scores = [S.candidate_scores(_raw(c), models) for c in old]
            old_cost_calls += len(old); reconstruction_calls = len(old)
        fresh_min = _minima(old_scores, models)
        for mode in missing:
            minima[mode] = fresh_min[mode]
    new_bank = [_raw(c) for c in additions]
    new_scores = [S.candidate_scores(c, models) for c in new_bank]
    new_min = _minima(new_scores, models)
    union_min = {m: min(v for v in (minima[m], new_min[m]) if v is not None)
        if any(v is not None for v in (minima[m], new_min[m])) else None for m in ACTIVE_MODES}
    for c in old:
        for mode in ACTIVE_MODES:
            j = U.MODES.index(mode)
            if c['features'][57 + j] and minima[mode] is not None and union_min[mode] is not None and union_min[mode] < minima[mode]:
                c['features'][26 + j] = U.encode(U.decode(c['features'][26 + j]) + minima[mode] - union_min[mode])
    inp = dict(frame, packet=result['packet'], families=result['families'], fallback=frame['initial_pose'])
    new_features = S.features_for_bank(inp, models, new_bank, new_scores)
    for source, candidate, score in zip(additions, new_features, new_scores):
        candidate.update({k: deepcopy(v) for k, v in source.items()})
        candidate['admission_rule'] = 'COMMON_COMPLETE_ACTION_VALID'
        family = candidate['feature_context_family']
        for mode in ACTIVE_MODES:
            j = U.MODES.index(mode); cost = score[family]['scores'][mode]
            valid = union_min[mode] is not None and S._finite_feature(cost) is not None
            candidate['features'][26 + j] = U.encode(cost - union_min[mode]) if valid else 0.
            candidate['features'][57 + j] = float(valid)
        _mask_absent(candidate)
    features = deepcopy(dict(old_feature)); features['candidates'] = old + new_features
    features['kernel_extra'] = dict(schema='OWN_FULL_NORMAL_PACKET_COMPLETE_ACTION_UNION_V1',
        actual_core_packet_sha256=U.payload_sha(result['packet']), actual_core_result_payload_sha256=U.payload_sha(result),
        old_feature_context='sealed exact source/configuration/artifact binding' if bound else 'fresh full old-bank numerical parity audit',
        old_context_binding=deepcopy(old_feature.binding) if bound else None,
        fresh_old_feature_max_error=parity_error, old_candidate_count=len(old), new_unique_complete_actions=len(additions),
        visited_extra_endpoints=len(extra_rows), complete_action_methods=sorted(methods),
        complete_action_duplicates=duplicates, complete_action_rejections=rejected,
        old_minima=minima, new_complete_action_minima=new_min, union_minima=union_min,
        old_minima_inverse_transform_spread={m: spread[m] for m in ACTIVE_MODES},
        old_individual_cost_evaluations=old_cost_calls, old_cost_reconstructions_for_missing_context=reconstruction_calls,
        missing_minima_recovered=missing if additions else [], new_individual_cost_evaluations=len(new_scores),
        old_individual_feature_scalars_bitwise_reused=26, old_all31_validity_masks_bitwise_reused=True,
        active_bank_relative_gaps_recomputed=list(ACTIVE_MODES), absent_energy_and_gap_indices=list(ABSENT_INDICES),
        AB_reduces_to_A=True, strict_numerical_pose_deduplication=True, physical_C2_or_90_degree_collapse=False,
        all_old_endpoints_preserved_by_actual_core_result=True,
        normalization_and_188D_bank_summaries='Recompute original action_data.inputs for each arm with its own train-only normalizer',
        new_pose_optimizer_calls=0, current_real_reference_reads=0, numerical_feature_atol=U.FEATURE_ATOL)
    D.validate_frame(features)
    costs = dict(new_complete_action_bank=new_bank, new_complete_action_scores=_public_scores(new_scores),
        old_minima=minima, union_minima=union_min, explicitly_unavailable_modes=['B', 'edge'],
        actual_core_packet_sha256=U.payload_sha(result['packet']))
    return features, costs


def union_frame(frame, result, old_feature, extra_rows, old_config):
    """Return features for the old core and all valid extra complete actions.

    Each extra row has id, method, actual_pose and optional candidate_id,
    producer_available, image_sha256, K. A PF row without candidate_id uses its
    unique method. Unavailable visited proposals are archived, never admitted.
    """
    return _union(frame, result, old_feature, extra_rows, old_config)[0]


def batch_features(inputs_path, core_root, core_arm, domain, extra_rows_path,
                   config_path, output, run, *, root_GO=False, extra_seal_path=None,
                   extra_methods=None, prior_run=None, extra_inputs=None,
                   extra_inputs_manifest=None):
    """Seal one arm/domain HEAD_FEATURES + FEATURE_SEAL for heads.prepare.

    core_root/core_arm/domain/0000 contains the core RESULT.json.gz,
    FEATURE.json.gz and FRAME_SEAL.json; core_root/PRODUCER_LOCK.json binds
    their feature configuration/source. No dataset work without explicit GO.
    """
    if root_GO is not True:
        raise PermissionError('Explicit own-kernel extra feature production GO required')
    U.check_plan(run); cv2.setNumThreads(1)
    started = time.monotonic(); cpu = time.process_time()
    inputs = list(U.rows(inputs_path)); ids = [f['id'] for f in inputs]
    if len(set(ids)) != len(ids) or domain not in ('real', 'synthetic'):
        raise ValueError('Unique ordered IDs and real/synthetic domain required')
    extras, frame_parents, composite, multiple = extra_inputs_for_frames(ids,
        extra_rows_path, extra_seal_path, extra_methods, extra_inputs=extra_inputs,
        extra_inputs_manifest=extra_inputs_manifest)
    extra_binding = composite if multiple else {key: value for key, value in composite['ordered_inputs'][0].items()
        if key not in ('input_index', 'admitted_methods', 'actual_selected_methods')}
    methods = set(extra_methods) if extra_methods is not None else None
    config = U.read(config_path); core_root = Path(core_root); output = Path(output)
    lock_path = core_root/'PRODUCER_LOCK.json'; lock = U.read(lock_path)
    parallel = parallel_lock(lock)
    parallel_state = _parallel_inputs(lock, inputs, domain, prior_run) if parallel else None
    closure = source_binding()
    U.write(output/'SOURCE_DEPENDENCY_CLOSURE.json', closure)
    dependencies = dict(inputs_sha256=U.sha(inputs_path), extra_rows_sha256=U.sha(extra_rows_path) if not multiple else None,
        configuration_sha256=U.sha(config_path), core_lock_sha256=U.sha(lock_path),
        core_arm=core_arm, domain=domain, source_dependency_closure_sha256=U.payload_sha(closure),
        extra_producer_binding=extra_binding, admitted_extra_methods=sorted(methods) if methods is not None else None,
        parallel_context_binding=parallel_state[3] if parallel else None,
        plan_sha256=U.sha(Path(run)/'CAMPAIGN_PLAN.json'))
    if multiple:
        dependencies['ordered_extra_inputs_sha256'] = U.payload_sha(composite)
    U.write(output/'UNION_LOCK.json', dependencies)
    records = []; row_seals = []; calls = Counter()
    for ordinal, frame in enumerate(inputs):
        folder = core_root/core_arm/domain/f'{ordinal:04d}'
        rp, fp, sp = (folder/name for name in ('RESULT.json.gz', 'FEATURE.json.gz', 'FRAME_SEAL.json'))
        result = Q.read(rp); old = Q.read(fp); core_seal = U.read(sp)
        context_payload = _parallel_context(frame, result, config, lock, domain, ordinal, core_arm,
            *parallel_state[:3]) if parallel else None
        bound = bound_core_feature(frame, result, old, config, lock, core_seal, rp, fp, context_payload=context_payload)
        out = output/'frames'/f'{ordinal:04d}'; seal_path = out/'SEAL.json'
        key = dict(frame_payload_sha256=U.payload_sha(frame), core_frame_seal_sha256=U.sha(sp),
            result_sha256=U.sha(rp), old_feature_sha256=U.sha(fp), extra_payload_sha256=U.payload_sha(extras[frame['id']]),
            union_lock_sha256=U.sha(output/'UNION_LOCK.json'))
        if multiple:
            key['ordered_extra_inputs_frame_sha256'] = U.payload_sha(dict(
                composite_inputs_sha256=U.payload_sha(composite), ordered_parents=frame_parents[frame['id']],
                ordered_rows_payload_sha256=key['extra_payload_sha256']))
        if seal_path.exists():
            saved = U.read(seal_path)
            if saved['dependencies'] != key:
                raise ValueError('Completed kernel-extra row dependency differs')
            for name, digest in saved['artifacts'].items():
                if U.sha(out/name) != digest:
                    raise ValueError('Completed kernel-extra artifact differs')
            feature = next(U.rows(out/'FEATURES.jsonl.gz'))
            calls['completed_row_cache_hits'] += 1
        else:
            tick = time.monotonic()
            feature, costs = _union(frame, result, bound, extras[frame['id']], config)
            if multiple:
                feature['kernel_extra']['ordered_extra_input_frame_sha256'] = key['ordered_extra_inputs_frame_sha256']
                costs['ordered_extra_input_frame_bindings'] = frame_parents[frame['id']]
                costs['ordered_extra_input_frame_sha256'] = key['ordered_extra_inputs_frame_sha256']
            U.save(out/'ACTUAL_CORE_RESULT_EXACT.jsonl.gz', [result])
            U.save(out/'ALL_EXTRA_ENDPOINTS.jsonl.gz', extras[frame['id']])
            U.save(out/'INDIVIDUAL_COSTS.jsonl.gz', [costs]); U.save(out/'FEATURES.jsonl.gz', [feature])
            extra_artifacts = ()
            if multiple:
                U.save(out/'EXTRA_INPUT_FRAME_BINDINGS.jsonl.gz', frame_parents[frame['id']])
                extra_artifacts = ('EXTRA_INPUT_FRAME_BINDINGS.jsonl.gz',)
            info = feature['kernel_extra']
            saved = dict(complete=True, id=frame['id'], ordinal=ordinal, domain=domain, core_arm=core_arm,
                dependencies=key, artifacts={name: U.sha(out/name) for name in
                    ('ACTUAL_CORE_RESULT_EXACT.jsonl.gz', 'ALL_EXTRA_ENDPOINTS.jsonl.gz', 'INDIVIDUAL_COSTS.jsonl.gz', 'FEATURES.jsonl.gz') + extra_artifacts},
                actual_core_payload_sha256=U.payload_sha(result), old_feature_context=info['old_feature_context'],
                resource=dict(wall_seconds=time.monotonic()-tick,
                    old_individual_cost_evaluations=info['old_individual_cost_evaluations'],
                    new_individual_cost_evaluations=info['new_individual_cost_evaluations'], new_pose_optimizer_calls=0))
            U.write(seal_path, saved); calls['new_rows'] += 1
            calls['new_individual_cost_evaluations'] += info['new_individual_cost_evaluations']
            calls['old_individual_cost_evaluations'] += info['old_individual_cost_evaluations']
        records.append(feature); row_seals.append(dict(id=frame['id'], seal_path=str(seal_path), sha256=U.sha(seal_path)))
        if ordinal % 100 == 0:
            print(U.canonical(dict(stage='OWN_KERNEL_EXTRA_FEATURES', core_arm=core_arm, domain=domain,
                done=ordinal+1, total=len(inputs), wall_seconds=time.monotonic()-started)), flush=True)
    U.save(output/'HEAD_FEATURES.jsonl.gz', records)
    seal_path = output/'FEATURE_SEAL.json'
    if seal_path.exists():
        seal = U.read(seal_path)
        if seal['dependencies'] != dependencies or seal['head_features_sha256'] != U.sha(output/'HEAD_FEATURES.jsonl.gz') or seal['frame_seals'] != row_seals:
            raise ValueError('Completed own-kernel aggregate differs')
        return seal
    seal = dict(schema='OWN_FULL_NORMAL_KERNEL_EXTRA_FEATURE_SEAL_V1', complete=True,
        rows=len(records), ordered_ids=ids, domain=domain, core_arm=core_arm,
        head_features_sha256=U.sha(output/'HEAD_FEATURES.jsonl.gz'), dependencies=dependencies,
        frame_seals=row_seals, actual_calls=dict(calls), original_core_endpoints_and_extra_endpoints_preserved=True,
        independent_E0_preserved=True, current_real_reference_reads=0, new_pose_optimizer_calls=0,
        wall_seconds=time.monotonic()-started, CPU_seconds=time.process_time()-cpu)
    U.write(seal_path, seal); return seal


def main():
    parser = argparse.ArgumentParser()
    for key in ('inputs-path', 'core-root', 'core-arm', 'domain', 'config-path', 'output', 'run'):
        parser.add_argument('--'+key, required=True)
    extra = parser.add_mutually_exclusive_group(required=True)
    extra.add_argument('--extra-rows-path')
    extra.add_argument('--extra-inputs-manifest')
    parser.add_argument('--root-go', action='store_true')
    parser.add_argument('--extra-seal-path'); parser.add_argument('--extra-methods', nargs='+')
    parser.add_argument('--prior-run')
    args = parser.parse_args()
    result = batch_features(args.inputs_path, args.core_root, args.core_arm, args.domain,
        args.extra_rows_path, args.config_path, args.output, args.run, root_GO=args.root_go,
        extra_seal_path=args.extra_seal_path, extra_methods=args.extra_methods, prior_run=args.prior_run,
        extra_inputs_manifest=args.extra_inputs_manifest)
    print(U.canonical({k: result[k] for k in ('complete', 'rows', 'head_features_sha256', 'actual_calls')}), flush=True)


if __name__ == '__main__':
    main()
