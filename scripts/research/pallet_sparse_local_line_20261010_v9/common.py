"""Byte bindings and GT-free streaming for the one original-C2 local arm.

This module imports only stdlib helpers. V8 population/protection/environment
utilities are reused unchanged, with their original schema retained in proof.
"""
from __future__ import annotations

from collections import Counter
import copy
from pathlib import Path

from ..pallet_same_observation_controls_20261010_v8 import common as V8

REPO, CODE = V8.REPO, Path(__file__).resolve().parent
DOC = REPO / '_docs/experiments/pallet_sparse_local_line_20261010_v9'
PRIVATE = Path('/tmp/pallet-sparse-local-line-private-20261010-v9')
PARENT_DOC, POINT_PARENT_DOC = V8.PARENT_DOC, V8.DOC
PUBLISHED_POINT_HEAD = '07d5f5c616956c70717f3df9ef0dd1f4c439169e'
CORRECTED, OLD = V8.CORRECTED, V8.OLD
PRIMARY = 'ROLE_BOUNDARY_LOCAL_POINT_LINE'
METHODS = (PRIMARY,)
COMPARATOR, PARENT_PRIMARY = V8.PRIMARY, V8.PARENT_PRIMARY
CONTRASTS = ((PRIMARY, 'BASE'), (PRIMARY, 'N3_SUBPIX'), (PRIMARY, COMPARATOR))
POLICY = dict(schema='fixed_same_sparse_ROLE_local_C2_execution_v9', methods=list(METHODS),
    frames=245, rows=245, primary=PRIMARY, comparator=COMPARATOR,
    observation='sealed V7 IMAGE_ROLE sparse q and actual unused semantic lines',
    numeric_fit_fill=False, original_H_immutable=True,
    initial_pose='stored N3 and available stored point-only physical poses are LOCAL optimizer starts only',
    residual_prior=False, both_registry_dimensions_equally_tried=True,
    start_sources=2, maximum_logical_starts_per_frame=4,
    loss='soft_l1', f_scale_px=8., max_nfev=50,
    whole_pool_soft_l1_cost_selection=True, initial_rank_is_diagnostic_only=True,
    final_observed_and_modeled_J_rank6=True, inlier8px_is_diagnostic_only=True,
    empty_line_pool_point_LOCAL_reported_separately=True,
    local_only=True, global_uniqueness_proven=False,
    NEW_is_numerical_LOCAL_output_not_accuracy_success=True,
    NEW_H_reprojection_and_no_refit=True, no_automatic_retry=True,
    head_N3_RGB_CAL_decode_initial_PnP_calls=0, training_updates=0, new_RGB=0,
    evaluation_is_unseen=False, performance_based_policy_changes=0,
    fresh_latency_measurement=False, replay_wall_is_not_deployment_latency=True)

require, read, rows, finite = V8.require, V8.read, V8.rows, V8.finite
binding, bound, same_binding = V8.binding, V8.bound, V8.same_binding
semantic_sha256, write_new = V8.semantic_sha256, V8.write_new
cohort_ids, quiet_snapshot = V8.cohort_ids, V8.quiet_snapshot
original_environment, legacy_context = V8.original_environment, V8.legacy_context


def parser(description, stages=('freeze', 'preflight', 'run')):
    p = V8.parser(description, stages)
    p.set_defaults(output=str(PRIVATE / 'geometry'), protocol=str(DOC / 'PROTOCOL.json'),
                   prior_bindings=str(DOC / 'PROTECTION_BEFORE.json'))
    p.add_argument('--point-parent', default=str(POINT_PARENT_DOC))
    p.add_argument('--cpu-protocol', default=str(DOC / 'CPU_TEST_PROTOCOL.json'))
    p.add_argument('--cpu-original', default=str(DOC / 'LOCAL_LINE_CHECKS.json'))
    p.add_argument('--cpu-review-protocol', default=str(DOC / 'CPU_REVIEW_PROTOCOL.json'))
    p.add_argument('--cpu-review-success', default=str(DOC / 'LOCAL_LINE_REVIEW_CHECKS.json'))
    p.add_argument('--cpu-success', default=str(DOC / 'JOINED_LOCAL_CONTRACT_CHECKS.json'))
    return p


def output_path(args, name, existing=False):
    require(name == Path(name).name and name not in ('', '.', '..'), 'new V9 output basename required')
    original = Path(args.output).absolute()
    require(not any(p.is_symlink() for p in (original, *original.parents)), 'symlink output ancestry')
    output = original.resolve()
    require(output == DOC.resolve() or output.is_relative_to(PRIVATE.resolve()), 'only new V9 DOC/private output')
    require(args.source_root and args.baseline_root, 'explicit source/baseline roots required')
    for dependency in (args.source_root, args.baseline_root, args.parent, args.point_parent, args.fits):
        root = Path(dependency).resolve()
        require(not output.is_relative_to(root) and not root.is_relative_to(output), 'V9 output overlaps protected dependency')
    target = output / name
    require(not target.is_symlink() and (existing or not target.exists()), 'preserve ' + name)
    return target


def input_paths(args):
    base = V8.input_paths(args)
    paths = dict(base)
    # Keep unchanged V8 code/contracts explicitly bound when own basenames
    # replace their keys. V8 input schemas are not relabelled as V9 evidence.
    for key, path in base.items():
        if path.is_relative_to(V8.CODE) or path.is_relative_to(V8.DOC):
            paths['reused_v8:' + key] = path
    for name in ('__init__.py', 'solver.py', 'pipeline.py', 'test_solver.py', 'common.py',
                 'runner.py', 'validation_checks.py', 'evaluator.py', 'statistics.py', 'verify.py', 'render.py',
                 'test_solver_review.py'):
        paths[name] = CODE / name
    for name in ('EVALUATION_CONTRACT_KO.md', 'REPRODUCE.md'):
        paths[name] = DOC / name
    paths.update(v5_line_solver=REPO / 'scripts/research/pallet_partial_line_independent_20261010_v5/solver.py',
                 original_local_kernel=REPO / 'scripts/research/pallet_observation_refiner_20261009_v1/point_line.py',
                 cpu_protocol=Path(args.cpu_protocol), cpu_success=Path(args.cpu_success),
                 cpu_original=Path(args.cpu_original), cpu_review_protocol=Path(args.cpu_review_protocol),
                 cpu_review_success=Path(args.cpu_review_success))
    for key in ('cpu_original', 'cpu_review_success'):
        original = paths[key]
        paths[key + '_started'] = original.with_name(original.stem + '_STARTED.json')
    # All prior attempts/protocols survive; success is an explicit authority,
    # not whichever receipt happens to have the most favourable result.
    for pattern in ('LOCAL_LINE_CHECKS*.json', 'LOCAL_LINE_REVIEW_CHECKS*.json',
                    'CPU_TEST_PROTOCOL*.json', 'CPU_REVIEW_PROTOCOL*.json', 'JOINED_LOCAL_CONTRACT_CHECKS*.json'):
        for path in sorted(DOC.glob(pattern)):
            paths['cpu_history:' + path.name] = path
    point = Path(args.point_parent)
    for name in ('PROTOCOL.json', 'GEOMETRY_SEAL.json', 'GEOMETRY_SEALED.jsonl.gz',
                 'CONTROL_RECEIPT.json', 'CONTROL_LEDGERS.jsonl.gz', 'PARENT_POPULATION_CHECKS.json',
                 'VALIDATION_PROTOCOL.json', 'VALIDATION_CHECKS.json',
                 'SCORING_RECEIPT.json', 'SCORING_PARITY.json', 'PREDICTIONS.jsonl.gz',
                 'FIXED_PREDICTIONS.jsonl.gz', 'METRICS.json'):
        paths['point:' + name] = point / name
    return paths


def protect(args):
    """Require the new protection of published V8, then reuse its verifier."""
    require(Path(args.prior_bindings).resolve() == (DOC / 'PROTECTION_BEFORE.json').resolve(),
            'the new V9 protection snapshot is required')
    snapshot = read(args.prior_bindings)
    require(snapshot.get('tracked', snapshot).get('original_research_head') == PUBLISHED_POINT_HEAD,
            'protection must cover the published V8 commit, not an older parent')
    return V8.protect(args)


POSTSEAL_ONLY_INPUTS = tuple(V8.POSTSEAL_ONLY_INPUTS) + (
    'point:SCORING_RECEIPT.json', 'point:SCORING_PARITY.json', 'point:PREDICTIONS.jsonl.gz',
    'point:FIXED_PREDICTIONS.jsonl.gz', 'point:METRICS.json', 'evaluator.py', 'statistics.py', 'verify.py')


def input_bindings(args):
    return {key: binding(path) for key, path in input_paths(args).items()}


def verify_protocol(args):
    protocol = read(args.protocol)
    require(protocol['schema'] == 'fixed_same_observation_local_C2_protocol_v9' and
            protocol['policy'] == POLICY and protocol['methods'] == list(METHODS) and
            protocol['frames'] == protocol['rows'] == 245 and
            protocol['contrasts'] == [list(pair) for pair in CONTRASTS], 'fixed local C2 policy/scope differs')
    require(protocol['inputs'] == input_bindings(args), 'frozen V9 code/evidence differs')
    return protocol


def point_receipts(args, bindings):
    folder = Path(args.point_parent)
    protocol, seal, receipt = [read(folder / name) for name in
        ('PROTOCOL.json', 'GEOMETRY_SEAL.json', 'CONTROL_RECEIPT.json')]
    require(protocol['schema'] == 'fixed_same_observation_C3_protocol_v8' and protocol['methods'] == list(V8.METHODS),
            'point comparator is not the fixed published C3 control')
    require(seal['complete'] is True and seal['GT_read_allowed'] is False and
            seal['frames'] == 245 and seal['rows'] == 735 and seal['ledger_rows'] == 245,
            'point control seal incomplete')
    require(receipt['complete'] is True and receipt['actual_complete_frames'] == 245 and
            receipt['error'] is None and receipt['cleanup_error'] is None and
            receipt['GT_access_during_controls'] is False and
            receipt['environment_and_monkeypatch_cleanup_completed'] is True, 'point control cleanup incomplete')
    for key, filename in (('protocol', 'PROTOCOL.json'), ('geometry', 'GEOMETRY_SEALED.jsonl.gz'),
                          ('ledgers', 'CONTROL_LEDGERS.jsonl.gz'), ('control_receipt', 'CONTROL_RECEIPT.json'),
                          ('parent_population_checks', 'PARENT_POPULATION_CHECKS.json')):
        require(same_binding(seal[key], bindings['point:' + filename]), 'point seal binding differs:' + key)
    require(same_binding(protocol['inputs']['parent:GEOMETRY_SEALED.jsonl.gz'],
                         bindings['parent:GEOMETRY_SEALED.jsonl.gz']) and
            same_binding(protocol['inputs']['parent:OBSERVATIONS.jsonl.gz'], bindings['parent:OBSERVATIONS.jsonl.gz']),
            'point control does not belong to the same V7 observation parent')
    validation = read(folder / 'VALIDATION_CHECKS.json')
    require(validation['complete'] is True and validation['passed'] is True, 'point control standalone validation failed')
    for key in ('GEOMETRY_SEAL.json', 'GEOMETRY_SEALED.jsonl.gz', 'CONTROL_LEDGERS.jsonl.gz', 'CONTROL_RECEIPT.json'):
        require(same_binding(validation['inputs'][key], bindings['point:' + key]), 'point standalone validation binding differs')
    return dict(complete=True, frames=245, rows=735, method=COMPARATOR,
        protocol=bindings['point:PROTOCOL.json'], geometry=bindings['point:GEOMETRY_SEALED.jsonl.gz'],
        receipt=bindings['point:CONTROL_RECEIPT.json'], standalone_validation=bindings['point:VALIDATION_CHECKS.json'])


def parent_frames(args):
    """Stream V7 parent/ROLE and the same-frame V8 Hrobust control together."""
    controls = iter(rows(Path(args.point_parent) / 'GEOMETRY_SEALED.jsonl.gz'))
    for parent, role in V8.parent_frames(args):
        group = V8._group(controls, parent['id'], 'method', V8.METHODS)
        control = group[COMPARATOR]
        require(all(row['session'] == parent['session'] for row in group.values()), 'point control session differs')
        require(control['parent_geometry_row_semantic_sha256'] == semantic_sha256(parent) and
                semantic_sha256(control['input_points']) == semantic_sha256(parent['input_points']) and
                control['hidden_initial'] == parent['hidden_initial'], 'same sparse point comparator join differs')
        require(control['head_arm'] == 'IMAGE_ROLE' and control['oracle'] is False and
                control['existing_control_replay'] is True, 'point comparator is not the existing masked robust replay')
        del group
        yield parent, role, control
    require(next(controls, None) is None, 'point control has rows outside population')


def population_proof(args, bindings):
    original = V8.population_proof(args, bindings)
    point = point_receipts(args, bindings)
    frames, sessions = 0, Counter()
    for parent, _, _ in parent_frames(args):
        frames += 1
        sessions[parent['session']] += 1
    require(frames == 245, 'local input population incomplete')
    return dict(schema='full_GT_free_local_C2_parent_population_checks_v9', complete=True, passed=True,
        frames=245, point_control_rows=735, sessions=dict(sessions), ordered_ids=cohort_ids(args),
        completion=copy.deepcopy(original['completion']), reused_unchanged_v8_population_proof=original,
        reused_proof_original_schema_retained=True, point_completion=point,
        full_population_checked_before_first_local_optimizer=True,
        GT_or_scored_inputs_decoded=False, full_parent_population_rows_retained=0,
        one_frame_primary_ROLE_and_point_control_kept=True,
        source_model_image_GT_PnP_optimizer_calls=0)
