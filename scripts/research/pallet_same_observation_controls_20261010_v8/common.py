"""Byte-bound V7 parent evidence and exclusive V8 diagnostic outputs.

Importing this module uses the standard library only.  It does not import the
control solver, a detector, Torch, images, annotations or a scoring adapter.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[3]
CODE = Path(__file__).resolve().parent
DOC = REPO / '_docs/experiments/pallet_same_observation_controls_20261010_v8'
PRIVATE = Path('/tmp/pallet-same-observation-controls-private-20261010-v8')
PARENT_DOC = REPO / '_docs/experiments/pallet_three_head_observation_20261010_v7'
CORRECTED = REPO / '_docs/experiments/pallet_kp_corrected_supervision_20261010_v1'
OLD = REPO / '_docs/experiments/pallet_observation_refiner_20261009_v1'
PRIMARY = 'ROLE_BOUNDARY_H_ROBUST'
METHODS = (PRIMARY, 'ROLE_BOUNDARY_H_STANDARD', 'ROLE_BOUNDARY_NO_MASK_ROBUST')
CONTRASTS = tuple((method, comparator) for method in METHODS for comparator in ('BASE', 'N3_SUBPIX')) + (
    ('ROLE_BOUNDARY_H_STANDARD', PRIMARY), ('ROLE_BOUNDARY_NO_MASK_ROBUST', PRIMARY))
PARENT_PRIMARY = 'IMAGE_ROLE_BOUNDARY_ONLY'
PARENT_METHODS = (PARENT_PRIMARY, 'GEOMETRY_ONLY_BOUNDARY_ONLY',
    'IMAGE_NO_ROLE_BOUNDARY_ONLY', 'GEOMETRY_ONLY_CORNERWISE_HYBRID',
    'IMAGE_NO_ROLE_CORNERWISE_HYBRID', 'IMAGE_ROLE_CORNERWISE_HYBRID',
    'N3_INDEPENDENT_ROBUST_H', 'N3_INDEPENDENT_ROBUST_NO_MASK')
HEADS = ('GEOMETRY_ONLY', 'IMAGE_NO_ROLE', 'IMAGE_ROLE')
TRUTH_FIELDS = {'corner', 'pose', 'mask_audit', 'baseline_corner', 'baseline_pose',
    'GT', 'ground_truth', 'reference_pose', 'human_states_native'}
POLICY = dict(schema='fixed_same_sparse_ROLE_controls_execution_v8', methods=list(METHODS),
    primary=PRIMARY, parent_method=PARENT_PRIMARY, frames=245, method_rows=735,
    selection='original fixed C3 controls on the sealed V7 ROLE sparse observation',
    observation_admission_changed=False, head_threshold_CAL_query_decode_changed=False,
    primary_is_existing_parent_replay=True, new_diagnostic_methods=2,
    same_numeric_bank_per_frame=True, initial_pose_numeric_prior=False,
    native_numeric_fit_fill=False, original_H_immutable=True,
    standard_solver='all active U; whole-U SSE/LM/rank; diagnostic inliers have no acceptance gate',
    robust_solver='unchanged V4 finite four-ID consensus and ambiguity handling',
    NEW_H_reprojection_once_and_no_refit=True, new_training_updates=0, new_RGB=0,
    new_detector_N3_head_initial_pose_CAL_decode_calls=0,
    GT_selection_or_tuning=False, fresh_latency_measurement=False,
    replay_wall_seconds_are_not_deployment_latency=True,
    unseen_accuracy_evaluation=False, no_automatic_retry=True)


def require(condition, message):
    if not condition:
        raise RuntimeError('SAME_OBSERVATION_GUARD: ' + message)


def read(path):
    with Path(path).open(encoding='utf-8') as stream:
        return json.load(stream)


def rows(path):
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        for line in stream:
            require(bool(line.strip()), 'empty row in ' + Path(path).name)
            yield json.loads(line)


def finite(value):
    if hasattr(value, 'tolist'):
        return finite(value.tolist())
    if isinstance(value, dict):
        return {str(key): finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def semantic_sha256(value):
    return hashlib.sha256(json.dumps(finite(value), sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def binding(path):
    path = Path(path)
    require(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)),
            'missing/symlink input ' + str(path))
    absolute = path.resolve()
    digest = hashlib.sha256()
    with absolute.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return dict(path=str(absolute.relative_to(REPO)) if absolute.is_relative_to(REPO) else absolute.name,
        origin='public_repository' if absolute.is_relative_to(REPO) else 'external_readonly_dependency',
        sha256=digest.hexdigest(), bytes=absolute.stat().st_size)


def same_binding(left, right):
    return all(left.get(key) == right.get(key) for key in ('sha256', 'bytes'))


def bound(path, expected, label):
    require(same_binding(binding(path), expected), label + ' SHA/bytes differ')


def write_new(path, value):
    path = Path(path)
    require(not path.exists() and not path.is_symlink(), 'preserve existing ' + path.name)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(finite(value), stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def output_path(args, name, existing=False):
    require(name == Path(name).name and name not in ('', '.', '..'), 'output basename required')
    unexpanded = Path(args.output).absolute()
    require(not any(p.is_symlink() for p in (unexpanded, *unexpanded.parents)), 'symlink output ancestry')
    output = unexpanded.resolve()
    require(output == DOC.resolve() or output.is_relative_to(PRIVATE.resolve()), 'only new V8 DOC/private output')
    require(args.source_root and args.baseline_root, 'explicit source-root and baseline-root are required')
    for dependency in (args.source_root, args.baseline_root, args.parent, args.fits):
        root = Path(dependency).resolve()
        require(not output.is_relative_to(root) and not root.is_relative_to(output), 'output overlaps prior dependency')
    target = output / name
    require(not target.is_symlink() and (existing or not target.exists()), 'preserve ' + name)
    return target


def parser(description, stages=('freeze', 'preflight', 'run')):
    p = argparse.ArgumentParser(description=description)
    if stages is not None:
        p.add_argument('stage', choices=stages)
    p.add_argument('--output', default=str(PRIVATE / 'geometry'))
    p.add_argument('--protocol', default=str(DOC / 'PROTOCOL.json'))
    p.add_argument('--parent', default=str(PARENT_DOC))
    p.add_argument('--cohort', default=str(CORRECTED / 'COHORT.json'))
    p.add_argument('--source-root', default=os.environ.get('PALLET_SOURCE_ROOT'))
    p.add_argument('--baseline-root', default=os.environ.get('PALLET_BASELINE_ROOT'))
    p.add_argument('--fits', default='/dev/shm/pallet-kp-supervision-repair-private-20261010/learned_fits')
    p.add_argument('--prior-bindings', default=str(DOC / 'PROTECTION_BEFORE.json'))
    p.add_argument('--remaining-seconds', type=float, default=3600.)
    return p


def input_paths(args):
    parent = Path(args.parent)
    paths = {name: CODE / name for name in ('__init__.py', 'controls.py', 'test_controls.py',
        'common.py', 'runner.py', 'validation_checks.py', 'evaluator.py', 'statistics.py', 'verify.py', 'render.py')}
    paths.update({name: DOC / name for name in ('CPU_TEST_PROTOCOL.json', 'CONTROL_CHECKS.json',
        'CONTROL_CHECKS_STARTED.json', 'CONTROL_CHECKS_ATTEMPT_A_REASON.json',
        'CONTROL_CHECKS_ATTEMPT_B.json', 'CONTROL_CHECKS_ATTEMPT_B_STARTED.json',
        'EVALUATION_CONTRACT_KO.md', 'REPRODUCE.md')})
    paths.update({'parent:' + name: parent / name for name in ('PROTOCOL.json', 'GEOMETRY_SEAL.json',
        'INFERENCE_RECEIPT.json', 'GEOMETRY_SEALED.jsonl.gz', 'OBSERVATIONS.jsonl.gz',
        'FIXED_GEOMETRY_SEALED.jsonl.gz', 'BASE_N3_PARITY.json',
        'PREDICTIONS.jsonl.gz', 'FIXED_PREDICTIONS.jsonl.gz')})
    paths.update(cohort=Path(args.cohort), parent_protection=Path(args.prior_bindings),
        v4_pose=REPO / 'scripts/research/pallet_cornerwise_independent_20261010_v4/pose.py',
        v1_solver=REPO / 'scripts/research/pallet_observation_refiner_20261009_v1/solver.py',
        v1_common=REPO / 'scripts/research/pallet_observation_refiner_20261009_v1/common.py',
        v1_inference=REPO / 'scripts/research/pallet_observation_refiner_20261009_v1/inference.py',
        v2_assemble=REPO / 'scripts/research/pallet_boundary_corner_refiner_20261010_v2/pipeline.py',
        v2_common=REPO / 'scripts/research/pallet_boundary_corner_refiner_20261010_v2/common.py',
        v2_protection=REPO / 'scripts/research/pallet_boundary_corner_refiner_20261010_v2/protection.py',
        v7_streamer=REPO / 'scripts/research/pallet_three_head_observation_20261010_v7/run.py',
        v7_common=REPO / 'scripts/research/pallet_three_head_observation_20261010_v7/common.py',
        quiet_helper=REPO / 'scripts/research/pallet_corner_mechanism_audit_20261010_v1/deployment_smoke.py')
    paths.update(v7_evaluate=REPO / 'scripts/research/pallet_three_head_observation_20261010_v7/evaluate.py',
        v7_statistics=REPO / 'scripts/research/pallet_three_head_observation_20261010_v7/statistics.py',
        v2_scoring_canary=REPO / 'scripts/research/pallet_boundary_corner_refiner_20261010_v2/evaluate.py',
        v1_metric=REPO / 'scripts/research/pallet_observation_refiner_20261009_v1/evaluate.py',
        reference_adapter=REPO / 'scripts/research/pallet_kp_corrected_supervision_20261010_v1/scoring_resume.py',
        reference_ID_mapping=CORRECTED / 'REFERENCE_ID_MAPPING.json',
        original_authority=OLD / 'INPUTS.json',
        original_context=REPO / 'scripts/research/pallet_kp_repair_runtime_20261010_v1/adapter.py',
        bootstrap_draws=REPO / '_docs/experiments/pallet_kp_difficulty_20261010_v1/BOOTSTRAP_SESSION_DRAWS.json.gz')
    paths['visual_case_protocol'] = REPO / '_docs/experiments/pallet_boundary_corner_refiner_20261010_v2/VISUAL_CASE_PROTOCOL.json'
    paths.update(source_GT_targets=Path(args.source_root) / 'data/pallet/results/pallet_posefix_replay_diagnosis_v1/TARGETS.json',
        original_BASE_N3_reference=Path(args.baseline_root) / '_docs/experiments/pallet_joint_action_handoff_20261006_v1/results/A_REAL_DEV_BASELINES.json',
        source_visibility_audit=Path(args.source_root) / '_docs/experiments/pallet_combined_closeout_20261003_v1/closeout_20261006_v1/visibility_square/STATIC_VISIBILITY_MERGE_AUDIT.json')
    return paths


# Byte-bind these before execution; never decode scored/reference inputs for
# observation/solver selection. Their actual use is only after complete seal.
POSTSEAL_ONLY_INPUTS = ('parent:PREDICTIONS.jsonl.gz', 'parent:FIXED_PREDICTIONS.jsonl.gz',
    'reference_ID_mapping', 'reference_adapter', 'v1_metric', 'v2_scoring_canary',
    'v7_evaluate', 'v7_statistics', 'original_context', 'bootstrap_draws',
    'source_GT_targets', 'original_BASE_N3_reference', 'source_visibility_audit', 'render.py', 'visual_case_protocol')


def input_bindings(args):
    return {name: binding(path) for name, path in input_paths(args).items()}


def verify_protocol(args):
    protocol = read(args.protocol)
    require(protocol['schema'] == 'fixed_same_observation_C3_protocol_v8' and
            protocol['policy'] == POLICY and protocol['methods'] == list(METHODS) and
            protocol['frames'] == 245 and protocol['rows'] == 735 and
            protocol['contrasts'] == [list(pair) for pair in CONTRASTS], 'fixed C3 scope/policy differs')
    require(protocol['inputs'] == input_bindings(args), 'frozen V8 code/parent evidence differs')
    return protocol


def cohort_ids(args):
    cohort = read(args.cohort)
    require(cohort['counts'] == dict(clean=153, moderate=92, severe_excluded=74, original=319),
            'cohort difficulty scope differs')
    ids = cohort['ids']
    require(len(ids) == len(set(ids)) == 245 and all(isinstance(k, str) for k in ids), 'cohort IDs differ')
    require([entry['id'] for entry in cohort['frames']] == ids, 'cohort order differs')
    require(all(entry['label'] in ('clean', 'moderate') for entry in cohort['frames']), 'Severe entered new controls')
    excluded = cohort['excluded_ids']
    require(len(excluded) == len(set(excluded)) == 74 and not set(ids).intersection(excluded), 'excluded IDs differ')
    return list(ids)


def parent_receipts(args, bindings):
    paths = input_paths(args)
    protocol, seal, receipt = [read(paths['parent:' + name]) for name in
        ('PROTOCOL.json', 'GEOMETRY_SEAL.json', 'INFERENCE_RECEIPT.json')]
    require(protocol['frames'] == 245 and protocol['methods'] == list(PARENT_METHODS) and
            protocol['primary'] == PARENT_PRIMARY, 'parent frozen scope differs')
    require(seal['complete'] is True and seal['GT_read_allowed'] is False and seal['frames'] == 245 and
            seal['rows'] == 1960 and seal['fixed_rows'] == 490 and seal['methods'] == list(PARENT_METHODS),
            'incomplete or non-GT-free parent seal')
    require(receipt['complete'] is True and receipt['actual_complete_frames'] == 245 and
            receipt['method_rows'] == 1960 and receipt['observation_rows'] == 735 and receipt['fixed_rows'] == 490 and
            receipt['cleanup_error'] is None and receipt['inference_error'] is None and
            receipt['GT_access_during_inference'] is False, 'parent did not finish cleanly')
    for key, file in (('protocol', 'PROTOCOL.json'), ('geometry', 'GEOMETRY_SEALED.jsonl.gz'),
                     ('observations', 'OBSERVATIONS.jsonl.gz'), ('fixed_geometry', 'FIXED_GEOMETRY_SEALED.jsonl.gz'),
                     ('parity', 'BASE_N3_PARITY.json')):
        require(same_binding(seal[key], bindings['parent:' + file]), 'parent seal binding differs: ' + key)
    require(same_binding(protocol['inputs']['cohort'], bindings['cohort']), 'parent cohort binding differs')
    for kind, count in (('geometry', 1960), ('observations', 735), ('fixed', 490)):
        stream = receipt['serialization']['streams'][kind]
        require(stream['rows'] == count and stream['published'] is True and not stream['close_errors'] and
                not stream['interruption_preservation_errors'], 'parent stream cleanup incomplete: ' + kind)
    require(read(paths['parent:BASE_N3_PARITY.json'])['passed'] is True, 'parent fixed prediction parity failed')
    return dict(complete=True, frames=245, geometry_rows=1960, observation_rows=735, fixed_rows=490,
        cleanup_error=None, protocol_sha256=bindings['parent:PROTOCOL.json']['sha256'],
        geometry_sha256=bindings['parent:GEOMETRY_SEALED.jsonl.gz']['sha256'],
        inference_receipt_sha256=bindings['parent:INFERENCE_RECEIPT.json']['sha256'])


def _group(iterator, identity, field, names):
    result = {}
    for _ in names:
        row = next(iterator, None)
        require(row is not None and row.get('id') == identity and row.get(field) in names,
                'parent grouped population/order differs: ' + identity + ':' + field)
        require(row[field] not in result and not TRUTH_FIELDS.intersection(row), 'duplicate/scored parent row')
        result[row[field]] = row
    require(set(result) == set(names), 'incomplete parent group')
    return result


def parent_frames(args):
    """One complete frame at a time; discard the other seven pose witnesses.

    This same full-population iterator is exhausted before the first PnP call,
    then reopened for execution. No incomplete parent prefix can start a solve.
    """
    folder = Path(args.parent)
    geometry = iter(rows(folder / 'GEOMETRY_SEALED.jsonl.gz'))
    observations = iter(rows(folder / 'OBSERVATIONS.jsonl.gz'))
    fixed = iter(rows(folder / 'FIXED_GEOMETRY_SEALED.jsonl.gz'))
    for identity in cohort_ids(args):
        g = _group(geometry, identity, 'method', PARENT_METHODS)
        o = _group(observations, identity, 'head_arm', HEADS)
        f = _group(fixed, identity, 'method', ('BASE', 'N3_SUBPIX'))
        primary, role = g[PARENT_PRIMARY], o['IMAGE_ROLE']
        session = primary['session']
        require(all(r['session'] == session for r in (*g.values(), *o.values(), *f.values())), 'parent session mismatch')
        require(all(r.get('GT_input') is False for r in o.values()), 'parent observations are not GT-free')
        for row in (*g.values(), *f.values()):
            for key in ('K', 'xyz', 'raw_hw', 'selected_index', 'fixed_metadata'):
                require(row[key] == primary[key], 'parent deployable metadata differs: ' + key)
        require(all(semantic_sha256(r['native_N3_points']) == semantic_sha256(role['native_N3_points']) and
                    semantic_sha256(r['initial_N3_pose']) == semantic_sha256(role['initial_N3_pose']) and
                    r['predicted_N3_hidden'] == role['predicted_N3_hidden'] for r in o.values()),
                'parent heads did not share original native/initial/H')
        require(semantic_sha256(primary['initial_pose']) == semantic_sha256(role['initial_N3_pose']) and
                primary['hidden_initial'] == primary['predicted_initial_N3_hidden'] == role['predicted_N3_hidden'],
                'ROLE parent initial pose/H differs from original observation')
        require(semantic_sha256(f['N3_SUBPIX']['native_points']) == semantic_sha256(role['native_N3_points']),
                'OBS native N3 differs from fixed N3 output')
        require(semantic_sha256(f['BASE']['native_points']) == semantic_sha256(role['original_base_points']),
                'OBS original Base differs from fixed Base output')
        # Detector center is an output invariant, never a ninth PnP corner.
        require(primary['input_points'][8] == role['native_N3_points'][8], 'sealed center mismatch')
        require(primary['observation_raw_logits_sha256'] == role.get('raw_logits_sha256'), 'ROLE logits digest join differs')
        del g, o, f
        yield primary, role
    require(next(geometry, None) is None and next(observations, None) is None and next(fixed, None) is None,
            'parent has rows outside full frozen populations')


def population_proof(args, bindings):
    completion = parent_receipts(args, bindings)
    frames = 0
    sessions = Counter()
    for primary, _ in parent_frames(args):
        frames += 1
        sessions[primary['session']] += 1
    require(frames == 245 and sum(sessions.values()) == 245, 'parent population incomplete')
    return dict(schema='full_GT_free_V7_parent_population_checks_v8', complete=True, passed=True,
        frames=frames, geometry_rows=1960, observation_rows=735, fixed_rows=490,
        methods=list(PARENT_METHODS), heads=list(HEADS), sessions=dict(sessions),
        ordered_ids=cohort_ids(args), completion=completion,
        full_population_checked_before_first_control_pose_path=True,
        scored_or_human_fields_passed_to_solver=False, full_parent_population_rows_retained=0,
        one_frame_primary_and_ROLE_observation_kept=True,
        native_source='OBSERVATIONS.IMAGE_ROLE.native_N3_points; fixed N3 parity checked',
        source_model_image_GT_PnP_calls=0)


def protect(args):
    # Reuse recorded tracked/source-byte protection, outside the numeric canary.
    from ..pallet_boundary_corner_refiner_20261010_v2.protection import verify
    snapshot = read(args.prior_bindings)
    # The parent may wrap the same generic snapshot as {tracked: ...}; a new
    # V8 snapshot protects the just-published V7 HEAD rather than its old base.
    result = verify(snapshot.get('tracked', snapshot), args.source_root)
    require(result['passed'], 'published parent/source protection failed')
    for entry in snapshot.get('additional_completed_audits', []):
        bound(REPO / entry['path'], entry, 'additional completed audit')
    return result


def legacy_context(args):
    """Unchanged scoring reference context; evaluator must first verify seal."""
    from ..pallet_three_head_observation_20261010_v7.common import legacy_context as context
    return context(args)


def quiet_snapshot():
    from ..pallet_corner_mechanism_audit_20261010_v1.deployment_smoke import quiet
    return quiet()


@contextmanager
def original_environment(args):
    """Minimal context for assemble's original hidden_mask import; no models.

    Explicit split CLI dependencies are required even though they are not read
    for observations. Do not invoke legacy source_modules/model construction.
    """
    names = ('PALLET_SOURCE_ROOT', 'PALLET_BASELINE_ROOT')
    saved = {name: os.environ.get(name) for name in names}
    requested = dict(zip(names, (str(Path(args.source_root).resolve()), str(Path(args.baseline_root).resolve()))))
    for name, value in requested.items():
        require(Path(value).is_dir(), 'original split dependency missing: ' + name)
        if saved[name] is not None:
            require(str(Path(saved[name]).resolve()) == value, 'CLI/environment split root differs: ' + name)
    os.environ.update(requested)
    try:
        existing = sys.modules.get('scripts.research.pallet_observation_refiner_20261009_v1.common')
        if existing is not None:
            require(existing.ROOT.resolve() == Path(requested['PALLET_SOURCE_ROOT']), 'previous original common root differs')
        yield requested
        existing = sys.modules.get('scripts.research.pallet_observation_refiner_20261009_v1.common')
        if existing is not None:
            require(existing.ROOT.resolve() == Path(requested['PALLET_SOURCE_ROOT']), 'original hidden-mask import resolved wrong root')
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
