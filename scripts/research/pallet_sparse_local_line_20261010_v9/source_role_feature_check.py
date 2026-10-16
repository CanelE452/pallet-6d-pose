"""Supplemental CAL128 replay of the existing Base-derived role features.

This new file is outside the frozen V9 accuracy core. Freeze it and its inputs
before replay. The unchanged initial_geometry function is AST-extracted without
importing the model/Torch. No image, detector, head, source depth/ray, training
or N3 runs. The original cached feature pose was not retained: matching roles
does not establish historical identity of the replayed pose or branch.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from contextlib import contextmanager
import copy
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[3]
CODE = Path(__file__).resolve().parent
DOC = REPO/'_docs/experiments/pallet_sparse_local_line_20261010_v9'
PRIVATE = Path('/tmp/pallet-sparse-local-line-private-20261010-v9')
CACHE = Path('/dev/shm/pallet-observation-private-20261009/learned_cache')
REPAIR = REPO/'_docs/experiments/pallet_kp_supervision_repair_20261010_v1'
DIFFICULTY = REPO/'_docs/experiments/pallet_kp_difficulty_20261010_v1'
LEGACY = REPO/'scripts/research/pallet_observation_refiner_20261009_v1'
OUTPUTS = ('SOURCE_ROLE_PROTOCOL.json', 'SOURCE_ROLE_STARTED.json', 'SOURCE_ROLE_CHECKS.json',
           'SOURCE_ROLE_ROWS.jsonl.gz')
PRIMITIVES = ('solvePnP', 'solvePnPGeneric', 'solvePnPRefineLM', 'projectPoints', 'Rodrigues', 'cornerSubPix', 'convexHull')
WHITELIST = ('Base_points', 'K', 'dimensions', 'raw_hw')
POLICY = dict(source_partition='calibration', fixed_indices=list(range(768, 896)), frames=128,
    query_count=84, edge_count=12, queries_per_edge=7, candidate_bins=65,
    role_feature_channels=[25, 26, 27], feature_dtype='float16', feature_shape=[1024, 84, 28, 65],
    model_input_anchor='frozen original Base selected detector candidate; never N3',
    role_definition='unchanged old initial_geometry projected convex-hull boundary/internal/unavailable',
    geometry_arguments=list(WHITELIST), source_GT_pose_used_as_role_input=False,
    independent_known_dimension_prior=False, both_legacy_registry_dimensions=True,
    no_new_role_policy_or_solver=True, original_feature_tensor_regenerated=False,
    original_pose_witness_missing=True, historical_pose_identity_verified=False,
    role_agreement_does_not_prove_historical_pose_identity=True,
    role_proxy_does_not_certify_actual_physical_boundary_or_visibility=True,
    full_1024_target_wire_contract_repeated=False, target_supervision_checked=False,
    source_metadata_has_other_fields=True, only_whitelisted_numeric_metadata_retained_for_replay=True,
    query_labels_or_source_R_t_or_depth_never_enter_geometry=True,
    detector_head_model_image_GT_depth_ray_N3_training_RGB_calls=0,
    fresh_role_pose_replay_calls=128, automatic_retry=False, performance_based_policy_changes=0)


def require(value, reason):
    if not value:
        raise ValueError(reason)


def read(path):
    with Path(path).open(encoding='utf-8') as stream:
        return json.load(stream)


def rows(path):
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        for line in stream:
            require(bool(line.strip()), 'blank source row')
            yield json.loads(line)


def binding(path):
    path = Path(path).absolute()
    require(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)), 'missing/symlink read-only dependency')
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for part in iter(lambda: stream.read(1024*1024), b''):
            h.update(part)
    resolved = path.resolve()
    return dict(path=str(resolved.relative_to(REPO)) if resolved.is_relative_to(REPO) else resolved.name,
                bytes=resolved.stat().st_size, sha256=h.hexdigest())


def safe(value):
    if hasattr(value, 'tolist'):
        return safe(value.tolist())
    if isinstance(value, dict):
        return {str(k): safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(v) for v in value]
    return None if isinstance(value, float) and not math.isfinite(value) else value


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(safe(value), stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def ast_function(path, name):
    tree = ast.parse(Path(path).read_text())
    found = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name]
    require(len(found) == 1, 'unchanged unique function absent: ' + name)
    return copy.deepcopy(found[0])


def ast_sha(node):
    return hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()


def extracted_contract(paths):
    initial = ast_function(paths['model_code'], 'initial_geometry')
    require([a.arg for a in initial.args.args] == ['points', 'K', 'xyz', 'hw'] and
            not initial.args.kwonlyargs and not initial.args.defaults,
            'original initial_geometry interface changed')
    module = ast.fix_missing_locations(ast.Module(body=[copy.deepcopy(initial)], type_ignores=[]))
    graph = ast.parse(paths['source_audit_code'].read_text())
    assignments = [n for n in graph.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'EDGES' for t in n.targets)]
    require(len(assignments) == 1, 'original source edge graph absent')
    edges = ast.literal_eval(assignments[0].value)
    require(len(edges) == 12 and all(len(e) == 2 for e in edges), 'original twelve edges required')
    return dict(initial_geometry_AST_sha256=ast_sha(initial), extracted_module_AST_sha256=ast_sha(module),
        visibility_AST_sha256=ast_sha(ast_function(paths['solver_code'], 'visibility')),
        cuboid_AST_sha256=ast_sha(ast_function(paths['solver_code'], 'cuboid')),
        projection_AST_sha256=ast_sha(ast_function(paths['solver_code'], 'project')),
        EDGES=[list(e) for e in edges], unchanged_body_extraction=True, model_module_imported=False), module, edges


def input_paths(args):
    return dict(code=Path(__file__), manifest=args.cache/'CACHE_MANIFEST.json', features=args.cache/'features.npy',
        source_metadata=args.metadata, original_cached_roles=args.ceiling,
        model_code=LEGACY/'model.py', training_code=LEGACY/'training.py', solver_code=LEGACY/'solver.py',
        source_audit_code=LEGACY/'source_audit.py')


def population(paths):
    manifest = read(paths['manifest'])
    require(manifest['complete'] is True and manifest['specs']['features'] ==
            dict(shape=POLICY['feature_shape'], dtype=POLICY['feature_dtype']), 'complete unchanged FP16 feature manifest required')
    records = manifest['records']
    require(len(records) == 1024 and [r['index'] for r in records] == list(range(1024)), 'original source manifest population/order differs')
    selected = [r for r in records if r['partition'] == 'calibration']
    require(len(selected) == 128 and [r['index'] for r in selected] == POLICY['fixed_indices'], 'fixed CAL128 partition/indices differ before PnP')
    require(all(r['variant'] == 0 and r['composition'].get('generated') is False and
        r['selected_detector_candidate'] is not None and isinstance(r['selected_points'], list) and
        len(r['selected_points']) == 9 and all(isinstance(p, list) and len(p) == 2 for p in r['selected_points']) for r in selected),
        'existing original-P0 selected Base observations required; no invented missing detections')
    by_index, metadata, roles = {r['index']: r for r in selected}, {}, {}
    for row in rows(paths['source_metadata']):
        i = row['index']
        if i not in by_index:
            continue
        old = by_index[i]
        require(i not in metadata and row['id'] == old['id'] and row['partition'] == old['partition'] and
            row['family'] == old['family'] and row['frozen_selected_points'] == old['selected_points'],
            'source numeric metadata/Base cache identity differs')
        # No R,t/query/target/depth field is accessed or retained in this packet.
        metadata[i] = dict(Base_points=copy.deepcopy(old['selected_points']), K=copy.deepcopy(row['K']),
                          dimensions=copy.deepcopy(row['dimensions']), raw_hw=copy.deepcopy(row['raw_hw']))
    for row in rows(paths['original_cached_roles']):
        i = row['index']
        if i not in by_index:
            continue
        old = by_index[i]
        require(i not in roles and row['id'] == old['id'] and row['partition'] == old['partition'] and
            row['frozen_selected_points'] == old['selected_points'], 'old cached role/Base identity differs')
        require(row['predicted_initial_H'] is None and row['predicted_initial_H_status'].startswith('NOT_RETAINED'),
                'historical initial pose/H witness assumption differs')
        role = row['predicted_role_query_ids']
        require(isinstance(role, list) and len(role) == 84 and all(type(v) is int and 0 <= v < 3 for v in role), 'saved role IDs invalid')
        roles[i] = copy.deepcopy(role)
    require(set(metadata) == set(roles) == set(POLICY['fixed_indices']), 'complete CAL metadata/roles required before PnP')
    for packet in metadata.values():
        require(set(packet) == set(WHITELIST), 'role geometry whitelist differs')
    return selected, metadata, roles


def guard(args, stage):
    for path in (args.output, args.cache, args.metadata, args.ceiling):
        require(not any(p.is_symlink() for p in (path.absolute(), *path.absolute().parents)), 'symlink dependency/output ancestry')
    out = args.output.resolve()
    require(out == DOC.resolve() or out.is_relative_to(PRIVATE.resolve()), 'only exclusive new V9 DOC/private supplemental output')
    require(out.is_dir(), 'existing output directory required')
    for name in OUTPUTS if stage == 'freeze' else OUTPUTS[1:]:
        require(not (out/name).exists(), 'preserve first/existing source role attempt:' + name)
    if stage == 'run':
        require((out/OUTPUTS[0]).is_file(), 'own freeze before source role replay')
    return out


def freeze(args):
    out = guard(args, 'freeze'); paths = input_paths(args)
    inputs = {k: binding(p) for k, p in paths.items()}
    extracted, _, _ = extracted_contract(paths)
    selected, _, _ = population(paths)
    require({k: binding(p) for k, p in paths.items()} == inputs, 'source inputs changed during structural freeze')
    write_new(out/OUTPUTS[0], dict(schema='supplemental_Base_source_CAL_role_replay_protocol_v1',
        inputs=inputs, extraction=extracted, policy=POLICY,
        fixed_identity=[dict(index=r['index'], id=r['id'], family=r['family'], partition=r['partition']) for r in selected],
        numeric_replay_executed_at_freeze=0, fresh_code_body_policy=False))
    print('SOURCE_ROLE_FROZEN', binding(out/OUTPUTS[0])['sha256'], flush=True)


@contextmanager
def file_canary(allowed, attempted):
    import builtins
    import io
    owners = [(builtins, 'open', builtins.open), (io, 'open', io.open), (Path, 'open', Path.open)]
    allowed = {p.resolve() for p in allowed}
    def wrap(fn):
        def call(path, *a, **kw):
            mode = kw.get('mode', a[0] if a and isinstance(a[0], str) else 'r')
            if isinstance(path, (str, Path)) and 'r' in mode:
                p = Path(path).absolute()
                if p.suffix.lower() in ('.png', '.jpg', '.jpeg', '.tiff', '.exr', '.pt', '.pth', '.usd', '.glb', '.obj', '.npz', '.npy') and p.resolve() not in allowed:
                    attempted.append(str(p)); raise ValueError('source role replay attempted image/model/depth/mesh read')
            return fn(path, *a, **kw)
        return call
    for owner, name, fn in owners:
        setattr(owner, name, wrap(fn))
    try:
        yield
    finally:
        for owner, name, fn in owners:
            setattr(owner, name, fn)


def run(args):
    out = guard(args, 'run'); paths = input_paths(args)
    own = read(out/OUTPUTS[0]); inputs = {k: binding(p) for k, p in paths.items()}
    extracted, module, edges = extracted_contract(paths)
    require(own['schema'] == 'supplemental_Base_source_CAL_role_replay_protocol_v1' and own['inputs'] == inputs and
        own['extraction'] == extracted and own['policy'] == POLICY, 'own frozen input/source/extraction policy differs')
    selected, packets, saved_roles = population(paths)
    require(own['fixed_identity'] == [dict(index=r['index'], id=r['id'], family=r['family'], partition=r['partition']) for r in selected],
            'own fixed CAL identity differs before replay')
    own_binding = binding(out/OUTPUTS[0])
    write_new(out/OUTPUTS[1], dict(schema='supplemental_Base_source_CAL_role_replay_started_v1',
        protocol=own_binding, inputs=inputs, extraction=extracted, policy=POLICY,
        fixed_frames=128, original_pose_witness_missing=True, historical_pose_identity_verified=False))
    counts = Counter(); primitive = Counter({k: 0 for k in PRIMITIVES}); statuses = Counter(); roles_count = Counter()
    discrepancies, attempted, cleanup_errors = [], [], []
    failure, finished, active = None, False, None
    began = time.monotonic(); cv2 = None; originals = {}; old_threads = None; writer = None
    try:
        import cv2
        import numpy as np
        from ..pallet_observation_refiner_20261009_v1 import solver as S
        require('torch' not in sys.modules, 'model/Torch library already imported in standalone source role replay')
        old_threads = cv2.getNumThreads(); cv2.setNumThreads(1)
        features = np.load(paths['features'], mmap_mode='r')
        require(list(features.shape) == POLICY['feature_shape'] and str(features.dtype) == 'float16', 'unchanged feature array header differs')
        originals = {name: getattr(cv2, name) for name in PRIMITIVES}
        for name, fn in originals.items():
            def counted(*a, _name=name, _fn=fn, **kw):
                primitive[_name] += 1
                result = _fn(*a, **kw)
                if _name == 'convexHull' and active is not None:
                    active['hull_calls'].append(safe(result))
                return result
            setattr(cv2, name, counted)
        class CountedBank(S.HypothesisBank):
            def __init__(self, points, K, xyz, image_size):
                counts['legacy_bank_constructor_entries'] += 1
                require(active is not None and np.array_equal(np.asarray(points), active['points'], equal_nan=True) and
                    np.array_equal(K, active['K']) and np.array_equal(xyz, active['dims']) and
                    image_size == (active['hw'][1], active['hw'][0]), 'nonwhitelisted numeric information entered role bank')
                super().__init__(points, K, xyz, image_size=image_size)
                active['bank'] = self
            def solve(self, *a, **kw):
                require(not a and kw == {'robust': False}, 'old role function ordinary route changed')
                counts['legacy_standard_role_solve_entries'] += 1
                result = super().solve(*a, **kw)
                counts['legacy_standard_role_solve_returns'] += 1
                active['pose_return'] = result
                return result
        def visibility(*a, **kw):
            counts['unchanged_visibility_helper_entries'] += 1
            result = S.visibility(*a, **kw)
            active['visibility_proxy_witness'] = dict(hidden=safe(result[0]), visible=safe(result[1]), cosines=safe(result[2]))
            return result
        namespace = dict(cv2=cv2, np=np, EDGES=edges, HypothesisBank=CountedBank, visibility=visibility,
                         __package__='scripts.research.pallet_observation_refiner_20261009_v1')
        exec(compile(module, str(paths['model_code']) + ':unchanged_initial_geometry_AST', 'exec'), namespace)
        initial_geometry = namespace['initial_geometry']
        raw_path = out/OUTPUTS[3]
        writer = gzip.open(raw_path, 'xt', encoding='utf-8', compresslevel=6)
        with file_canary([paths['features']], attempted):
            for record in selected:
                i = record['index']; p = packets[i]
                require(set(p) == set(WHITELIST), 'geometry input packet is not exact whitelist')
                active = dict(index=i, id=record['id'], points=np.asarray(p['Base_points'], float),
                    K=np.asarray(p['K'], float), dims=np.asarray(p['dimensions'], float), hw=p['raw_hw'],
                    hull_calls=[], visibility_proxy_witness=None)
                before = primitive.copy()
                counts['unchanged_initial_geometry_entries'] += 1
                pose, role, H = initial_geometry(active['points'], active['K'], active['dims'], active['hw'])
                counts['unchanged_initial_geometry_returns'] += 1
                require(role.shape == (12, 3), 'unchanged role output shape')
                expected = np.repeat(role, 7, axis=0)[:, :, None]*np.ones((1, 1, 65), np.float32)
                expected = expected.astype(np.float16)
                cached = np.array(features[i, :, 25:28, :])
                mismatch = np.flatnonzero(np.any(cached != expected, axis=(1, 2))).tolist()
                one_hot = bool(np.all((cached == 0) | (cached == 1)) and np.all(cached.sum(axis=1) == 1))
                constant_bins = bool(np.all(cached == cached[:, :, 32:33]))
                cache_ids = cached[:, :, 32].argmax(1).tolist()
                edge_constant = all(len(set(cache_ids[e*7:(e+1)*7])) == 1 for e in range(12))
                old_id_match = cache_ids == saved_roles[i]
                counts['frames_complete'] += 1
                counts['queries_checked'] += 84
                counts['role_channel_values_checked'] += 84*3*65
                counts['cached_role_replay_mismatched_queries'] += len(mismatch)
                counts['cached_roles_not_one_hot_frames'] += int(not one_hot)
                counts['cached_roles_vary_by_bin_frames'] += int(not constant_bins)
                counts['cached_roles_vary_within_edge_frames'] += int(not edge_constant)
                counts['old_cached_role_ID_mismatched_frames'] += int(not old_id_match)
                statuses['available' if pose['available'] else 'unavailable'] += 1
                roles_count.update(cache_ids)
                if mismatch or not (one_hot and constant_bins and edge_constant and old_id_match):
                    discrepancies.append(dict(index=i, id=record['id'], mismatched_queries=mismatch,
                        cached_one_hot=one_hot, constant_across_bins=constant_bins, constant_within_edge=edge_constant,
                        original_cached_role_ID_match=old_id_match))
                witness = dict(index=i, id=record['id'], family=record['family'], partition='calibration',
                    geometry_input=copy.deepcopy(p), pose_replay=safe(pose), predicted_roles12=safe(role),
                    replay_role_query_ids=np.repeat(role, 7, axis=0).argmax(1).tolist(), cached_role_query_ids=cache_ids,
                    original_saved_role_query_ids=saved_roles[i], all_65_bins_role_channels_match=not mismatch,
                    mismatch_query_ids=mismatch, cached_one_hot=one_hot, constant_across_bins=constant_bins,
                    constant_within_edge=edge_constant, original_cached_role_ID_match=old_id_match,
                    replay_H=list(H), hull_calls=active['hull_calls'], visibility_proxy_witness=active['visibility_proxy_witness'],
                    replay_bank_dimensions=safe(active['bank'].dims), replay_bank_names=active['bank'].names,
                    full_legacy_bank_ledger=dict(active['bank'].ledger),
                    actual_OpenCV_entry_counts={k: primitive[k]-before[k] for k in PRIMITIVES},
                    original_pose_witness_missing=True, historical_pose_identity_verified=False,
                    source_GT_pose_used_as_role_input=False, initial_geometry_body_unmodified=True,
                    physical_boundary_or_global_pose_accuracy_certified=False)
                writer.write(json.dumps(witness, ensure_ascii=False, separators=(',', ':'), allow_nan=False)+'\n')
                active = None
        writer.close(); writer = None
        with raw_path.open('rb') as stream:
            os.fsync(stream.fileno())
        require(counts['frames_complete'] == counts['unchanged_initial_geometry_entries'] ==
            counts['unchanged_initial_geometry_returns'] == counts['legacy_bank_constructor_entries'] ==
            counts['legacy_standard_role_solve_entries'] == counts['legacy_standard_role_solve_returns'] == 128,
            'exact fixed128 ordinary role replay calls required')
        require(primitive['cornerSubPix'] == 0 and not attempted and 'torch' not in sys.modules,
                'unexpected subpixel/image/model/depth work')
        require({k: binding(p) for k, p in paths.items()} == inputs and binding(out/OUTPUTS[0]) == own_binding,
                'byte-bound inputs/code/protocol changed during replay')
        finished = True
    except BaseException as error:
        partial = None
        if active is not None:
            partial = {k: safe(v) for k, v in active.items() if k != 'bank'}
            if active.get('bank') is not None:
                partial['partial_legacy_bank_ledger'] = dict(active['bank'].ledger)
        failure = dict(type=type(error).__name__, message=str(error), active_index=active.get('index') if active else None,
                       active_partial_witness=partial)
    finally:
        if writer is not None:
            try:
                writer.close()
            except BaseException as error:
                cleanup_errors.append(dict(type=type(error).__name__, message=str(error)))
        if cv2 is not None:
            for name, fn in originals.items():
                try:
                    setattr(cv2, name, fn)
                except BaseException as error:
                    cleanup_errors.append(dict(type=type(error).__name__, message=str(error)))
            if old_threads is not None:
                try:
                    cv2.setNumThreads(old_threads)
                except BaseException as error:
                    cleanup_errors.append(dict(type=type(error).__name__, message=str(error)))
    complete = finished and failure is None and not cleanup_errors
    passed = complete and not discrepancies
    raw_binding = binding(out/OUTPUTS[3]) if (out/OUTPUTS[3]).is_file() else None
    write_new(out/OUTPUTS[2], dict(schema='supplemental_Base_source_CAL_role_replay_checks_v1',
        complete=complete, passed=passed, protocol=own_binding, inputs=inputs, extraction=extracted,
        policy=POLICY, counts=dict(counts), actual_OpenCV_entry_counts=dict(primitive), pose_statuses=dict(statuses),
        cached_role_query_counts=dict(roles_count), discrepancies=discrepancies, error=failure,
        cleanup_errors=cleanup_errors, attempted_forbidden_reads=attempted, rows=raw_binding,
        original_pose_witness_missing=True, historical_pose_identity_verified=False,
        source_metadata_only_whitelist_retained=True, source_GT_pose_used_as_role_input=False,
        new_model_detector_head_image_GT_depth_ray_N3_training_RGB_calls=0,
        full_1024_target_wire_contract_repeated=False, target_supervision_checked=False,
        original_inputs_preserved=complete, partial_or_complete_raw_rows_preserved_without_overwrite=True,
        wall_seconds=time.monotonic()-began, latency_benchmark=False, automatic_retry=False))
    print('SOURCE_ROLE_REPLAY', 'PASS' if passed else 'FAIL', dict(counts), flush=True)
    if not passed:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('freeze', 'run'))
    parser.add_argument('--output', type=Path, default=DOC)
    parser.add_argument('--cache', type=Path, default=CACHE)
    parser.add_argument('--metadata', type=Path, default=REPAIR/'READY_SOURCE_TARGET_ROWS.jsonl.gz')
    parser.add_argument('--ceiling', type=Path, default=DIFFICULTY/'SOURCE_CEILING_ROWS.jsonl.gz')
    args = parser.parse_args()
    (freeze if args.stage == 'freeze' else run)(args)


if __name__ == '__main__':
    main()
