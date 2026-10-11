"""Own-frozen evaluation of two fresh V19 solves on the unchanged DEV245 proxy.

The complete V13 score function and V7 statistics kernels remain unchanged.
Only paired topology and geometry filenames change in an isolated namespace.
Existing Base/N3, V17 ordinary and V18 restored scores are never rescored.
All computation and private reference access belong to ROOT after full PASS.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import copy
import csv
import gzip
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
from types import SimpleNamespace

PRIVATE = Path('/tmp/pallet-finite-support-orthogonal-private-20261011-v19')
NEW_METHODS = ('HYBRID_OBSERVED_FRAME_FACTOR_SUBSET_CONSENSUS', 'HYBRID_OBSERVED_FRAME_POINT_ONLY')
PRIMARY, POINT_ONLY = NEW_METHODS
ORDINARY_METHODS = ('HYBRID_STANDARD_PNP', 'NATIVE_STANDARD_PNP')
OLD_POINT = 'HYBRID_POINT_CONSENSUS_RESTORED'
METHODS = ('BASE', 'N3_SUBPIX', *ORDINARY_METHODS, OLD_POINT, *NEW_METHODS)
CONTRASTS = ((PRIMARY, POINT_ONLY), (PRIMARY, 'N3_SUBPIX'), (PRIMARY, ORDINARY_METHODS[1]),
             (PRIMARY, ORDINARY_METHODS[0]), (PRIMARY, OLD_POINT), (POINT_ONLY, OLD_POINT))
ARITHMETIC = dict(saved_scored_rows=1715, source_score_rows_streamed=1960,
    historical_original_rows_ignored=245, newly_scored_rows=490, posthoc_rows=490,
    methods=7, strata=3, moment_distributions=189, inherited_moment_scalars=1134,
    additional_population_moment_scalars=378, paired_groups=54, metric_CIs=162,
    supplemental_paired_matched_distributions=486, supplemental_paired_inherited_moment_scalars=2916,
    supplemental_paired_population_moment_scalars=972,
    new_bootstrap_draws=0)


def require(value, message):
    if not value: raise ValueError(message)


def read(path): return json.loads(Path(path).read_bytes())


def binding(path):
    path = Path(path)
    require(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)), 'missing/symlink input: ' + str(path))
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''): digest.update(block)
    return dict(path=str(path.resolve()), bytes=path.stat().st_size, sha256=digest.hexdigest())


def bound(path, expected, label):
    actual = binding(path)
    require(all(actual[k] == expected[k] for k in ('bytes', 'sha256')), label + ' SHA/bytes differ')


def output(args, name):
    require(name == Path(name).name and name not in ('', '.', '..'), 'new output basename required')
    out = Path(args.output)
    require(out.is_absolute() and not any(p.is_symlink() for p in (out, *out.parents)), 'absolute non-symlink output required')
    require(out.resolve().is_relative_to(PRIVATE.resolve()) and out.resolve() != PRIVATE.resolve(), 'new private evaluation subdirectory required')
    for dependency in (args.repo, args.source_root, args.baseline_root, args.geometry_input, args.validation_input,
                       args.parent, args.ordinary_parent, args.point_parent, args.v9_input, args.fits, args.first_geometry_input):
        root = Path(dependency).resolve()
        require(not out.resolve().is_relative_to(root) and not root.is_relative_to(out.resolve()), 'output overlaps protected input')
    target = out / name
    require(not target.exists() and not target.is_symlink(), 'preserve ' + name)
    return target


def protection(args):
    def snapshot(root):
        def command(*values): return subprocess.check_output(['git', '-C', str(root), *values])
        return dict(HEAD=command('rev-parse', 'HEAD').decode().strip(),
            index_sha256=hashlib.sha256(command('ls-files', '--stage', '-z')).hexdigest(),
            tracked_diff_sha256=hashlib.sha256(command('diff', '--binary')).hexdigest(),
            status_sha256=hashlib.sha256(command('status', '--porcelain=v1', '-uall')).hexdigest())
    return dict(research=snapshot(args.repo), original=snapshot(args.source_root))


def configure(args):
    """CLI supplies split roots before importing existing source definitions."""
    global C, E13, E15, E18, S, PARITY
    sys.dont_write_bytecode = True
    repo = Path(args.repo).resolve(); sys.path.insert(0, str(repo))
    C = importlib.import_module('scripts.research.pallet_sparse_local_line_20261010_v9.common')
    E13 = importlib.import_module('scripts.research.pallet_finite_edge_consensus_repair_20261011_v13.evaluation')
    E15 = importlib.import_module('scripts.research.pallet_factor_subset_consensus_20261011_v15.evaluation')
    E18 = importlib.import_module('scripts.research.pallet_consensus_decision_repair_20261011_v18.evaluation')
    S = importlib.import_module('scripts.research.pallet_three_head_observation_20261010_v7.statistics')
    PARITY = importlib.import_module('scripts.research.pallet_sparse_local_line_20261010_v9.runtime_review')
    require(C.REPO.resolve() == repo, 'source package resolved a different public repository')
    args.protocol = str(Path(args.geometry_input) / 'GEOMETRY_PROTOCOL.json')


def geometry_paths(args):
    g, v = Path(args.geometry_input), Path(args.validation_input)
    paths = {n: g / n for n in ('GEOMETRY_PROTOCOL.json', 'GEOMETRY_STARTED.json', 'GEOMETRY_RECEIPT.json', 'GEOMETRY.jsonl.gz', 'LEDGERS.jsonl.gz')}
    paths.update({n: v / n for n in ('VALIDATION_PROTOCOL.json', 'VALIDATION_STARTED.json', 'VALIDATION_CHECKS.json')})
    paths.update(checker=PRIVATE / 'validation.py', ready_proof=PRIVATE / 'GEOMETRY_READY_PROOF.json',
        ready_driver=PRIVATE / 'run_geometry_ready.py', original_driver=PRIVATE / 'run_geometry.py')
    for n in ('GEOMETRY_PROTOCOL.json', 'GEOMETRY_STARTED.json', 'GEOMETRY_RECEIPT.json'):
        paths['first:' + n] = Path(args.first_geometry_input) / n
    return paths


def geometry_gate(args):
    """Four-tuple compatible with the unchanged historical score function.

    The receipt is the completed-stream authority; it is not renamed a seal.
    No raw row, numerical geometry or private reference is opened by this gate.
    """
    p = geometry_paths(args)
    core, started, receipt = [read(p[n]) for n in ('GEOMETRY_PROTOCOL.json', 'GEOMETRY_STARTED.json', 'GEOMETRY_RECEIPT.json')]
    own, start, check = [read(p[n]) for n in ('VALIDATION_PROTOCOL.json', 'VALIDATION_STARTED.json', 'VALIDATION_CHECKS.json')]
    require(core['schema'] == 'observed_frame_repair_geometry_protocol_v19' and core['methods'] == list(NEW_METHODS)
        and core['frames'] == core['ledger_rows'] == 245 and core['rows'] == 490 and core['limits']['GT_allowed'] is False,
        'fixed GT-free paired245 geometry protocol required')
    require(core['environment'] == dict(PALLET_SOURCE_ROOT=str(Path(args.source_root).resolve()),
        PALLET_BASELINE_ROOT=str(Path(args.baseline_root).resolve())), 'same explicit source/baseline roots as actual geometry')
    require(receipt['schema'] == 'observed_frame_repair_geometry_receipt_v19' and receipt['complete'] is True
        and receipt['error'] is None and not receipt['cleanup_errors'] and receipt['inputs_unchanged'] is True
        and receipt['environment_restored'] is True and receipt['inputs'] == core['inputs'] == started['inputs']
        and receipt['protection_before'] == receipt['protection_after'] == core['protection']
        and receipt['actual_execution']['frames_complete'] == receipt['actual_execution']['ledger_rows'] == 245
        and receipt['actual_execution']['geometry_rows'] == 490, 'completed490/245 with restored cleanup required')
    for name, item in (('GEOMETRY_PROTOCOL.json', receipt['protocol']), ('GEOMETRY_PROTOCOL.json', started['protocol']),
                       ('GEOMETRY.jsonl.gz', receipt['geometry']), ('LEDGERS.jsonl.gz', receipt['ledgers'])):
        bound(p[name], item, 'completed geometry ' + name)
    require(own['schema'] == 'observed_frame_hybrid_scalar_protocol_v19' and check['schema'] == 'observed_frame_hybrid_scalar_checks_v19'
        and check['complete'] is True and check['passed'] is True and check['failure_count'] == 0 and check['exception'] is None
        and check['new_production_calls'] == 0 and check['inputs'] == own['inputs'] == start['inputs']
        and start['new_production_calls'] == 0, 'actual independent full V19 geometry PASS required')
    require(all(check['actual_counts'][k] == n for k, n in (('geometry_rows', 490), ('ledger_rows', 245), ('frames', 245))), 'full245 checker coverage required')
    require(check['old_full_V16_validation_passed'] is False and check['limits']['old_full_V16_564_FAIL_remains_FAIL'] is True,
        'new PASS does not reclassify original full564 FAIL')
    for item in (check['own_protocol'], start['own_protocol']): bound(p['VALIDATION_PROTOCOL.json'], item, 'actual checker protocol')
    bound(p['checker'], check['checker'], 'current independent checker')
    bound(p['checker'], own['checker'], 'own checker authority')
    for alias, name in (('geometry', 'GEOMETRY.jsonl.gz'), ('ledgers', 'LEDGERS.jsonl.gz'), ('geometry_protocol', 'GEOMETRY_PROTOCOL.json'),
                        ('geometry_started', 'GEOMETRY_STARTED.json'), ('geometry_receipt', 'GEOMETRY_RECEIPT.json')):
        bound(p[name], own['inputs'][alias], 'independently checked ' + alias)
    bound(args.cohort, own['inputs']['cohort'], 'unchanged cohort')
    for key, expected in core['inputs'].items():
        if key.startswith('source:'):
            bound(Path(args.repo) / key.removeprefix('source:'), expected, 'CPU/geometry-frozen source ' + key)
    require(core['ordered_ids'] == read(args.cohort)['ids'] and len(core['ordered_ids']) == len(set(core['ordered_ids'])) == 245, 'ordered full245 cohort')
    first = read(p['first:GEOMETRY_RECEIPT.json'])
    require(first['complete'] is False and first['error']['type'] == 'TypeError' and first['actual_execution'].get('geometry_rows', 0) == 0,
        'original zero-row serialization failure remains preserved')
    return core, receipt, receipt, check


def control_authority(args):
    """Existing scored controls are byte-bound; their references are not reopened."""
    E18.ordinary_authority(args)
    fixed = read(Path(args.v9_input) / 'SCORING_RECEIPT.json')
    require(fixed['complete'] is True and fixed['frames'] == 245 and fixed['fixed_control_rows'] == 490, 'existing fixed Base/N3 scores')
    bound(Path(args.v9_input) / 'FIXED_PREDICTIONS.jsonl.gz', fixed['fixed_predictions'], 'fixed original scores')
    folder = Path(args.point_parent)
    check, own, start, metrics = [read(folder / n) for n in ('VERIFICATION.json', 'STATISTICS_VERIFICATION_PROTOCOL.json', 'STATISTICS_VERIFICATION_STARTED.json', 'METRICS.json')]
    require(check['schema'] == 'independent_point_consensus_decision_statistics_checks_v18' and check['complete'] is True and check['passed'] is True
        and check['failure_count'] == 0 and check['error'] is None and check['inputs'] == own['inputs'] == start['inputs']
        and check['actual_counts']['saved_scored_rows_streamed'] == 1470 and check['actual_counts']['moment_scalars_checked'] == 972
        and check['actual_counts']['metric_CI_slots_checked'] == 135 and metrics['complete'] is True, 'existing V18 independent1470/972/135 PASS')
    for item in (check['protocol'], start['protocol']): bound(folder / 'STATISTICS_VERIFICATION_PROTOCOL.json', item, 'V18 verifier protocol')
    bound(folder / 'METRICS.json', check['inputs']['METRICS'], 'V18 independently checked metrics')
    bound(folder / 'PREDICTIONS.jsonl.gz', check['inputs']['new_scores'], 'V18 independently checked paired scores')
    require(metrics['V16_full_scalar_validation_passed'] is False, 'historical full564 FAIL is not recertified')
    for alias, path in (('fixed_scores', Path(args.v9_input) / 'FIXED_PREDICTIONS.jsonl.gz'), ('ordinary_scores', Path(args.ordinary_parent) / 'PREDICTIONS.jsonl.gz'),
                        ('cohort', Path(args.cohort)), ('same_role_observations', Path(args.parent) / 'OBSERVATIONS.jsonl.gz')):
        bound(path, metrics['inputs'][alias], 'same old point comparator ' + alias)


def isolated_score(args):
    original = next(n for n in ast.parse(Path(E13.__file__).read_bytes()).body if isinstance(n, ast.FunctionDef) and n.name == 'score')
    canonical = ast.dump(original, include_attributes=False)
    target, replacement = ast.parse('from .pipeline import METHODS').body[0], ast.parse('METHODS = _new_METHODS').body[0]
    names = {'GEOMETRY_SEALED.jsonl.gz': 'GEOMETRY.jsonl.gz', 'GEOMETRY_SEAL.json': 'GEOMETRY_RECEIPT.json', 'PARENT_POPULATION_CHECKS.json': 'GEOMETRY_PROTOCOL.json'}
    score_input = ast.parse("R.DOC / 'PREDICTIONS.jsonl.gz'", mode='eval').body
    score_output = ast.parse("Path(args.output) / 'PREDICTIONS.jsonl.gz'", mode='eval').body
    expected = dict(local_method_import=1, configured_scored_rows=1, scored_rows=1, completion_count=1, score_output_binding=1,
                    **{'filename:' + k: n for k, n in (('GEOMETRY_SEALED.jsonl.gz', 1), ('GEOMETRY_SEAL.json', 2), ('PARENT_POPULATION_CHECKS.json', 1))})
    def transform(node, reverse=False):
        changes = Counter(); mapping = {v: k for k, v in names.items()} if reverse else names
        class Adapter(ast.NodeTransformer):
            def visit_BinOp(self, value):
                before, after = (score_output, score_input) if reverse else (score_input, score_output)
                if ast.dump(value, include_attributes=False) == ast.dump(before, include_attributes=False):
                    changes['score_output_binding'] += 1; return copy.deepcopy(after)
                return self.generic_visit(value)
            def visit_ImportFrom(self, value):
                if not reverse and ast.dump(value, include_attributes=False) == ast.dump(target, include_attributes=False):
                    changes['local_method_import'] += 1; return copy.deepcopy(replacement)
                return self.generic_visit(value)
            def visit_Assign(self, value):
                if reverse and ast.dump(value, include_attributes=False) == ast.dump(replacement, include_attributes=False):
                    changes['local_method_import'] += 1; return copy.deepcopy(target)
                return self.generic_visit(value)
            def visit_keyword(self, value):
                if value.arg in ('configured_scored_rows', 'scored_rows') and isinstance(value.value, ast.Constant) and value.value.value == (490 if reverse else 245):
                    changes[value.arg] += 1; return ast.keyword(arg=value.arg, value=ast.Constant(245 if reverse else 490))
                return self.generic_visit(value)
            def visit_Compare(self, value):
                before, after = ('count==490', 'count==245') if reverse else ('count==245', 'count==490')
                if ast.dump(value, include_attributes=False) == ast.dump(ast.parse(before, mode='eval').body, include_attributes=False):
                    changes['completion_count'] += 1; return ast.parse(after, mode='eval').body
                return self.generic_visit(value)
            def visit_Constant(self, value):
                if isinstance(value.value, str) and value.value in mapping:
                    source = mapping[value.value] if reverse else value.value
                    changes['filename:' + source] += 1; return ast.Constant(mapping[value.value])
                return value
        return Adapter().visit(copy.deepcopy(node)), dict(changes)
    changed, changes = transform(original); restored, undo = transform(changed, True)
    require(changes == undo == expected and ast.dump(restored, include_attributes=False) == canonical, 'whole score AST inverse proof, topology/filenames only')
    facade = SimpleNamespace(DOC=Path(args.geometry_input), output=output, frozen=lambda a: geometry_gate(a),
        A=SimpleNamespace(protect=lambda a: protection(a)), audit_args=lambda a: a)
    namespace = dict(vars(E13)); namespace.update(R=facade, geometry_gate=geometry_gate, _new_METHODS=NEW_METHODS)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[changed], type_ignores=[])), str(E13.__file__) + ' [V19 topology only]', 'exec'), namespace)
    return namespace['score'], dict(original_source=binding(E13.__file__), exact_AST_topology_changes=expected,
        whole_canonical_function_restoration_proven=True, reference_and_error_formula_AST_changes=0,
        parent_module_globals_modified=False, proper_original_package_context_retained=True,
        historical_geometry_seal_field_refers_to_completed_GEOMETRY_RECEIPT=True)


def input_paths(args):
    paths = geometry_paths(args)
    paths.update(adapter=Path(__file__), cohort=Path(args.cohort), observations=Path(args.parent) / 'OBSERVATIONS.jsonl.gz',
        fixed_scores=Path(args.v9_input) / 'FIXED_PREDICTIONS.jsonl.gz', fixed_receipt=Path(args.v9_input) / 'SCORING_RECEIPT.json',
        ordinary_scores=Path(args.ordinary_parent) / 'PREDICTIONS.jsonl.gz', point_scores=Path(args.point_parent) / 'PREDICTIONS.jsonl.gz',
        bootstrap=S.BOOTSTRAP, unchanged_score=Path(E13.__file__), unchanged_statistics=Path(S.__file__),
        unchanged_compact=Path(E15.__file__), control_authority=Path(E18.__file__), unchanged_parity=Path(PARITY.__file__))
    paths.update({'ordinary:' + k: v for k, v in E18.ordinary_paths(args).items()})
    for n in ('SCORING_RECEIPT.json', 'METRICS.json', 'VERIFICATION.json', 'STATISTICS_VERIFICATION_PROTOCOL.json', 'STATISTICS_VERIFICATION_STARTED.json'):
        paths['point:' + n] = Path(args.point_parent) / n
    return paths


def freeze(args):
    geometry_gate(args); control_authority(args); _, dispatch = isolated_score(args)
    require(not Path(args.output).exists(), 'fresh evaluation output required')
    C.write_new(output(args, 'EVALUATION_PROTOCOL.json'), dict(schema='observed_frame_postseal_evaluation_protocol_v19',
        inputs={k: binding(p) for k, p in input_paths(args).items()}, methods=list(METHODS), new_methods=list(NEW_METHODS),
        contrasts=[list(c) for c in CONTRASTS], arithmetic=ARITHMETIC, score_dispatch=dispatch, protection=protection(args),
        source_root=str(Path(args.source_root).resolve()), baseline_root=str(Path(args.baseline_root).resolve()),
        reference='unchanged N3-phase GEOMETRIC_PROXY known DEV; not independently measured physical pose truth',
        sample_moments_definition='unchanged V7 ddof=1; population fields additionally named ddof=0',
        point_reproduction_atol=1e-7, point_reproduction_rtol=0, new_seed_or_training_or_RGB=0,
        frozen_before_new_GT_access_and_statistical_arithmetic=True, old_full564_scalar_FAIL_remains_FAIL=True))


def frozen(args):
    geometry_gate(args); control_authority(args); own = read(Path(args.output) / 'EVALUATION_PROTOCOL.json'); _, dispatch = isolated_score(args)
    require(own['schema'] == 'observed_frame_postseal_evaluation_protocol_v19' and own['inputs'] == {k: binding(p) for k, p in input_paths(args).items()}
        and own['methods'] == list(METHODS) and own['new_methods'] == list(NEW_METHODS) and own['contrasts'] == [list(c) for c in CONTRASTS]
        and own['score_dispatch'] == dispatch and own['protection'] == protection(args), 'frozen V19 evaluation source/control/protection differs')
    return own


def score(args):
    own = frozen(args); fn, dispatch = isolated_score(args); before = (E13.R, E13.geometry_gate, E13.R.P.METHODS) if hasattr(E13.R, 'P') else (E13.R, E13.geometry_gate)
    C.write_new(output(args, 'SCORING_AUTHORITY.json'), dict(schema='isolated_observed_frame_proxy_score_authority_v19',
        inputs=own['inputs'], own_evaluation_protocol=binding(Path(args.output) / 'EVALUATION_PROTOCOL.json'),
        methods=list(NEW_METHODS), score_dispatch=dispatch, reference_or_error_formula_changes=0, numerical_geometry_calls=0))
    try: fn(args)
    finally:
        after = (E13.R, E13.geometry_gate, E13.R.P.METHODS) if hasattr(E13.R, 'P') else (E13.R, E13.geometry_gate)
        require(before == after, 'immutable parent globals changed')
    frozen(args)


def score_gate(args):
    own = frozen(args); folder = Path(args.output)
    receipt, authority, start = [read(folder / n) for n in ('SCORING_RECEIPT.json', 'SCORING_AUTHORITY.json', 'SCORING_STARTED.json')]
    require(receipt['schema'] == 'repair_postseal_proxy_scoring_v13' and receipt['complete'] is True and receipt['scored_rows'] == 490
        and receipt['frames'] == 245 and receipt['methods'] == list(NEW_METHODS) and receipt['GT_access_only_after_complete_geometry_and_cleanup'] is True,
        'complete newly scored490 required')
    require(authority['schema'] == 'isolated_observed_frame_proxy_score_authority_v19' and authority['inputs'] == own['inputs']
        and authority['score_dispatch'] == own['score_dispatch'] and start['configured_scored_rows'] == 490
        and start['GT_only_after_full_seal_cleanup_and_independent_PASS'] is True, 'actual isolated score authority')
    for p, item in ((folder / 'EVALUATION_PROTOCOL.json', authority['own_evaluation_protocol']), (folder / 'PREDICTIONS.jsonl.gz', receipt['predictions']),
                    (args.protocol, receipt['protocol']), (Path(args.geometry_input) / 'GEOMETRY_RECEIPT.json', receipt['geometry_seal']),
                    (Path(args.validation_input) / 'VALIDATION_PROTOCOL.json', start['independent_validation'])):
        bound(p, item, 'actual score binding')
    return own, receipt


def posthoc(row, n3, observation):
    result = S.posthoc(dict(row, method='IMAGE_ROLE_CORNERWISE_HYBRID'), n3, observation)
    solver = row['solver']; new = bool(row['new_pose_estimated']); best = solver.get('best_candidate')
    result.update(method=row['method'], posthoc_kernel_label_adapter_only=True,
        finite_geometry_diagnostics=dict(solver_state=solver['state'], actual_fit_ids=solver['fit_input_ids'] if new else [],
            actual_fit_edges=solver['fit_line_edges'] if new else [], final8px_inliers=solver['final_inliers'] if new else [],
            stored_actual_inlier_geometry=copy.deepcopy(solver.get('point_line_geometry')) if new else None,
            stored_stable_incidence=copy.deepcopy(best.get('stable_incidence_guard')) if new and best else None,
            containment_frame='fixed observed tangent' if row['method'] == PRIMARY else 'no line factors',
            optimizer_and_decision_policy_from_geometry_not_changed_by_score=True,
            physical_ownership_certified=False, global_uniqueness_proven=False, NEW_is_not_accuracy_success=True))
    result['mask_definition_provenance'] = dict(score_known_excludes_UNKNOWN_UNANNOTATED=True,
        legacy_mask_audit=row.get('legacy_mask_audit_before_known_state_correction'),
        fallback_preserves_full_original_N3=bool(row['fallback_used']), approved_NEW_H_fit_exclusion=bool(new),
        scored_mask_applied_marker_is_attempt_contract_not_fallback_refit=True)
    return result


def summaries(rows):
    """Keep unchanged V7 moments and add explicitly separate population moments."""
    import numpy as np
    result = S.summaries(rows)
    selected = dict(operational=[r for r in rows if r['pose']['available']],
        new_pose=[r for r in rows if r['pose']['available'] and r['new_pose_estimated']],
        fallback=[r for r in rows if r['pose']['available'] and r['fallback_used']])
    for scope, values in selected.items():
        for key in S.METRICS:
            a = np.asarray([S.metric(r, key) for r in values], np.float64)
            result['metrics'][scope][key].update(population_variance=float(a.var(ddof=0)) if len(a) else None,
                                               population_std=float(a.std(ddof=0)) if len(a) else None)
    return result


def matched_moments(groups, candidate, comparator, selected):
    """Same recorded errors/deltas on the pair's exact common frame IDs."""
    import numpy as np
    result = {}
    for key in S.METRICS:
        a = [S.metric(groups[candidate][fid], key) for fid in selected]
        b = [S.metric(groups[comparator][fid], key) for fid in selected]
        scopes = {}
        for name, values in (('candidate', a), ('comparator', b), ('delta', [x-y for x, y in zip(a, b)])):
            values = np.asarray(values, np.float64); value = S.moments(values, S.UNITS[key])
            value.update(population_variance=float(values.var(ddof=0)) if len(values) else None,
                         population_std=float(values.std(ddof=0)) if len(values) else None)
            scopes[name] = value
        result[key] = scopes
    return result


def stats(args):
    import numpy as np
    own, _ = score_gate(args); paths = input_paths(args)
    paths.update(new_scores=Path(args.output) / 'PREDICTIONS.jsonl.gz', scoring_receipt=Path(args.output) / 'SCORING_RECEIPT.json',
        scoring_authority=Path(args.output) / 'SCORING_AUTHORITY.json', scoring_started=Path(args.output) / 'SCORING_STARTED.json',
        evaluation_protocol=Path(args.output) / 'EVALUATION_PROTOCOL.json')
    inputs = {k: binding(p) for k, p in paths.items()}
    for n in ('STATISTICS_STARTED.json', 'METRICS.json', 'METRICS.csv', 'POSTHOC_ROWS.jsonl.gz', 'POINT_REPRODUCTION.json', 'STATISTICS_FAILURE.json'): output(args, n)
    C.write_new(output(args, 'STATISTICS_STARTED.json'), dict(inputs=inputs, arithmetic=ARITHMETIC, methods=list(METHODS),
        contrasts=[list(c) for c in CONTRASTS], new_seed_draws=0, new_geometry_or_GT_score_calls=0))
    try:
        groups = {m: {} for m in METHODS}
        for key, allowed, expected in (('fixed_scores', METHODS[:2], 490), ('ordinary_scores', ORDINARY_METHODS, 490),
                                      ('point_scores', ('HYBRID_POINT_CONSENSUS_ORIGINAL', OLD_POINT), 490), ('new_scores', NEW_METHODS, 490)):
            count = 0; source_seen = set()
            for full in C.rows(paths[key]):
                row = E15.compact(full); count += 1
                require(row['method'] in allowed, 'fixed control stream method differs')
                identity = (row['method'], row['id']); require(identity not in source_seen, 'duplicate source method/ID'); source_seen.add(identity)
                if row['method'] not in groups: continue
                require(row['id'] not in groups[row['method']], 'duplicate method/ID'); groups[row['method']][row['id']] = row
            require(count == expected, 'full paired stream490 required: ' + key)
            require(all(sum(method == m for method, _ in source_seen) == 245 for m in allowed), 'all source control methods retain245')
        cohort = read(args.cohort); ids = cohort['ids']; labels = {r['id']: r['label'] for r in cohort['frames']}
        require(len(ids) == 245 and all(set(g) == set(ids) for g in groups.values()), 'all seven methods retain245')
        parity = []
        for fid in ids:
            hybrid, old, line, point = [groups[m][fid] for m in (ORDINARY_METHODS[0], OLD_POINT, *NEW_METHODS)]
            for key in ('input_points', 'hidden_initial', 'original_native_N3_points', 'selected_corner_ids', 'head_checkpoint_sha256', 'calibration_binding', 'head_arm', 'observation_supply'):
                require(hybrid[key] == old[key] == line[key] == point[key], 'same frozen hybrid input: ' + key)
            require(all(r['original_native_N3_points'] == groups['N3_SUBPIX'][fid]['native_points'] for r in (hybrid, old, line, point)), 'same N3 native phase')
            require(all(point[k] == old[k] for k in ('new_pose_estimated', 'fallback_used', 'output_status')), 'point-only reproduction status differs')
            pose = PARITY.pose_parity(point['actual_pose'], old['actual_pose'])
            p, q = np.asarray(point['native_points'], float), np.asarray(old['native_points'], float)
            require(p.shape == q.shape == (9, 2) and np.allclose(p, q, rtol=0, atol=pose['atol'], equal_nan=True), 'point-only display reproduction differs')
            parity.append(dict(id=fid, pose=pose, displayed_point_atol=pose['atol'], status_equal=True))
        with gzip.open(S.BOOTSTRAP, 'rt') as stream: bootstrap = json.load(stream)
        draws = np.asarray(bootstrap['counts'], np.int64); sessions = bootstrap['sessions']
        require(draws.shape == (10000, 13) and len(set(sessions)) == 13 and (draws >= 0).all() and (draws.sum(1) == 13).all(), 'unchanged13session10000draws')
        index = {s: i for i, s in enumerate(sessions)}
        observations = {r['id']: r for r in C.rows(paths['observations']) if r['head_arm'] == 'IMAGE_ROLE'}
        require(set(observations) == set(ids), 'same full ROLE observations')
        post = [posthoc(groups[m][fid], groups['N3_SUBPIX'][fid], observations[fid]) for m in NEW_METHODS for fid in ids]
        strata = {}
        for name, chosen in (('combined', ids), ('easy', [i for i in ids if labels[i] == 'clean']), ('medium', [i for i in ids if labels[i] == 'moderate'])):
            comparisons = {}
            for a, b in CONTRASTS:
                scopes = {scope: S.paired(groups[a], groups[b], chosen, index, draws,
                    'candidate_new_pose' if b == 'N3_SUBPIX' and scope == 'both_new_pose' else scope) for scope in S.SCOPES}
                if b == 'N3_SUBPIX':
                    scopes['both_new_pose'].update(scope='candidate_NEW_and_fixed_available',
                        fixed_control_NOT_relabelled_NEW=True, comparator_fixed_available=True)
                for value in scopes.values():
                    value['matched_recorded_error_and_delta_moments'] = matched_moments(groups, a, b, value['pair_ids'])
                comparisons[a + '_minus_' + b] = scopes
            strata[name] = dict(frames=len(chosen), ids=chosen, methods={m: summaries([groups[m][i] for i in chosen]) for m in METHODS},
                contrasts=comparisons, diagnostics={m: S.diagnostic_summary([r for r in post if r['method'] == m and r['id'] in set(chosen)]) for m in NEW_METHODS})
        def both_better(other):
            value = strata['combined']['contrasts'][PRIMARY + '_minus_' + other]['common_operational']
            return value['common_frames'] == 245 and all(value['metrics'][m]['mean_delta'] < 0 for m in ('translation_cm', 'rotation_deg'))
        require({k: binding(p) for k, p in paths.items()} == inputs, 'statistical inputs changed'); frozen(args)
        S.C.save_rows(output(args, 'POSTHOC_ROWS.jsonl.gz'), post)
        C.write_new(output(args, 'POINT_REPRODUCTION.json'), dict(complete=True, passed=True, frames=245, rows=parity,
            inherited_tolerance_source=binding(PARITY.__file__), atol=1e-7, rtol=0, physical_accuracy_claim=False))
        C.write_new(output(args, 'METRICS.json'), dict(schema='observed_frame_raw_row_metrics_v19', complete=True, inputs=inputs,
            methods=list(METHODS), contrasts=[list(c) for c in CONTRASTS], strata=strata, scopes=list(S.SCOPES), actual_arithmetic=ARITHMETIC,
            posthoc_rows=binding(Path(args.output) / 'POSTHOC_ROWS.jsonl.gz'), point_reproduction=binding(Path(args.output) / 'POINT_REPRODUCTION.json'),
            primary_both_T_R_means_better_than_N3_full245=both_better('N3_SUBPIX'), primary_both_T_R_means_better_than_point_full245=both_better(POINT_ONLY),
            reference=own['reference'], V16_full_scalar_validation_passed=False, new_geometry_or_models_or_GT_score_calls=0,
            population_variance_ddof=0, population_std_ddof=0, unchanged_sample_variance_ddof=1,
            all245_preserved_no_performance_filter=True, no_global_uniqueness_or_physical_ownership_claim=True))
        with output(args, 'METRICS.csv').open('x', newline='', encoding='utf-8') as stream:
            fields = ('stratum', 'method', 'population', 'metric', 'n', 'mean', 'sample_variance', 'sample_std', 'population_variance', 'population_std', 'median', 'P90', 'max', 'unit')
            writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
            for name, value in strata.items():
                for method, summary in value['methods'].items():
                    for scope, metrics in summary['metrics'].items():
                        for metric, moment in metrics.items(): writer.writerow(dict(stratum=name, method=method, population=scope, metric=metric, **moment))
        print('V19_METRICS_COMPLETE', both_better('N3_SUBPIX'), both_better(POINT_ONLY), flush=True)
    except BaseException as exc:
        C.write_new(output(args, 'STATISTICS_FAILURE.json'), dict(complete=False, inputs=inputs,
            error=dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc()), automatic_retry=False,
            output_prefix_preserved=True, new_geometry_or_models_or_GT_score_calls=0))
        raise


def parser():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('stage', choices=('freeze', 'score', 'stats'))
    p.add_argument('--repo', type=Path, required=True); p.add_argument('--source-root', type=Path, required=True); p.add_argument('--baseline-root', type=Path, required=True)
    p.add_argument('--geometry-input', type=Path, default=PRIVATE / 'geometry_ready'); p.add_argument('--validation-input', type=Path, default=PRIVATE / 'validation')
    p.add_argument('--first-geometry-input', type=Path, default=PRIVATE / 'geometry'); p.add_argument('--output', type=Path, default=PRIVATE / 'evaluation')
    p.add_argument('--fits', default='/dev/shm/pallet-kp-supervision-repair-private-20261010/learned_fits')
    p.add_argument('--cohort', type=Path); p.add_argument('--parent', type=Path); p.add_argument('--v9-input', type=Path)
    p.add_argument('--ordinary-parent', type=Path); p.add_argument('--point-parent', type=Path)
    return p


def main():
    args = parser().parse_args(); doc = args.repo.resolve() / '_docs/experiments'
    defaults = dict(cohort=doc / 'pallet_kp_corrected_supervision_20261010_v1/COHORT.json', parent=doc / 'pallet_three_head_observation_20261010_v7',
        v9_input=doc / 'pallet_sparse_local_line_20261010_v9', ordinary_parent=doc / 'pallet_hybrid_standard_controls_20261011_v17',
        point_parent=doc / 'pallet_consensus_decision_repair_20261011_v18')
    for k, value in defaults.items():
        if getattr(args, k) is None: setattr(args, k, value)
    environment = dict(PALLET_SOURCE_ROOT=str(args.source_root.resolve()), PALLET_BASELINE_ROOT=str(args.baseline_root.resolve()))
    saved = {k: os.environ.get(k) for k in environment}; os.environ.update(environment)
    try:
        configure(args); {'freeze': freeze, 'score': score, 'stats': stats}[args.stage](args)
    finally:
        for k, value in saved.items():
            if value is None: os.environ.pop(k, None)
            else: os.environ[k] = value


if __name__ == '__main__': main()
