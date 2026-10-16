"""One fixed V21 closure, scored after its own complete point geometry PASS.

Only the new 245 poses are proxy-scored. Seven actual frozen V19 controls and
existing session draws are replayed without new fit, seed, model or image calls.
ROOT alone executes this evaluator; metadata and score routing are isolated.
The historical full V16 scalar audit remains FAIL; NEW is numerical availability.
"""
from __future__ import annotations
import argparse
import ast
import builtins
from collections import Counter
from contextlib import contextmanager
import copy
import csv
import gzip
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
from types import SimpleNamespace

PRIVATE = Path('/tmp/pallet-whole-u-closure-private-20261011-v21')
HISTORY = Path('/tmp/pallet-finite-support-orthogonal-private-20261011-v19')
HISTORY_SOURCE_SHA = 'bad7bdf3271098b5728165fa8868fbb60cbcb13e04271115883094da1814dd2a'
HISTORY_STATS_SHA = '4cad27af50af1711ac1817e3c24cfddf73386b535f1cf8c30950950ef2ab1b0c'
PRIMARY = 'HYBRID_POINT_CONSENSUS_WHOLE_U_CLOSURE'
NEW_METHODS = (PRIMARY,)
ORDINARY_METHODS = ('HYBRID_STANDARD_PNP', 'NATIVE_STANDARD_PNP')
OLD_POINT = 'HYBRID_POINT_CONSENSUS_RESTORED'
HISTORY_NEW = ('HYBRID_OBSERVED_FRAME_FACTOR_SUBSET_CONSENSUS', 'HYBRID_OBSERVED_FRAME_POINT_ONLY')
OLD_METHODS = ('BASE', 'N3_SUBPIX', *ORDINARY_METHODS, OLD_POINT, *HISTORY_NEW)
METHODS = (*OLD_METHODS, PRIMARY)
CONTRASTS = tuple((PRIMARY, method) for method in OLD_METHODS)
ARITHMETIC = dict(saved_scored_rows=1960, source_score_rows_streamed=2205,
    historical_original_rows_ignored=245, newly_scored_rows=245, posthoc_rows=245,
    methods=8, strata=3, moment_distributions=216, inherited_moment_scalars=1296,
    additional_population_moment_scalars=432, paired_groups=63, metric_CIs=189,
    supplemental_paired_matched_distributions=567, supplemental_paired_inherited_moment_scalars=3402,
    supplemental_paired_population_moment_scalars=1134, total_distributions=783,
    total_moment_fields=6264, new_bootstrap_draws=0)

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

def protection(args):
    def snapshot(root):
        def command(*values): return subprocess.check_output(['git', '-C', str(root), *values])
        return dict(HEAD=command('rev-parse', 'HEAD').decode().strip(),
            index_sha256=hashlib.sha256(command('ls-files', '--stage', '-z')).hexdigest(),
            tracked_diff_sha256=hashlib.sha256(command('diff', '--binary')).hexdigest(),
            status_sha256=hashlib.sha256(command('status', '--porcelain=v1', '-uall')).hexdigest())
    return dict(research=snapshot(args.repo), original=snapshot(args.source_root))

def output(args, name):
    require(name == Path(name).name and name not in ('', '.', '..'), 'exclusive output basename')
    out = Path(args.output).absolute()
    require(not any(p.is_symlink() for p in (out, *out.parents)) and out.is_relative_to(PRIVATE)
        and out != PRIVATE, 'new non-symlink private V21 evaluation subdirectory')
    for dependency in (args.repo, args.source_root, args.baseline_root, args.geometry_input,
                       args.validation_input, args.parent, args.ordinary_parent, args.point_parent,
                       args.v9_input, args.fits, args.history_evaluation, args.history_validation):
        root = Path(dependency).resolve()
        require(not out.is_relative_to(root) and not root.is_relative_to(out), 'output overlaps protected input')
    target = out / name
    require(not target.exists() and not target.is_symlink(), 'preserve ' + name)
    return target

@contextmanager
def early_reference_canary(args):
    """Block source references/models/images until this evaluator opens its score phase."""
    allowed = [False]; attempted = []
    saved = [(builtins, 'open', builtins.open), (io, 'open', io.open), (Path, 'open', Path.open)]
    tokens = ('TARGETS.json', 'STATIC_VISIBILITY', 'GEOMETRY_RESOLVED_POSE_GT',
              'AXIS_REVIEW_MANIFEST', 'target_cache', 'target_arrays', 'keypoint_annotations')
    extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.pt', '.pth', '.onnx', '.npy', '.npz')
    fits = Path(args.fits).resolve()
    def guarded(fn):
        def call(path, *values, **keywords):
            if not allowed[0] and isinstance(path, (str, os.PathLike)):
                p = Path(path).absolute(); mode = str(keywords.get('mode', values[0] if values else 'r'))
                if ('r' in mode or '+' in mode) and p.suffix != '.py':
                    denied = p.is_relative_to(fits) or p.suffix.lower() in extensions
                    denied = denied or any(t.lower() in str(p).lower() for t in tokens)
                    if denied:
                        attempted.append(dict(path=str(p), mode=mode))
                        raise AssertionError('V21_EARLY_REFERENCE_ASSET_CANARY:' + p.name)
            return fn(path, *values, **keywords)
        return call
    args._reference_permission = allowed; args._reference_attempts = attempted
    for module, name, fn in saved: setattr(module, name, guarded(fn))
    try: yield
    finally:
        for module, name, fn in saved: setattr(module, name, fn)
        require(not attempted, 'forbidden early reference/model/image read attempted')

def configure(args):
    """CLI supplies split roots before importing existing source definitions."""
    global C, E13, E15, E18, S
    sys.dont_write_bytecode = True
    repo = Path(args.repo).resolve(); sys.path.insert(0, str(repo))
    C = importlib.import_module('scripts.research.pallet_sparse_local_line_20261010_v9.common')
    E13 = importlib.import_module('scripts.research.pallet_finite_edge_consensus_repair_20261011_v13.evaluation')
    E15 = importlib.import_module('scripts.research.pallet_factor_subset_consensus_20261011_v15.evaluation')
    E18 = importlib.import_module('scripts.research.pallet_consensus_decision_repair_20261011_v18.evaluation')
    S = importlib.import_module('scripts.research.pallet_three_head_observation_20261010_v7.statistics')
    require(C.REPO.resolve() == repo, 'source package resolved a different public repository')
    args.protocol = str(Path(args.geometry_input) / 'GEOMETRY_PROTOCOL.json')

def geometry_paths(args):
    g, v = Path(args.geometry_input), Path(args.validation_input)
    paths = {n: g / n for n in ('GEOMETRY_PROTOCOL.json', 'GEOMETRY_STARTED.json', 'GEOMETRY_RECEIPT.json', 'GEOMETRY.jsonl.gz', 'LEDGERS.jsonl.gz')}
    paths.update({n: v / n for n in ('VALIDATION_PROTOCOL.json', 'VALIDATION_STARTED.json', 'VALIDATION_CHECKS.json')})
    paths.update(checker=PRIVATE / 'validation.py', geometry_driver=PRIVATE / 'run_geometry.py',
        closure_solver=PRIVATE / 'solver.py', closure_pipeline=PRIVATE / 'pipeline.py')
    return paths

def geometry_gate(args):
    """No raw stream or reference is decoded until complete own point scalar PASS."""
    p = geometry_paths(args)
    core, started, receipt = [read(p[n]) for n in ('GEOMETRY_PROTOCOL.json', 'GEOMETRY_STARTED.json', 'GEOMETRY_RECEIPT.json')]
    own, start, check = [read(p[n]) for n in ('VALIDATION_PROTOCOL.json', 'VALIDATION_STARTED.json', 'VALIDATION_CHECKS.json')]
    require(core['schema'] == 'whole_U_closure_geometry_protocol_v21' and core['methods'] == list(NEW_METHODS)
        and core['frames'] == core['ledger_rows'] == core['rows'] == 245 and core['limits']['GT_allowed'] is False,
        'fixed GT-free full245 point geometry')
    require(core['environment'] == dict(PALLET_SOURCE_ROOT=str(Path(args.source_root).resolve()),
        PALLET_BASELINE_ROOT=str(Path(args.baseline_root).resolve())), 'actual geometry split-root environment')
    require(receipt['schema'] == 'whole_U_closure_geometry_receipt_v21' and receipt['complete'] is True
        and receipt['error'] is None and not receipt['cleanup_errors'] and receipt['inputs_unchanged'] is True
        and receipt['environment_restored'] is True and receipt['counter_dispatch_closed'] is True
        and all(value is True for value in receipt['restoration'].values())
        and receipt['inputs'] == core['inputs'] == started['inputs']
        and receipt['protection_before'] == receipt['protection_after'] == core['protection']
        and receipt['actual_execution'] == dict(frames_started=245, geometry_rows=245, ledger_rows=245, frames_complete=245),
        'actual245 cleanup and dispatch closure required')
    require(core['limits']['old564_scalar_FAIL_retained'] is True and receipt['limits'] == core['limits'] == started['limits'],
        'historical full564 FAIL retained')
    for name, item in (('GEOMETRY_PROTOCOL.json', receipt['protocol']), ('GEOMETRY_PROTOCOL.json', started['protocol']),
                       ('GEOMETRY.jsonl.gz', receipt['geometry']), ('LEDGERS.jsonl.gz', receipt['ledgers'])):
        bound(p[name], item, 'actual geometry ' + name)
    require(own['schema'] == 'whole_U_closure_point_scalar_protocol_v21'
        and check['schema'] == 'whole_U_closure_point_scalar_checks_v21'
        and check['complete'] is True and check['passed'] is True and check['failure_count'] == 0
        and check['exception'] is None and check['new_production_calls'] == start['new_production_calls'] == 0
        and check['inputs'] == own['inputs'] == start['inputs'], 'own independent full245 point scalar PASS')
    require(all(check['actual_counts'][key] == 245 for key in ('frames', 'geometry_rows', 'ledger_rows'))
        and check['old_full_V16_validation_passed'] is False, 'full245 point scope and preserved original fullFAIL')
    for item in (check['own_protocol'], start['own_protocol']): bound(p['VALIDATION_PROTOCOL.json'], item, 'actual own checker protocol')
    for item in (check['checker'], own['checker']): bound(p['checker'], item, 'current independent checker')
    for alias, name in (('geometry', 'GEOMETRY.jsonl.gz'), ('ledgers', 'LEDGERS.jsonl.gz'), ('geometry_protocol', 'GEOMETRY_PROTOCOL.json'),
                        ('geometry_started', 'GEOMETRY_STARTED.json'), ('geometry_receipt', 'GEOMETRY_RECEIPT.json')):
        bound(p[name], own['inputs'][alias], 'independently checked ' + alias)
    for key, expected in core['inputs'].items():
        checked = own['inputs']['geometry_input:' + key]
        bound(checked['path'], expected, 'current geometry-frozen input ' + key)
        require(all(checked[k] == expected[k] for k in ('bytes', 'sha256')), 'independent whole source closure ' + key)
    require(core['ordered_ids'] == read(args.cohort)['ids'] and len(core['ordered_ids']) == len(set(core['ordered_ids'])) == 245,
        'ordered full245 same cohort')
    bound(args.cohort, core['inputs']['cohort'], 'same actual geometry cohort')
    return core, receipt, receipt, check

def historical_parent_authority(args):
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

def history_paths(args):
    folder, validator = Path(args.history_evaluation), Path(args.history_validation)
    result = {'history:' + n: folder / n for n in ('METRICS.json', 'STATISTICS_STARTED.json', 'EVALUATION_PROTOCOL.json',
        'SCORING_RECEIPT.json', 'SCORING_AUTHORITY.json', 'SCORING_STARTED.json', 'PREDICTIONS.jsonl.gz',
        'POINT_REPRODUCTION.json')}
    result.update({'history_validation:' + n: validator / n for n in ('STATISTICS_CHECK_PROTOCOL.json',
        'STATISTICS_CHECK_STARTED.json', 'STATISTICS_CHECKS.json')})
    result.update(history_adapter=HISTORY / 'evaluate.py', history_statistics_checker=HISTORY / 'stats_check.py')
    return result

def control_authority(args):
    """Seven already scored controls, complete independently checked history, exact bytes."""
    historical_parent_authority(args)
    paths = history_paths(args); folder = Path(args.history_evaluation)
    m, e, s, score, authority, started = [read(folder / n) for n in ('METRICS.json', 'EVALUATION_PROTOCOL.json',
        'STATISTICS_STARTED.json', 'SCORING_RECEIPT.json', 'SCORING_AUTHORITY.json', 'SCORING_STARTED.json')]
    own, start, check = [read(Path(args.history_validation) / n) for n in ('STATISTICS_CHECK_PROTOCOL.json',
        'STATISTICS_CHECK_STARTED.json', 'STATISTICS_CHECKS.json')]
    require(m['schema'] == 'observed_frame_raw_row_metrics_v19' and m['complete'] is True
        and m['methods'] == e['methods'] == s['methods'] == list(OLD_METHODS)
        and m['V16_full_scalar_validation_passed'] is False, 'actual seven complete historical proxy controls')
    require(check['schema'] == 'independent_observed_frame_statistics_checks_v19'
        and own['schema'] == 'independent_observed_frame_statistics_protocol_v19'
        and check['complete'] is True and check['passed'] is True and check['error'] is None
        and check['new_production_calls'] == 0 and check['old_full_V16_validation_passed'] is False
        and check['limits']['old_full_V16_564_FAIL_remains_FAIL'] is True
        and check['inputs'] == own['inputs'] == start['inputs'], 'independently checked historical seven streams')
    expected = dict(saved_error_rows_streamed=1960, saved_error_rows_selected=1715, frames=245, methods=7,
        moment_distributions_checked=675, moment_scalars_checked=5400, paired_groups_checked=54, metric_CI_slots_checked=162)
    require(all(check['actual_counts'][k] == value for k, value in expected.items()), 'exact old675/5400/54/162 coverage')
    for item in (check['own_protocol'], start['own_protocol']): bound(paths['history_validation:STATISTICS_CHECK_PROTOCOL.json'], item, 'historical stats protocol')
    bound(paths['history_statistics_checker'], check['checker'], 'unchanged historical stats checker')
    require(binding(paths['history_statistics_checker'])['sha256'] == HISTORY_STATS_SHA
        and binding(paths['history_adapter'])['sha256'] == HISTORY_SOURCE_SHA, 'exact immutable history source')
    for alias, name in (('METRICS.json', 'METRICS.json'), ('STATISTICS_STARTED.json', 'STATISTICS_STARTED.json'),
        ('EVALUATION_PROTOCOL.json', 'EVALUATION_PROTOCOL.json'), ('SCORING_RECEIPT.json', 'SCORING_RECEIPT.json'),
        ('SCORING_AUTHORITY.json', 'SCORING_AUTHORITY.json'), ('SCORING_STARTED.json', 'SCORING_STARTED.json')):
        bound(folder / name, check['inputs'][alias], 'independently checked history ' + alias)
    bound(paths['history_adapter'], check['inputs']['adapter'], 'actual old evaluator')
    require(score['schema'] == 'repair_postseal_proxy_scoring_v13' and score['complete'] is True
        and score['frames'] == 245 and score['scored_rows'] == 490 and score['methods'] == list(HISTORY_NEW)
        and score['GT_access_only_after_complete_geometry_and_cleanup'] is True, 'actual old paired490 proxy score')
    bound(folder / 'PREDICTIONS.jsonl.gz', score['predictions'], 'old pair raw error stream')
    bound(folder / 'PREDICTIONS.jsonl.gz', check['inputs']['new_scores'], 'independently checked old pair raw stream')
    require(authority['schema'] == 'isolated_observed_frame_proxy_score_authority_v19'
        and authority['inputs'] == e['inputs'] and authority['score_dispatch'] == e['score_dispatch']
        and authority['reference_or_error_formula_changes'] == authority['numerical_geometry_calls'] == 0
        and started['configured_scored_rows'] == 490
        and started['GT_only_after_full_seal_cleanup_and_independent_PASS'] is True, 'old isolated unchanged score authority')
    bound(folder / 'EVALUATION_PROTOCOL.json', authority['own_evaluation_protocol'], 'old own score authority')
    require(m['inputs'] == s['inputs'] and all(m['inputs'][k] == item for k, item in e['inputs'].items()),
        'old complete evaluation closure remains in statistics')
    for alias, path in (('fixed_scores', Path(args.v9_input) / 'FIXED_PREDICTIONS.jsonl.gz'),
        ('ordinary_scores', Path(args.ordinary_parent) / 'PREDICTIONS.jsonl.gz'),
        ('point_scores', Path(args.point_parent) / 'PREDICTIONS.jsonl.gz'), ('bootstrap', S.BOOTSTRAP),
        ('cohort', args.cohort), ('observations', Path(args.parent) / 'OBSERVATIONS.jsonl.gz')):
        bound(path, m['inputs'][alias], 'same historical stream ' + alias)
    reproduction = read(folder / 'POINT_REPRODUCTION.json')
    require(reproduction['complete'] is True and reproduction['passed'] is True and reproduction['frames'] == 245,
        'historical point-only reproduction authority remains distinct from new candidate closure')
    bound(folder / 'POINT_REPRODUCTION.json', m['point_reproduction'], 'old point reproduction proof')
    return m

def isolated_score(args):
    original = next(n for n in ast.parse(Path(E13.__file__).read_bytes()).body if isinstance(n, ast.FunctionDef) and n.name == 'score')
    canonical = ast.dump(original, include_attributes=False)
    target, replacement = ast.parse('from .pipeline import METHODS').body[0], ast.parse('METHODS = _new_METHODS').body[0]
    names = {'GEOMETRY_SEALED.jsonl.gz': 'GEOMETRY.jsonl.gz', 'GEOMETRY_SEAL.json': 'GEOMETRY_RECEIPT.json', 'PARENT_POPULATION_CHECKS.json': 'GEOMETRY_PROTOCOL.json'}
    score_input = ast.parse("R.DOC / 'PREDICTIONS.jsonl.gz'", mode='eval').body
    score_output = ast.parse("Path(args.output) / 'PREDICTIONS.jsonl.gz'", mode='eval').body
    expected = dict(local_method_import=1, score_output_binding=1,
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
    exec(compile(ast.fix_missing_locations(ast.Module(body=[changed], type_ignores=[])), str(E13.__file__) + ' [V21 single-method routing only]', 'exec'), namespace)
    return namespace['score'], dict(original_source=binding(E13.__file__), exact_AST_topology_changes=expected,
        whole_canonical_function_restoration_proven=True, reference_and_error_formula_AST_changes=0,
        parent_module_globals_modified=False, proper_original_package_context_retained=True,
        historical_geometry_seal_field_refers_to_completed_GEOMETRY_RECEIPT=True)

def source_witness():
    source = HISTORY / 'evaluate.py'; require(binding(source)['sha256'] == HISTORY_SOURCE_SHA, 'immutable evaluator arithmetic source')
    old = ast.parse(source.read_bytes()); current = ast.parse(Path(__file__).read_bytes())
    def node(tree, name): return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    reused = ('summaries', 'matched_moments')
    require(all(ast.dump(node(old, n), include_attributes=False) == ast.dump(node(current, n), include_attributes=False)
        for n in reused), 'whole inherited moment function AST identity')
    old_control = copy.deepcopy(node(old, 'control_authority')); old_control.name = 'historical_parent_authority'
    require(ast.dump(old_control, include_attributes=False) == ast.dump(node(current, 'historical_parent_authority'), include_attributes=False),
        'whole historical control authority AST identity except its local function name')
    return dict(immutable_evaluator=binding(source), whole_unchanged_moment_functions=list(reused),
        inherited_sample_and_bootstrap_formula_AST_changes=0, old_controls_rescored_or_refit=False,
        single_fixed_algorithm_no_setting_search=True)

def input_paths(args):
    paths = geometry_paths(args)
    paths.update(adapter=Path(__file__), cohort=Path(args.cohort), observations=Path(args.parent) / 'OBSERVATIONS.jsonl.gz',
        fixed_scores=Path(args.v9_input) / 'FIXED_PREDICTIONS.jsonl.gz', fixed_receipt=Path(args.v9_input) / 'SCORING_RECEIPT.json',
        ordinary_scores=Path(args.ordinary_parent) / 'PREDICTIONS.jsonl.gz', point_scores=Path(args.point_parent) / 'PREDICTIONS.jsonl.gz',
        history_scores=Path(args.history_evaluation) / 'PREDICTIONS.jsonl.gz', bootstrap=S.BOOTSTRAP,
        unchanged_score=Path(E13.__file__), unchanged_statistics=Path(S.__file__), unchanged_compact=Path(E15.__file__),
        control_authority=Path(E18.__file__))
    paths.update(history_paths(args)); paths.update({'ordinary:' + k: v for k, v in E18.ordinary_paths(args).items()})
    for n in ('SCORING_RECEIPT.json', 'METRICS.json', 'VERIFICATION.json', 'STATISTICS_VERIFICATION_PROTOCOL.json', 'STATISTICS_VERIFICATION_STARTED.json'):
        paths['point:' + n] = Path(args.point_parent) / n
    checked = read(Path(args.validation_input) / 'VALIDATION_PROTOCOL.json')['inputs']
    for key in read(Path(args.geometry_input) / 'GEOMETRY_PROTOCOL.json')['inputs']:
        paths['geometry_input:' + key] = Path(checked['geometry_input:' + key]['path'])
    for namespace, protocol in (('validation_input:', Path(args.validation_input) / 'VALIDATION_PROTOCOL.json'),
                                ('history_input:', Path(args.history_evaluation) / 'EVALUATION_PROTOCOL.json')):
        for key, value in read(protocol)['inputs'].items(): paths[namespace + key] = Path(value['path'])
    return paths

def freeze(args):
    geometry_gate(args); control_authority(args); _, dispatch = isolated_score(args); witness = source_witness()
    require(not Path(args.output).exists(), 'fresh evaluator directory')
    C.write_new(output(args, 'EVALUATION_PROTOCOL.json'), dict(schema='whole_U_closure_postseal_evaluation_protocol_v21',
        inputs={k: binding(p) for k, p in input_paths(args).items()}, methods=list(METHODS), new_methods=list(NEW_METHODS),
        contrasts=[list(c) for c in CONTRASTS], arithmetic=ARITHMETIC, score_dispatch=dispatch, source_witness=witness,
        protection=protection(args), source_root=str(Path(args.source_root).resolve()), baseline_root=str(Path(args.baseline_root).resolve()),
        reference='unchanged N3-phase GEOMETRIC_PROXY known DEV; not independently measured physical pose truth',
        sample_moments_definition='unchanged V7 ddof=1; additionally named population variance/std ddof=0',
        no_old_control_rescore_or_fit=True, new_seed_or_training_or_RGB=0,
        early_reference_canary=dict(released_only_for_score_after_own_full_geometry_PASS=True,
            frozen_before_new_GT_access=True, blocked_early_attempts=list(args._reference_attempts)),
        frozen_before_new_GT_access_and_statistical_arithmetic=True, old_full564_scalar_FAIL_remains_FAIL=True))

def frozen(args):
    geometry_gate(args); control_authority(args); own = read(Path(args.output) / 'EVALUATION_PROTOCOL.json'); _, dispatch = isolated_score(args)
    require(own['schema'] == 'whole_U_closure_postseal_evaluation_protocol_v21'
        and own['inputs'] == {k: binding(p) for k, p in input_paths(args).items()}
        and own['methods'] == list(METHODS) and own['new_methods'] == list(NEW_METHODS)
        and own['contrasts'] == [list(c) for c in CONTRASTS] and own['arithmetic'] == ARITHMETIC
        and own['score_dispatch'] == dispatch and own['source_witness'] == source_witness()
        and own['protection'] == protection(args), 'frozen whole-U evaluation source/control/protection differs')
    return own

def score(args):
    own = frozen(args); fn, dispatch = isolated_score(args)
    before = (E13.R, E13.geometry_gate, E13.R.P.METHODS) if hasattr(E13.R, 'P') else (E13.R, E13.geometry_gate)
    C.write_new(output(args, 'SCORING_AUTHORITY.json'), dict(schema='isolated_whole_U_closure_proxy_score_authority_v21',
        inputs=own['inputs'], own_evaluation_protocol=binding(Path(args.output) / 'EVALUATION_PROTOCOL.json'),
        methods=list(NEW_METHODS), score_dispatch=dispatch, reference_or_error_formula_changes=0,
        numerical_geometry_calls=0, early_reference_canary_attempts_before_release=list(args._reference_attempts)))
    args._reference_permission[0] = True
    try: fn(args)
    finally:
        args._reference_permission[0] = False
        after = (E13.R, E13.geometry_gate, E13.R.P.METHODS) if hasattr(E13.R, 'P') else (E13.R, E13.geometry_gate)
        require(before == after, 'immutable parent globals changed')
    frozen(args)

def score_gate(args):
    own = frozen(args); folder = Path(args.output)
    receipt, authority, start = [read(folder / n) for n in ('SCORING_RECEIPT.json', 'SCORING_AUTHORITY.json', 'SCORING_STARTED.json')]
    require(receipt['schema'] == 'repair_postseal_proxy_scoring_v13' and receipt['complete'] is True and receipt['scored_rows'] == 245
        and receipt['frames'] == 245 and receipt['methods'] == list(NEW_METHODS)
        and receipt['GT_access_only_after_complete_geometry_and_cleanup'] is True, 'complete newly scored245')
    require(authority['schema'] == 'isolated_whole_U_closure_proxy_score_authority_v21' and authority['inputs'] == own['inputs']
        and authority['score_dispatch'] == own['score_dispatch'] and start['configured_scored_rows'] == 245
        and start['GT_only_after_full_seal_cleanup_and_independent_PASS'] is True, 'actual isolated score authority')
    for path, item in ((folder / 'EVALUATION_PROTOCOL.json', authority['own_evaluation_protocol']),
        (folder / 'PREDICTIONS.jsonl.gz', receipt['predictions']), (args.protocol, receipt['protocol']),
        (Path(args.geometry_input) / 'GEOMETRY_RECEIPT.json', receipt['geometry_seal']),
        (Path(args.validation_input) / 'VALIDATION_PROTOCOL.json', start['independent_validation'])):
        bound(path, item, 'actual score binding')
    return own, receipt

def posthoc(row, n3, observation):
    result = S.posthoc(dict(row, method='IMAGE_ROLE_CORNERWISE_HYBRID'), n3, observation)
    solver = row['solver']; new = bool(row['new_pose_estimated']); best = solver.get('best_candidate')
    result.update(method=row['method'], posthoc_kernel_label_adapter_only=True,
        finite_geometry_diagnostics=dict(solver_state=solver['state'], actual_fit_ids=solver['fit_input_ids'] if new else [],
            actual_fit_edges=solver['fit_line_edges'] if new else [], final8px_inliers=solver['final_inliers'] if new else [],
            stored_actual_inlier_geometry=copy.deepcopy(solver.get('point_line_geometry')) if new else None,
            stored_stable_incidence=copy.deepcopy(best.get('stable_incidence_guard')) if new and best else None,
            containment_frame='no line factors',
            closure_candidate_kind=best.get('proposal_kind') if new and best else None,
            closure_ledger_scope='exact raw4/refit plus live whole-U standard/LM candidates; no new independent observations',
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
    for n in ('STATISTICS_STARTED.json', 'METRICS.json', 'METRICS.csv', 'POSTHOC_ROWS.jsonl.gz', 'SAME_OBSERVATION_JOIN.json', 'STATISTICS_FAILURE.json'): output(args, n)
    C.write_new(output(args, 'STATISTICS_STARTED.json'), dict(inputs=inputs, arithmetic=ARITHMETIC, methods=list(METHODS),
        contrasts=[list(c) for c in CONTRASTS], new_seed_draws=0, new_geometry_or_GT_score_calls=0))
    try:
        groups = {m: {} for m in METHODS}
        for key, allowed, expected in (('fixed_scores', METHODS[:2], 490), ('ordinary_scores', ORDINARY_METHODS, 490),
                                      ('point_scores', ('HYBRID_POINT_CONSENSUS_ORIGINAL', OLD_POINT), 490), ('history_scores', HISTORY_NEW, 490), ('new_scores', NEW_METHODS, 245)):
            count = 0; source_seen = set()
            for full in C.rows(paths[key]):
                row = E15.compact(full); count += 1
                require(row['method'] in allowed, 'fixed control stream method differs')
                identity = (row['method'], row['id']); require(identity not in source_seen, 'duplicate source method/ID'); source_seen.add(identity)
                if row['method'] not in groups: continue
                require(row['id'] not in groups[row['method']], 'duplicate method/ID'); groups[row['method']][row['id']] = row
            require(count == expected, 'full configured source stream required: ' + key)
            require(all(sum(method == m for method, _ in source_seen) == 245 for m in allowed), 'all source control methods retain245')
        cohort = read(args.cohort); ids = cohort['ids']; labels = {r['id']: r['label'] for r in cohort['frames']}
        require(len(ids) == 245 and all(set(g) == set(ids) for g in groups.values()), 'all eight methods retain245')
        joins = []
        for fid in ids:
            hybrid = groups[ORDINARY_METHODS[0]][fid]
            same_q = [groups[m][fid] for m in (ORDINARY_METHODS[0], OLD_POINT, *HISTORY_NEW, PRIMARY)]
            fields = ('input_points', 'hidden_initial', 'original_native_N3_points', 'selected_corner_ids',
                      'head_checkpoint_sha256', 'calibration_binding', 'head_arm', 'observation_supply')
            for key in fields: require(all(row[key] == hybrid[key] for row in same_q), 'same frozen hybrid input: ' + key)
            require(all(row['original_native_N3_points'] == groups['N3_SUBPIX'][fid]['native_points'] for row in same_q), 'same original N3 native phase')
            require(groups[ORDINARY_METHODS[1]][fid]['input_points'] == groups['N3_SUBPIX'][fid]['native_points'], 'native ordinary uses original native q')
            require(len(groups[PRIMARY][fid]['native_points']) == 9
                and groups[PRIMARY][fid]['native_points'][8] == hybrid['native_points'][8], 'detector center preserved')
            joins.append(dict(id=fid, methods=[row['method'] for row in same_q], same_selected_q_H_and_native_phase=True,
                same_selected_corner_ids_head_CAL_and_supply=True, detector_center_preserved=True,
                numerical_pose_and_status_equality_not_required_for_new_coverage=True))
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
                    'candidate_new_pose' if b in METHODS[:2] and scope == 'both_new_pose' else scope) for scope in S.SCOPES}
                if b in METHODS[:2]:
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
        C.write_new(output(args, 'SAME_OBSERVATION_JOIN.json'), dict(complete=True, passed=True, frames=245, rows=joins,
            method_population='same hybrid q for ordinary hybrid, oldREST, V19 line/point and V21 closure only',
            native_ordinary_has_original_native_q=True, fixed_baseline_H_and_fit_metadata_unrecorded=True,
            numerical_parity_not_claimed=True, physical_accuracy_claim=False))
        C.write_new(output(args, 'METRICS.json'), dict(schema='whole_U_closure_raw_row_metrics_v21', complete=True, inputs=inputs,
            methods=list(METHODS), contrasts=[list(c) for c in CONTRASTS], strata=strata, scopes=list(S.SCOPES), actual_arithmetic=ARITHMETIC,
            posthoc_rows=binding(Path(args.output) / 'POSTHOC_ROWS.jsonl.gz'), same_observation_join=binding(Path(args.output) / 'SAME_OBSERVATION_JOIN.json'),
            primary_both_T_R_means_better_than_N3_full245=both_better('N3_SUBPIX'), primary_both_T_R_means_better_than_old_point_full245=both_better(HISTORY_NEW[1]),
            full245_T_R_mean_comparisons={method:both_better(method) for method in OLD_METHODS},
            historical_metrics=binding(Path(args.history_evaluation) / 'METRICS.json'),
            historical_statistics_checks=binding(Path(args.history_validation) / 'STATISTICS_CHECKS.json'),
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
        print('V21_METRICS_COMPLETE', both_better('N3_SUBPIX'), both_better(HISTORY_NEW[1]), flush=True)
    except BaseException as exc:
        C.write_new(output(args, 'STATISTICS_FAILURE.json'), dict(complete=False, inputs=inputs,
            error=dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc()), automatic_retry=False,
            output_prefix_preserved=True, new_geometry_or_models_or_GT_score_calls=0))
        raise

def parser():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('stage', choices=('freeze', 'score', 'stats'))
    p.add_argument('--repo', type=Path, required=True); p.add_argument('--source-root', type=Path, required=True)
    p.add_argument('--baseline-root', type=Path, required=True)
    p.add_argument('--geometry-input', type=Path, default=PRIVATE / 'geometry')
    p.add_argument('--validation-input', type=Path, default=PRIVATE / 'validation')
    p.add_argument('--output', type=Path, default=PRIVATE / 'evaluation')
    p.add_argument('--history-evaluation', type=Path, default=HISTORY / 'evaluation')
    p.add_argument('--history-validation', type=Path, required=True)
    p.add_argument('--fits', default='/dev/shm/pallet-kp-supervision-repair-private-20261010/learned_fits')
    for key in ('cohort', 'parent', 'v9-input', 'ordinary-parent', 'point-parent'): p.add_argument('--' + key, type=Path)
    return p

def main():
    args = parser().parse_args(); doc = args.repo.resolve() / '_docs/experiments'
    defaults = dict(cohort=doc / 'pallet_kp_corrected_supervision_20261010_v1/COHORT.json',
        parent=doc / 'pallet_three_head_observation_20261010_v7', v9_input=doc / 'pallet_sparse_local_line_20261010_v9',
        ordinary_parent=doc / 'pallet_hybrid_standard_controls_20261011_v17', point_parent=doc / 'pallet_consensus_decision_repair_20261011_v18')
    for key, value in defaults.items():
        if getattr(args, key) is None: setattr(args, key, value)
    args.output = args.output.absolute()
    environment = dict(PALLET_SOURCE_ROOT=str(args.source_root.resolve()), PALLET_BASELINE_ROOT=str(args.baseline_root.resolve()))
    saved = {key: os.environ.get(key) for key in environment}; os.environ.update(environment)
    try:
        with early_reference_canary(args):
            configure(args); {'freeze':freeze, 'score':score, 'stats':stats}[args.stage](args)
    finally:
        for key, value in saved.items():
            if value is None: os.environ.pop(key, None)
            else: os.environ[key] = value

if __name__ == '__main__': main()
