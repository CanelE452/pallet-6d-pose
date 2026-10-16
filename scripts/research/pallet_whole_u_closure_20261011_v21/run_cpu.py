"""One ROOT-owned CPU run; freeze is stdlib-only and precedes scientific imports.

The unchanged V19 primitive/optimizer instrumentation is compiled only after
CPU_STARTED. Synthetic natural fixtures and controlled dispatch fixtures are
reported separately; this gate makes no real-frame accuracy or latency claim.
"""
import argparse
import ast
import contextlib
import copy
import hashlib
import importlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import unittest


PRIVATE = Path(__file__).resolve().parent
INSTRUMENT_SOURCE = Path('/tmp/pallet-finite-support-orthogonal-private-20261011-v19/run_geometry.py')
INSTRUMENT_SHA = 'd964a806959951b2b987094b3a7d290cc587a65ce5c583281d8a4e5b33c0c302'
EXPECTED = {
    'solver': '1603401ed1f7b1cde301b272bfc1f2f6d3c3935f371ba9c7bafd56e2a893d099',
    'tests': 'd97b927ae0c68f06c59bc035a2f797348b2ec150867ecd3abbcf0ad01b1ac003'}
LIMITS = dict(synthetic_CPU_contracts_only=True, real_rows_RGB_models_GT_training_loaded=0,
    new_settings_thresholds_weights_seeds_head_CAL_changes=0,
    automatic_retry=False, old_V16_full_scalar_FAIL_remains_FAIL=True,
    numerical_NEW_is_not_physical_accuracy_success=True,
    elapsed_is_not_full_path_latency=True,
    scientific_calls_before_STARTED=0,
    natural_convergence_expectations_require_actual_test_results=True,
    controlled_refit_and_tie_fixtures_are_not_natural_convergence_evidence=True)


def require(value, message):
    if not value: raise ValueError(message)


def binding(path):
    path = Path(path); digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''): digest.update(block)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=digest.hexdigest())


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def git_snapshot(root):
    def command(*args):
        return subprocess.check_output(['git', '-C', str(root), *args])
    return dict(HEAD=command('rev-parse', 'HEAD').decode().strip(),
        index_sha256=hashlib.sha256(command('ls-files', '--stage', '-z')).hexdigest(),
        tracked_diff_sha256=hashlib.sha256(command('diff', '--binary')).hexdigest(),
        cached_diff_sha256=hashlib.sha256(command('diff', '--cached', '--binary')).hexdigest(),
        status_sha256=hashlib.sha256(command('status', '--porcelain=v1', '-uall')).hexdigest())


def instrument_tree():
    raw = INSTRUMENT_SOURCE.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == INSTRUMENT_SHA, 'frozen V19 instrumentation source changed')
    tree = ast.parse(raw)
    functions = []
    for name in ('instrument', 'optimizer'):
        found = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name]
        require(len(found) == 1, 'one unchanged V19 instrumentation function: ' + name)
        functions.append(copy.deepcopy(found[0]))
    module = ast.Module(body=functions, type_ignores=[])
    proof = dict(source=binding(INSTRUMENT_SOURCE), whole_function_AST_changes=0,
        functions=['instrument', 'optimizer'],
        canonical_AST_sha256=hashlib.sha256(ast.dump(module, include_attributes=False).encode()).hexdigest(),
        least_squares_callbacks_counted=['residual_entries', 'jacobian_entries'],
        OpenCV_LM_is_not_SciPy_least_squares=True)
    return module, proof


def input_paths(repo):
    paths = {'solver': PRIVATE/'solver.py', 'tests': PRIVATE/'test_solver.py',
             'runner': Path(__file__).resolve(), 'instrumentation_source': INSTRUMENT_SOURCE}
    # Deliberately bind a superset of the production import closure. This
    # enumerates only Python source bytes, never models or scientific rows.
    files = sorted((repo/'scripts/research').rglob('*.py'))
    require(files, 'research Python source closure required')
    for path in files: paths['source:'+str(path.relative_to(repo))] = path
    for key, expected in EXPECTED.items():
        require(binding(paths[key])['sha256'] == expected, 'fixed V21 source changed: ' + key)
    return paths


def load_module(name, path):
    specification = importlib.util.spec_from_file_location(name, path)
    require(specification is not None and specification.loader is not None, 'explicit private module specification')
    module = importlib.util.module_from_spec(specification)
    previous = sys.modules.get(name)
    sys.modules[name] = module
    try: specification.loader.exec_module(module)
    except BaseException:
        if previous is None: sys.modules.pop(name, None)
        else: sys.modules[name] = previous
        raise
    return module, previous


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('freeze', 'run'))
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--baseline-root', type=Path, required=True)
    parser.add_argument('--original-repo', type=Path)
    parser.add_argument('--output', type=Path, default=PRIVATE/'cpu')
    args = parser.parse_args()
    repo, original, out = args.repo.resolve(), (args.original_repo or args.source_root).resolve(), args.output.resolve()
    require(out == PRIVATE/'cpu', 'exclusive fixed private21/cpu output only')
    for protected in (repo, original):
        require(not out.is_relative_to(protected) and not protected.is_relative_to(out), 'CPU output disjoint from protected repositories')
    require(args.source_root.is_dir() and args.baseline_root.is_dir(), 'explicit configured source and baseline directories')
    paths = input_paths(repo); inputs = {key:binding(path) for key,path in paths.items()}
    tree, instrument_proof = instrument_tree()
    test_tree = ast.parse(paths['tests'].read_bytes())
    configured = sorted(node.name for cls in test_tree.body if isinstance(cls,ast.ClassDef)
        for node in cls.body if isinstance(node,ast.FunctionDef) and node.name.startswith('test_'))
    require(len(configured) == len(set(configured)) == 7, 'seven unchanged unique CPU test groups')
    fixture_scopes = {name:('CONTROLLED_DISPATCH_WITH_ACTUAL_GEOMETRY' if name.startswith('test_controlled_')
        else 'SOURCE_AND_INPUT_CONTRACT' if name.startswith('test_contract_') else 'NATURAL_SYNTHETIC_GEOMETRY') for name in configured}
    environment = dict(PALLET_SOURCE_ROOT=str(args.source_root.resolve()),
                       PALLET_BASELINE_ROOT=str(args.baseline_root.resolve()))
    protection = dict(research=git_snapshot(repo), original=git_snapshot(original))
    proto = out/'CPU_PROTOCOL.json'
    if args.phase == 'freeze':
        require(not out.exists(), 'fresh absent CPU output; original failures are never overwritten')
        out.mkdir(parents=True)
        write(proto, dict(schema='whole_U_closure_actual_CPU_protocol_v21', inputs=inputs,
            configured_cases=configured, fixture_scopes=fixture_scopes, environment=environment,
            protection=protection, original_repo=str(original), repo=str(repo), limits=LIMITS,
            instrumentation_source_proof=instrument_proof, source_frozen_before_scientific_import=True,
            opencv_threads=1, python=sys.executable))
        print('V21_CPU_PROTOCOL_FROZEN'); return
    frozen = json.loads(proto.read_bytes())
    for key,value in dict(inputs=inputs,configured_cases=configured,fixture_scopes=fixture_scopes,
        environment=environment,protection=protection,original_repo=str(original),repo=str(repo),
        limits=LIMITS,instrumentation_source_proof=instrument_proof).items():
        require(frozen[key] == value, 'exact frozen CPU authority: ' + key)
    require(not (out/'CPU_STARTED.json').exists() and not (out/'CPU_CHECKS.json').exists(), 'single actual CPU run; no retry')
    write(out/'CPU_STARTED.json', dict(schema='whole_U_closure_actual_CPU_started_v21',
        protocol=binding(proto), inputs=inputs, environment=environment, protection=protection,
        configured_cases=configured, fixture_scopes=fixture_scopes, limits=LIMITS))
    saved_environment = {key:os.environ.get(key) for key in environment}
    saved_path = list(sys.path); saved_bytecode = sys.dont_write_bytecode
    test_solver_module_name = 'private_v21_point_closure_solver'
    previous_test_solver_module = sys.modules.get(test_solver_module_name)
    os.environ.update(environment); sys.path.insert(0, str(repo)); sys.dont_write_bytecode = True
    calls, restore, originals, loaded = {}, [], [], []
    error = None; result = None; versions = None; source_proof = None; cleanup = []; log = io.StringIO()
    started = time.monotonic()
    try:
        import cv2
        import numpy as np
        scope = dict(calls=calls, restore=restore)
        exec(compile(ast.fix_missing_locations(tree), str(INSTRUMENT_SOURCE)+':unchanged CPU hooks', 'exec'), scope)
        instrument = scope['instrument']
        old_threads = cv2.getNumThreads(); cv2.setNumThreads(1)
        restore.append(lambda:cv2.setNumThreads(old_threads))
        for name in ('solvePnP', 'solvePnPGeneric', 'solvePnPRefineLM', 'projectPoints', 'Rodrigues', 'cornerSubPix'):
            originals.append((cv2, name, getattr(cv2,name))); instrument(cv2,name,'cv2.'+name)
        originals.append((np.linalg,'svd',np.linalg.svd)); instrument(np.linalg,'svd','numpy.linalg.svd')
        parent = importlib.import_module('scripts.research.pallet_finite_edge_consensus_repair_20261011_v13.solver')
        original_opt = parent.least_squares
        calls['least_squares'] = dict(entries=0,returns=0,exceptions=0,success_reports=0,residual_entries=0,jacobian_entries=0)
        scope['original_opt'] = original_opt
        parent.least_squares = scope['optimizer']
        originals.append((parent,'least_squares',original_opt)); restore.append(lambda:setattr(parent,'least_squares',original_opt))
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            source, previous = load_module('private_v21_CPU_solver', paths['solver']); loaded.append(('private_v21_CPU_solver',previous))
            tests, previous = load_module('private_v21_CPU_tests', paths['tests']); loaded.append(('private_v21_CPU_tests',previous))
            suite = unittest.defaultTestLoader.loadTestsFromModule(tests)
            result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
            source_proof = source.SOURCE_PROOF
        versions = dict(numpy=np.__version__,opencv=cv2.__version__)
    except BaseException as exception:
        error = dict(type=type(exception).__name__,message=str(exception),traceback=traceback.format_exc())
    finally:
        for callback in reversed(restore):
            try: callback()
            except BaseException as exception: cleanup.append(dict(type=type(exception).__name__,message=str(exception)))
        for name,previous in reversed(loaded):
            if previous is None: sys.modules.pop(name,None)
            else: sys.modules[name]=previous
        if previous_test_solver_module is None: sys.modules.pop(test_solver_module_name,None)
        else: sys.modules[test_solver_module_name]=previous_test_solver_module
        for key,value in saved_environment.items():
            if value is None: os.environ.pop(key,None)
            else: os.environ[key]=value
        sys.path[:] = saved_path; sys.dont_write_bytecode = saved_bytecode
    after = {key:binding(path) for key,path in paths.items()}
    protected_after = dict(research=git_snapshot(repo),original=git_snapshot(original))
    restored = dict(primitive_and_optimizer_identities=all(getattr(obj,name) is value for obj,name,value in originals),
        split_environment=all(os.environ.get(key)==value for key,value in saved_environment.items()),
        sys_path=sys.path==saved_path,bytecode_setting=sys.dont_write_bytecode==saved_bytecode,
        private_test_solver_namespace=sys.modules.get(test_solver_module_name) is previous_test_solver_module,
        opencv_threads=(cv2.getNumThreads()==old_threads) if 'old_threads' in locals() else None)
    counters_closed = all(value['entries']==value['returns']+value['exceptions'] for value in calls.values())
    passed = (error is None and result is not None and result.wasSuccessful() and result.testsRun==7
        and not result.skipped and after==inputs and protected_after==protection and not cleanup
        and all(value is True for value in restored.values()) and counters_closed)
    checks = dict(schema='whole_U_closure_actual_CPU_checks_v21',complete=True,passed=passed,
        tests_run=0 if result is None else result.testsRun,configured_cases=configured,fixture_scopes=fixture_scopes,
        skipped=[] if result is None else [dict(test=str(test),reason=reason) for test,reason in result.skipped],
        failures=[] if result is None else [dict(test=str(test),traceback=value) for test,value in result.failures],
        errors=[] if result is None else [dict(test=str(test),traceback=value) for test,value in result.errors],
        exception=error,cleanup_errors=cleanup,restoration=restored,counter_dispatch_closed=counters_closed,
        actual_CPU_numerical_calls=calls,source_isolation_proof=source_proof,versions=versions,
        inputs=inputs,source_unchanged=after==inputs,protection_before=protection,protection_after=protected_after,
        protocol=binding(proto),runner=inputs['runner'],instrumentation_source_proof=instrument_proof,
        environment=environment,limits=LIMITS,elapsed_seconds=time.monotonic()-started,log=log.getvalue())
    write(out/'CPU_CHECKS.json', checks)
    print(json.dumps(dict(passed=passed,tests_run=checks['tests_run'],failure_count=len(checks['failures']),
        error_count=len(checks['errors']),exception=error,calls=calls),ensure_ascii=False))
    if not passed: sys.exit(1)


if __name__ == '__main__': main()
