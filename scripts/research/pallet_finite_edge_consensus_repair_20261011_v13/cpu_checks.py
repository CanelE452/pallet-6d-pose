"""Freeze source bytes before meaningful deterministic CPU geometry tests."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import time
import traceback

REPO=Path(__file__).resolve().parents[3]
DOC=REPO/'_docs/experiments/pallet_finite_edge_consensus_repair_20261011_v13'
CODE=Path(__file__).resolve().parent
FILES=('CPU_PROTOCOL.json','CPU_STARTED.json','CPU_CHECKS.json')

def binding(p):
    h=hashlib.sha256(p.read_bytes())
    return dict(path=str(p.relative_to(REPO)),bytes=p.stat().st_size,sha256=h.hexdigest())

def write(p,data):
    with p.open('x') as f:json.dump(data,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')

def inputs():
    paths=[CODE/n for n in ('solver.py','test_solver.py','cpu_checks.py')]
    paths.append(DOC/'METHOD_CONTRACT_KO.md')
    for version,name in (('pallet_cornerwise_independent_20261010_v4','pose.py'),
        ('pallet_partial_line_independent_20261010_v5','solver.py'),
        ('pallet_sparse_local_line_20261010_v9','solver.py'),
        ('pallet_observation_refiner_20261009_v1','solver.py'),
        ('pallet_observation_refiner_20261009_v1','point_line.py')):
        paths.append(REPO/'scripts/research'/version/name)
    return {str(p.relative_to(REPO)):binding(p) for p in paths}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=('freeze','run'))
    args=p.parse_args()
    for name in FILES if args.stage=='freeze' else FILES[1:]:
        if (DOC/name).exists():raise ValueError('preserve '+name)
    current=inputs()
    if args.stage=='freeze':
        write(DOC/FILES[0],dict(schema='finite_edge_CPU_protocol_v13',inputs=current,
            numerical_tests_before_freeze=0,deterministic_fixture_new_seed=0,
            real_frame_model_RGB_GT_dataset_reads=0,controlled_optimizer_tests_are_not_natural_fits=True))
        print('FINITE_EDGE_CPU_FROZEN',flush=True);return
    own=json.loads((DOC/FILES[0]).read_text())
    if own['inputs']!=current:raise ValueError('frozen CPU source differs')
    write(DOC/FILES[1],dict(protocol=binding(DOC/FILES[0]),actual_test_attempt=1))
    before_threads=None;cv2=None;began=time.monotonic();error=None;result=None;stream=io.StringIO();cleanup_errors=[]
    try:
        import unittest
        import cv2
        from . import test_solver
        before_threads=cv2.getNumThreads()
        cv2.setNumThreads(1)
        suite=unittest.defaultTestLoader.loadTestsFromModule(test_solver)
        result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    except BaseException as e:error=dict(type=type(e).__name__,message=str(e),traceback=traceback.format_exc())
    finally:
        if before_threads is not None:
            try:cv2.setNumThreads(before_threads)
            except BaseException as e:cleanup_errors.append(dict(type=type(e).__name__,message=str(e)))
    unchanged=inputs()==current
    restored=before_threads is not None and not cleanup_errors and cv2.getNumThreads()==before_threads
    passed=error is None and result is not None and result.wasSuccessful() and unchanged and restored
    write(DOC/FILES[2],dict(schema='finite_edge_CPU_checks_v13',complete=True,passed=passed,
        own_protocol=binding(DOC/FILES[0]),inputs=current,tests_run=0 if result is None else result.testsRun,
        failures=[] if result is None else [dict(test=str(t),traceback=s) for t,s in result.failures],
        errors=[] if result is None else [dict(test=str(t),traceback=s) for t,s in result.errors],
        error=error,actual_unittest_output=stream.getvalue(),source_unchanged=unchanged,
        opencv_threads_restored=restored,cleanup_errors=cleanup_errors,
        elapsed_seconds=time.monotonic()-began,replay_wall_is_deployment_latency=False,
        real_frame_model_RGB_GT_dataset_reads=0,controlled_optimizer_tests_are_not_natural_fits=True))
    print('FINITE_EDGE_CPU',passed,0 if result is None else result.testsRun,stream.getvalue(),flush=True)
    if not passed:raise SystemExit(1)

if __name__=='__main__':main()
