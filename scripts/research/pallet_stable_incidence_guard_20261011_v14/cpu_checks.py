"""Freeze the incidence guard before seven controlled CPU contracts."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import time
import traceback

REPO=Path(__file__).resolve().parents[3]
CODE=Path(__file__).resolve().parent
DOC=REPO/'_docs/experiments/pallet_stable_incidence_guard_20261011_v14'
FILES=('CPU_PROTOCOL.json','CPU_STARTED.json','CPU_CHECKS.json')

def binding(p):
    return dict(path=str(p.relative_to(REPO)),bytes=p.stat().st_size,
        sha256=hashlib.sha256(p.read_bytes()).hexdigest())

def write(p,data):
    with p.open('x') as f:json.dump(data,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')

def inputs():
    paths=[CODE/n for n in ('pipeline.py','test_guard.py','cpu_checks.py','__init__.py')]
    paths.append(DOC/'METHOD_CONTRACT_KO.md')
    paths.append(REPO/'_docs/experiments/pallet_finite_edge_consensus_repair_20261011_v13/CUSP_OBSERVABILITY_DIAGNOSIS_KO.md')
    for version,name in (
        ('pallet_finite_edge_consensus_repair_20261011_v13','solver.py'),
        ('pallet_finite_edge_consensus_repair_20261011_v13','pipeline.py'),
        ('pallet_finite_edge_consensus_repair_20261011_v13','test_solver.py'),
        ('pallet_endpoint_observability_repair_20261011_v11','pipeline.py'),
        ('pallet_sparse_local_line_20261010_v9','solver.py'),
        ('pallet_sparse_local_line_20261010_v9','pipeline.py'),
        ('pallet_cornerwise_independent_20261010_v4','pose.py'),
        ('pallet_partial_line_independent_20261010_v5','solver.py'),
        ('pallet_observation_refiner_20261009_v1','solver.py'),
        ('pallet_same_observation_controls_20261010_v8','controls.py'),
        ('pallet_boundary_corner_refiner_20261010_v2','pipeline.py')):
        paths.append(REPO/'scripts/research'/version/name)
    return {str(p.relative_to(REPO)):binding(p) for p in paths}

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=('freeze','run'))
    args=parser.parse_args()
    for name in FILES if args.stage=='freeze' else FILES[1:]:
        if (DOC/name).exists():raise ValueError('preserve '+name)
    current=inputs()
    if args.stage=='freeze':
        write(DOC/FILES[0],dict(schema='stable_incidence_CPU_protocol_v14',inputs=current,
            numerical_tests_before_freeze=0,configured_test_groups=7,
            real_frame_model_RGB_GT_dataset_reads=0,new_seed=0,
            selected_candidate_packets_controlled=True,optimizer_convergence_not_claimed=True))
        print('STABLE_INCIDENCE_CPU_FROZEN',flush=True);return
    own=json.loads((DOC/FILES[0]).read_text())
    if own['inputs']!=current:raise ValueError('frozen CPU source differs')
    write(DOC/FILES[1],dict(protocol=binding(DOC/FILES[0]),actual_test_attempt=1))
    cv2=None;threads=None;result=None;error=None;cleanup=[];stream=io.StringIO();began=time.monotonic()
    try:
        import unittest
        import cv2
        from . import test_guard
        threads=cv2.getNumThreads();cv2.setNumThreads(1)
        suite=unittest.defaultTestLoader.loadTestsFromModule(test_guard)
        result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    except BaseException as e:error=dict(type=type(e).__name__,message=str(e),traceback=traceback.format_exc())
    finally:
        if threads is not None:
            try:cv2.setNumThreads(threads)
            except BaseException as e:cleanup.append(dict(type=type(e).__name__,message=str(e)))
    unchanged=inputs()==current;restored=threads is not None and not cleanup and cv2.getNumThreads()==threads
    passed=error is None and result is not None and result.testsRun==7 and result.wasSuccessful() and unchanged and restored
    write(DOC/FILES[2],dict(schema='stable_incidence_CPU_checks_v14',complete=True,passed=passed,
        own_protocol=binding(DOC/FILES[0]),inputs=current,tests_run=0 if result is None else result.testsRun,
        failures=[] if result is None else [dict(test=str(t),traceback=s) for t,s in result.failures],
        errors=[] if result is None else [dict(test=str(t),traceback=s) for t,s in result.errors],
        error=error,actual_unittest_output=stream.getvalue(),source_unchanged=unchanged,
        opencv_threads_restored=restored,cleanup_errors=cleanup,elapsed_seconds=time.monotonic()-began,
        real_frame_model_RGB_GT_dataset_reads=0,new_seed=0,optimizer_convergence_not_claimed=True))
    print('STABLE_INCIDENCE_CPU',passed,0 if result is None else result.testsRun,stream.getvalue(),flush=True)
    if not passed:raise SystemExit(1)

if __name__=='__main__':main()
