"""ROOT-owned actual CPU fixture gate, frozen before production imports."""
import argparse
import ast
import contextlib
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import sys
import time
import traceback
import unittest


PACKAGE = 'scripts.research.pallet_finite_support_orthogonal_audit_20261011_v19'
DIRECTORIES = ('pallet_observation_refiner_20261009_v1', 'pallet_cornerwise_independent_20261010_v4',
               'pallet_partial_line_independent_20261010_v5', 'pallet_sparse_local_line_20261010_v9',
               'pallet_finite_edge_consensus_repair_20261011_v13', 'pallet_stable_incidence_guard_20261011_v14',
               'pallet_factor_subset_consensus_20261011_v15', 'pallet_consensus_decision_repair_20261011_v18',
               'pallet_finite_support_orthogonal_audit_20261011_v19')


def require(v, reason):
    if not v: raise ValueError(reason)


def binding(p):
    return dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())


def write(p, d):
    with p.open('x') as f:
        json.dump(d,f,indent=2,ensure_ascii=False,allow_nan=False)
        f.write('\n');f.flush();os.fsync(f.fileno())


def main():
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['freeze','run'])
    parser.add_argument('--repo',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args();repo=a.repo.resolve();out=a.output.absolute()
    require((repo/'scripts/research/pallet_finite_support_orthogonal_audit_20261011_v19/test_geometry.py').is_file(),'actual test source required')
    files=sorted(p for d in DIRECTORIES for p in (repo/'scripts/research'/d).rglob('*.py'))
    inputs={str(p.relative_to(repo)):binding(p) for p in files}
    own=binding(Path(__file__).resolve())
    test_tree=ast.parse((repo/'scripts/research/pallet_finite_support_orthogonal_audit_20261011_v19/test_geometry.py').read_text())
    configured=sorted(n.name for c in test_tree.body if isinstance(c,ast.ClassDef)
                      for n in c.body if isinstance(n,ast.FunctionDef) and n.name.startswith('test_'))
    require(configured and len(configured)==len(set(configured)),'nonempty unique fixed CPU cases')
    limits=dict(fixture_geometry_only=True,new_images_RGB_training_GT_dataset_evaluation=0,
                real_rows_and_model_weights_loaded=0,pose_accuracy_and_full_path_latency_claim=False,
                parent_production_numerical_calls_are_counted_for_CPU_contracts=True,
                old_V16_geometry_FAIL_remains_FAIL=True,automatic_retry=False)
    if a.phase=='freeze':
        require(not out.exists(),'fresh absent CPU output required');out.mkdir(parents=True)
        write(out/'CPU_PROTOCOL.json',dict(schema='observed_frame_actual_CPU_protocol_v19',inputs=inputs,
            runner=own,package=PACKAGE,python=sys.executable,opencv_threads=1,
            configured_cases=configured,source_frozen_before_any_production_import=True,limits=limits))
        print('CPU_PROTOCOL_FROZEN');return
    protocol=json.loads((out/'CPU_PROTOCOL.json').read_bytes())
    require(protocol['inputs']==inputs and protocol['runner']==own,'exact frozen code bytes')
    require(protocol['configured_cases']==configured,'exact frozen test cases')
    require(not(out/'CPU_STARTED.json').exists() and not(out/'CPU_CHECKS.json').exists(),'one actual CPU run only')
    write(out/'CPU_STARTED.json',dict(schema='observed_frame_actual_CPU_started_v19',inputs=inputs,
        protocol=binding(out/'CPU_PROTOCOL.json'),runner=own,limits=limits))
    sys.path.insert(0,str(repo));calls={};old_cv={};restore=[];error=None;result=None;log=io.StringIO()
    t0=time.monotonic()
    try:
        import cv2
        import numpy as np
        original_svd=np.linalg.svd;calls['numpy.linalg.svd']=dict(entries=0,returns=0,exceptions=0)
        def svd(*args,**kwargs):
            calls['numpy.linalg.svd']['entries']+=1
            try:value=original_svd(*args,**kwargs)
            except BaseException:calls['numpy.linalg.svd']['exceptions']+=1;raise
            calls['numpy.linalg.svd']['returns']+=1;return value
        np.linalg.svd=svd;restore.append(lambda:setattr(np.linalg,'svd',original_svd))
        threads=cv2.getNumThreads();cv2.setNumThreads(1);restore.append(lambda:cv2.setNumThreads(threads))
        for name in ('solvePnP','solvePnPGeneric','solvePnPRefineLM','projectPoints','Rodrigues'):
            original=getattr(cv2,name);old_cv[name]=original;calls[name]=dict(entries=0,returns=0,exceptions=0)
            def counted(*args,_name=name,_fn=original,**kwargs):
                calls[_name]['entries']+=1
                try:value=_fn(*args,**kwargs)
                except BaseException:calls[_name]['exceptions']+=1;raise
                calls[_name]['returns']+=1;return value
            setattr(cv2,name,counted)
        restore.append(lambda:[setattr(cv2,n,f) for n,f in old_cv.items()])
        source=importlib.import_module(PACKAGE+'.solver')
        original_opt=source.S13.least_squares;calls['least_squares']=dict(entries=0,returns=0,exceptions=0)
        def opt(*args,**kwargs):
            calls['least_squares']['entries']+=1
            try:value=original_opt(*args,**kwargs)
            except BaseException:calls['least_squares']['exceptions']+=1;raise
            calls['least_squares']['returns']+=1;return value
        source.S13.least_squares=opt;restore.append(lambda:setattr(source.S13,'least_squares',original_opt))
        suite=unittest.defaultTestLoader.loadTestsFromName(PACKAGE+'.test_geometry')
        with contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
            result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
        source_proof=source.SOURCE_PROOF
        versions=dict(numpy=np.__version__,opencv=cv2.__version__)
    except BaseException as exc:
        error=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())
        source_proof=versions=None
    finally:
        cleanup=[]
        for f in reversed(restore):
            try:f()
            except BaseException as exc:cleanup.append(dict(type=type(exc).__name__,message=str(exc)))
    after={str(p.relative_to(repo)):binding(p) for p in files}
    successful=(error is None and result is not None and result.wasSuccessful()
                and result.testsRun==len(configured) and not result.skipped and after==inputs and not cleanup)
    checks=dict(schema='observed_frame_actual_CPU_checks_v19',complete=True,passed=successful,
        tests_run=0 if result is None else result.testsRun,
        configured_cases=configured,skipped=[] if result is None else [dict(test=str(t),reason=s) for t,s in result.skipped],
        failures=[] if result is None else [dict(test=str(t),traceback=s) for t,s in result.failures],
        errors=[] if result is None else [dict(test=str(t),traceback=s) for t,s in result.errors],
        exception=error,cleanup_errors=cleanup,source_unchanged=after==inputs,
        actual_CPU_numerical_entry_calls=calls,source_isolation_proof=source_proof,versions=versions,
        source_bindings=inputs,protocol=binding(out/'CPU_PROTOCOL.json'),runner=own,
        elapsed_verification_seconds=time.monotonic()-t0,
        elapsed_is_not_pipeline_latency=True,log=log.getvalue(),limits=limits)
    write(out/'CPU_CHECKS.json',checks)
    print(json.dumps(dict(passed=successful,tests_run=checks['tests_run'],failure_count=len(checks['failures']),error_count=len(checks['errors']),exception=error,calls=calls),ensure_ascii=False))
    if not successful:sys.exit(1)


if __name__=='__main__':main()
