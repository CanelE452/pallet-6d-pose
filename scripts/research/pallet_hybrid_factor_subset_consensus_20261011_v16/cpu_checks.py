"""Freeze five synthetic hybrid composition contracts before normal package imports or mathematics."""
import argparse
from collections import Counter
import builtins
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time
import traceback

REPO=Path(__file__).resolve().parents[3]
CODE=Path(__file__).resolve().parent
DOC=REPO/'_docs/experiments/pallet_hybrid_factor_subset_consensus_20261011_v16'
FILES=('CPU_PROTOCOL.json','CPU_STARTED.json','CPU_CHECKS.json')
PRIMITIVES=('solvePnP','solvePnPGeneric','solvePnPRefineLM','projectPoints','Rodrigues','cornerSubPix')

def binding(p):
    return dict(path=str(p.relative_to(REPO)),bytes=p.stat().st_size,
        sha256=hashlib.sha256(p.read_bytes()).hexdigest())

def write(p,value):
    with p.open('x') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')

def inputs():
    paths=[CODE/n for n in ('__init__.py','pipeline.py','test_pipeline.py','cpu_checks.py')]
    paths.append(DOC/'METHOD_CONTRACT_KO.md')
    modules={
        'pallet_factor_subset_consensus_20261011_v15':('solver.py','pipeline.py'),
        'pallet_finite_edge_consensus_repair_20261011_v13':('solver.py','pipeline.py','test_solver.py'),
        'pallet_stable_incidence_guard_20261011_v14':('pipeline.py',),
        'pallet_endpoint_observability_repair_20261011_v11':('pipeline.py',),
        'pallet_unused_line_endpoint_audit_20261011_v10':('audit.py',),
        'pallet_sparse_local_line_20261010_v9':('solver.py','pipeline.py','common.py'),
        'pallet_same_observation_controls_20261010_v8':('controls.py','common.py'),
        'pallet_partial_line_heldout_20261010_v6':('solver.py',),
        'pallet_partial_line_independent_20261010_v5':('solver.py',),
        'pallet_cornerwise_independent_20261010_v4':('pose.py','selection.py'),
        'pallet_boundary_corner_refiner_20261010_v2':('pipeline.py','common.py'),
        'pallet_observation_refiner_20261009_v1':('solver.py','point_line.py','inference.py','common.py')}
    for version,names in modules.items():
        root=REPO/'scripts/research'/version
        for name in ('__init__.py',*names):
            if (root/name).is_file():paths.append(root/name)
    for p in (REPO/'scripts/__init__.py',REPO/'scripts/research/__init__.py'):
        if p.is_file():paths.append(p)
    return {str(p.relative_to(REPO)):binding(p) for p in paths}

def environment(args):
    roots={name:str(Path(value).resolve()) for name,value in
        (('PALLET_SOURCE_ROOT',args.source_root),('PALLET_BASELINE_ROOT',args.baseline_root))}
    if not all(Path(value).is_dir() for value in roots.values()):raise ValueError('explicit existing split roots required')
    return roots

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('freeze','run'))
    parser.add_argument('--source-root',required=True);parser.add_argument('--baseline-root',required=True)
    args=parser.parse_args();configured=environment(args)
    for name in FILES if args.stage=='freeze' else FILES[1:]:
        if (DOC/name).exists():raise ValueError('preserve '+name)
    current=inputs()
    if args.stage=='freeze':
        write(DOC/FILES[0],dict(schema='hybrid_factor_subset_CPU_protocol_v16',inputs=current,
            configured_environment=configured,configured_test_groups=5,numerical_tests_before_freeze=0,
            normal_package_import=True,original_frozen_sources_changed=False,
            real_frame_model_RGB_GT_dataset_reads=0,new_seed=0,
            controlled_corner_selection_fixture_is_not_fresh_V4_selection=True))
        print('HYBRID_FACTOR_SUBSET_CPU_FROZEN',flush=True);return
    own=json.loads((DOC/FILES[0]).read_text())
    if own['schema']!='hybrid_factor_subset_CPU_protocol_v16' or own['inputs']!=current or own['configured_environment']!=configured:
        raise ValueError('frozen CPU source/environment differs')
    write(DOC/FILES[1],dict(protocol=binding(DOC/FILES[0]),actual_test_attempt=1,
        configured_environment=configured,configured_test_groups=5,normal_package_import=True))
    before_env={k:os.environ.get(k) for k in configured};opens=(builtins.open,io.open,Path.open)
    saved=[];cv2=None;threads=None;result=None;error=None;cleanup=[];asset_reads=[]
    stream=io.StringIO();began=time.monotonic();loaded=[]
    ce,cr,cx=(Counter({n:0 for n in PRIMITIVES}) for _ in range(3))
    svd=Counter(dict(entries=0,returns=0,exceptions=0))
    optimizer=Counter(dict(entries=0,returns=0,exceptions=0,residual_entries=0,jacobian_entries=0))
    def permitted(file):
        if isinstance(file,(str,bytes,os.PathLike)):
            p=Path(os.fsdecode(file)).absolute()
            if any(p.is_relative_to(Path(root)) for root in configured.values()) and p.suffix not in ('.py','.pyc','.so'):
                asset_reads.append(str(p));raise RuntimeError('SYNTHETIC_CPU_REAL_ASSET_READ')
    def guarded_open(file,*a,**kw):permitted(file);return opens[0](file,*a,**kw)
    def guarded_io(file,*a,**kw):permitted(file);return opens[1](file,*a,**kw)
    def guarded_path(self,*a,**kw):permitted(self);return opens[2](self,*a,**kw)
    try:
        os.environ.update(configured)
        builtins.open,io.open,Path.open=guarded_open,guarded_io,guarded_path
        import unittest
        import cv2
        import numpy as np
        from . import test_pipeline
        from ..pallet_finite_edge_consensus_repair_20261011_v13 import solver as finite
        threads=cv2.getNumThreads();cv2.setNumThreads(1)
        for name in PRIMITIVES:
            fn=getattr(cv2,name);saved.append((cv2,name,fn))
            def counted(*a,_fn=fn,_name=name,**kw):
                ce[_name]+=1
                try:value=_fn(*a,**kw)
                except BaseException:cx[_name]+=1;raise
                cr[_name]+=1;return value
            setattr(cv2,name,counted)
        original_svd=np.linalg.svd;saved.append((np.linalg,'svd',original_svd))
        def counted_svd(*a,**kw):
            svd['entries']+=1
            try:value=original_svd(*a,**kw)
            except BaseException:svd['exceptions']+=1;raise
            svd['returns']+=1;return value
        np.linalg.svd=counted_svd
        original_optimizer=finite.least_squares;saved.append((finite,'least_squares',original_optimizer))
        def counted_optimizer(fun,x0,*a,**kw):
            optimizer['entries']+=1
            def residual(*v,**options):optimizer['residual_entries']+=1;return fun(*v,**options)
            jac=kw.get('jac')
            if callable(jac):
                def counted_jac(*v,**options):optimizer['jacobian_entries']+=1;return jac(*v,**options)
                kw['jac']=counted_jac
            try:value=original_optimizer(residual,x0,*a,**kw)
            except BaseException:optimizer['exceptions']+=1;raise
            optimizer['returns']+=1;return value
        finite.least_squares=counted_optimizer
        suite=unittest.defaultTestLoader.loadTestsFromModule(test_pipeline)
        if suite.countTestCases()!=5:raise ValueError('exact five hybrid functional groups required')
        result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
        for module in tuple(sys.modules.values()):
            filename=getattr(module,'__file__',None)
            if filename:
                path=Path(filename).resolve()
                if path.is_relative_to(REPO/'scripts/research') and path.suffix=='.py':loaded.append(binding(path))
        if any(v['path'] not in current or current[v['path']]!=v for v in loaded):
            raise ValueError('actual project import outside prefrozen CPU source closure')
    except BaseException as exc:error=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())
    finally:
        for owner,name,fn in reversed(saved):
            try:setattr(owner,name,fn)
            except BaseException as exc:cleanup.append(dict(type=type(exc).__name__,message=str(exc)))
        if threads is not None:
            try:cv2.setNumThreads(threads)
            except BaseException as exc:cleanup.append(dict(type=type(exc).__name__,message=str(exc)))
        builtins.open,io.open,Path.open=opens
        for key,value in before_env.items():
            if value is None:os.environ.pop(key,None)
            else:os.environ[key]=value
    unchanged=inputs()==current
    restored=threads is not None and cv2.getNumThreads()==threads and not cleanup
    environment_restored=before_env=={k:os.environ.get(k) for k in before_env}
    instruments_restored=all(getattr(o,n) is f for o,n,f in saved)
    passed=error is None and result is not None and result.testsRun==5 and result.wasSuccessful() and unchanged and restored and environment_restored and instruments_restored and not asset_reads
    write(DOC/FILES[2],dict(schema='hybrid_factor_subset_CPU_checks_v16',complete=True,passed=passed,
        own_protocol=binding(DOC/FILES[0]),inputs=current,configured_environment=configured,
        tests_run=0 if result is None else result.testsRun,
        failures=[] if result is None else [dict(test=str(t),traceback=s) for t,s in result.failures],
        errors=[] if result is None else [dict(test=str(t),traceback=s) for t,s in result.errors],
        error=error,actual_unittest_output=stream.getvalue(),source_unchanged=unchanged,
        opencv_threads_restored=restored,environment_restored=environment_restored,
        instruments_restored=instruments_restored,asset_canary_restored=opens==(builtins.open,io.open,Path.open),
        cleanup_errors=cleanup,forbidden_real_asset_reads=asset_reads,actual_imported_project_sources=loaded,
        actual_OpenCV_entry_calls=dict(ce),actual_OpenCV_return_calls=dict(cr),actual_OpenCV_exception_calls=dict(cx),
        actual_SVD_dispatch=dict(svd),actual_instrumented_optimizer_dispatch=dict(optimizer),
        all_Generic_and_SciPy_test_solves_are_natural=True,
        elapsed_seconds=time.monotonic()-began,replay_wall_is_deployment_latency=False,
        real_frame_model_RGB_GT_dataset_reads=len(asset_reads),normal_package_import=True,
        controlled_corner_selection_fixture_is_not_fresh_V4_selection=True))
    print('HYBRID_FACTOR_SUBSET_CPU',passed,0 if result is None else result.testsRun,stream.getvalue(),flush=True)
    if not passed:raise SystemExit(1)

if __name__=='__main__':main()
