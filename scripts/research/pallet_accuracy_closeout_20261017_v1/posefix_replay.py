"""Frozen first-RAW PoseFix -> two inherited SubPix caps -> unchanged F.

Only prediction/known-geometry/RGB inputs enter inference. Accuracy evaluation
is deliberately absent; legacy output parity is a separate command/process.
"""
from __future__ import annotations
import argparse
import ast
import builtins
from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import resource
import sys
import time

import cv2
import numpy as np

VIS = Path('_docs/experiments/pallet_visibility_curriculum_20261013_v1/nonsevere_v1')
OLD = VIS/'posefix_sub_comparison_v1'
PF = Path('data/pallet/results/pallet_posefix_replay_diagnosis_v1')
NATIVE143 = VIS/'remainder_after_test49_v1/native_input_preparation_v1/NATIVE_INPUTS143.jsonl.gz'
RAW143 = VIS/'remainder_after_test49_v1/posefix_baseline_supplement_v4_14/PREDICTIONS.jsonl.gz'
CACHE = PF/'predictions/seed1_REAL_DEV.npz'
CHECKPOINT = 'b9670d1a42db8d4e80ae1efe55eb237f05f81f7bd089cefe4051bbedd7d0a581'
CACHE_SHA = '683b7395860865f7e10cba5e1dfb7c04e84234dd296340416c4759b671c95f06'
SUBPIX = dict(winSize=[5, 5], zeroZone=[-1, -1], minimum_image_extent=15,
              criteria=dict(maxCount=40, epsilon=.001))
METHODS = ('POSEFIX_RAW', 'POSEFIX_SUBPIX_PF_ANCHOR', 'POSEFIX_SUBPIX_YOLO_ANCHOR')
PARITY_ARMS = ('NATIVE_BASE', 'INITIAL_N3_SUBPIX') + METHODS
SOURCES = ('scripts/research/pallet_dim_conditioned_p_v1/pose.py',
           'scripts/research/pallet_training_free_compare_20261007_v1/methods.py',
           'scripts/paper/pose_metric_closure_v1/run_pose_evaluation.py',
           'challenge/evaluation_v2/pnp_selector.py',
           'scripts/annotate/pallet_geometry.py')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4*1024*1024), b''): h.update(b)
    return h.hexdigest()


def read(path): return json.loads(Path(path).read_text())


def rows(path):
    with gzip.open(path, 'rt') as f: return [json.loads(x) for x in f if x.strip()]


def finite(x):
    if isinstance(x, dict): return {str(k): finite(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [finite(v) for v in x]
    if isinstance(x, np.ndarray): return finite(x.tolist())
    if isinstance(x, np.generic): return finite(x.item())
    if isinstance(x, float) and not math.isfinite(x): return None
    return x


def write(path, value):
    with Path(path).open('x') as f:
        json.dump(finite(value), f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def save(path, records):
    # Deterministic compression, so content identity is independently verifiable.
    with Path(path).open('xb') as raw:
        with gzip.GzipFile(fileobj=raw, mode='wb', mtime=0) as z:
            for r in records:
                z.write((json.dumps(finite(r), separators=(',', ':'), allow_nan=False)+'\n').encode())


def timing(start, cpu):
    return dict(wall_seconds=time.monotonic()-start, process_cpu_seconds=time.process_time()-cpu,
                peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                cpu_threads=1, new_NN_forwards=0, new_training_steps=0)


def functions(path, names, namespace):
    tree = ast.parse(Path(path).read_text())
    nodes = [x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name in names]
    if {x.name for x in nodes} != set(names): raise ValueError('Unchanged function unavailable')
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(path), 'exec'), namespace)
    return namespace


@contextmanager
def input_boundary(repo, output, protocol):
    """Disallow scientific answers, neural/training entrypoints and outside writes.

    Libraries, source imports, fonts and runtime caches remain readable. OpenCV
    image reads are explicitly hashed before use because native C I/O is outside
    Python's open hooks. All scientific decoded inputs are bound by protocol.
    """
    originals = (builtins.open, io.open, Path.open)
    log = dict(scientific_answer_reads=0, forbidden_probe_denied=0, decoded_input_paths=[],
               native_RGB_reads_explicitly_SHA_checked=True,
               policy='prediction-only exact bound files; no references, scores, oracle, target labels or NN')
    forbidden = ('FIXED_REFERENCES', 'ISOLATED_CAL_SCORING', 'ISOLATED_TEST_SCORING',
                 'TARGETS.json', 'SCORED_ROWS', 'RESULTS.json', 'SCORING', 'ORACLE',
                 '/annotations/', 'GEOMETRY_RESOLVED_POSE_GT', '/metrics/')
    def wrapper(fn):
        def guarded(path, *args, **kwargs):
            if not isinstance(path, (str, bytes, Path)): return fn(path, *args, **kwargs)
            text = str(path)
            mode = kwargs.get('mode', args[0] if args else 'r')
            if any(c in str(mode) for c in 'wax+'):
                if not Path(path).resolve().is_relative_to(output.resolve()):
                    raise PermissionError('Inference writes outside new owned directory')
            elif any(token in text for token in forbidden):
                log['forbidden_probe_denied'] += 1
                raise PermissionError('Forbidden scientific answer read: '+text)
            return fn(path, *args, **kwargs)
        return guarded
    builtins.open, io.open, Path.open = map(wrapper, originals)
    try:
        for sentinel in ('FIXED_REFERENCES.json', 'TARGETS.json', 'ISOLATED_CAL_SCORING.json'):
            try: builtins.open(output/sentinel, 'rb')
            except PermissionError: pass
            else: raise AssertionError('Answer I/O boundary probe failed')
        yield log
    finally:
        builtins.open, io.open, Path.open = originals


def prepare(repo, private):
    start, cpu = time.monotonic(), time.process_time()
    out = private/'posefix_inputs_v1'; out.mkdir()
    original_protocol = read(repo/OLD/'PROTOCOL.json')
    if sha(repo/CACHE) != CACHE_SHA: raise ValueError('Frozen neural cache changed')
    pf_inputs = {r['id']: r for r in read(repo/PF/'INPUTS.json') if r['split']=='REAL_DEV'}
    if len(pf_inputs) != 319: raise ValueError('Exact REAL_DEV319 source required')
    entries = read(repo/OLD/'BASELINE_INPUTS.json')['frames']
    if Counter(r['frame']['partition'] for r in entries) != {'CAL':46, 'TEST':49}:
        raise ValueError('Exact COMMON95 population required')
    with np.load(repo/CACHE, allow_pickle=False) as z:
        ids = [str(v) for v in z['ids']]
        index = {v:i for i,v in enumerate(ids)}
        points = z['points'][:,0,1].copy()
        if str(z['checkpoint_sha256'].item()) != CHECKPOINT: raise ValueError('Checkpoint mismatch')
        pass0 = z['points'][:,0,0].copy()
    native = rows(repo/NATIVE143)
    if len(native)!=143 or len({r['id']for r in native})!=143: raise ValueError('Exact additional143 required')
    prepared=[]; q0_mismatch=0; support_mismatch=0
    metadata = []
    for entry in entries:
        f=entry['frame']; support=f['prediction_support']
        if isinstance(support,dict): support=support['BASE']
        metadata.append(dict(id=f['id'], session=f['session'], partition=f['partition'], image=f['image'],
            image_sha256=f['image_sha256'], raw_hw=f['raw_hw'], K=f['K'], xyz=f['xyz'],
            historical_YOLO_points=f['points']['BASE'], historical_PF_points=entry['PoseFix_points'],
            historical_support=support, N3_points=f['points']['N3_SUBPIX'], historical_initial_poses=f['initial_poses']))
    for entry in native:
        f=entry['frame']
        metadata.append(dict(id=f['id'],session=f['session'],partition='REMAINDER',image=f['image'],
            image_sha256=f['image_sha256'],raw_hw=f['raw_hw'],K=f['K'],xyz=f['known_physical_dimensions_xyz_m']))
    if len(metadata)!=238 or len({r['id']for r in metadata})!=238: raise ValueError('Unique238 source population required')
    for f in metadata:
        fid=f['id']; p=pf_inputs[fid]; pf=points[index[fid]]; q0=np.asarray(p['points'],float); support=np.asarray(p['valid'],bool)
        if any(p[k]!=f[k]for k in ('image','image_sha256','raw_hw')): raise ValueError('Original PF RGB metadata changed')
        if p['pad']!=0 or not p['usable'] or not p['detected']: raise ValueError('Unusable input must be explicitly handled')
        if not np.array_equal(pass0[index[fid]],q0,equal_nan=True): raise ValueError('Original RAW pass0 anchor changed')
        if q0.shape!=(9,2) or support.shape!=(9,): raise ValueError('Native9 schema')
        if not np.array_equal(pf[8],q0[8],equal_nan=True) or not np.array_equal(pf[~support],q0[~support],equal_nan=True):
            raise ValueError('PF center/missing support not preserved')
        if f['partition']!='REMAINDER':
            if not np.array_equal(np.asarray(f.pop('historical_YOLO_points'),float),q0,equal_nan=True): q0_mismatch+=1
            if not np.array_equal(np.asarray(f.pop('historical_support'),bool),support): support_mismatch+=1
            if not np.array_equal(np.asarray(f.pop('historical_PF_points'),float),pf,equal_nan=True):
                raise ValueError('Historical95 first uncapped PF tensor changed')
        prepared.append(dict(**f,YOLO_points=q0.tolist(),PoseFix_points=pf.tolist(),support=support.tolist(),
            seed=1,step=6000,neural_tensor='points[row,0,1]',NN_forward_executed=False,source_basis=False))
    if q0_mismatch or support_mismatch: raise ValueError('PF source differs from original95 anchors/support')
    # Source identity is checked against the pre-existing protocol before execution.
    sources = list(SOURCES)
    for relative in sources:
        expected=original_protocol['source_bindings'].get(str(Path('/home/minjae/Documents/github/pallet-pose')/relative))
        if expected and sha(repo/relative)!=expected: raise ValueError('Historical F/SubPix source changed: '+relative)
    inputs = [CACHE,PF/'INPUTS.json',OLD/'PROTOCOL.json',OLD/'BASELINE_INPUTS.json',NATIVE143,
              VIS/'remainder_after_test49_v1/native_input_preparation_v1/INPUT_SEAL.json']
    bindings = dict(schema='POSEFIX_FAIR_SOURCE_INPUT_BINDINGS_V1',repo=str(repo),
        sources={p:sha(repo/p)for p in sources},inputs={str(p):sha(repo/p)for p in inputs},
        replay_source=dict(path=str(Path(__file__).resolve()),sha256=sha(__file__)),
        image_bindings=[dict(id=r['id'],image=r['image'],sha256=r['image_sha256'])for r in prepared],
        checkpoint_sha256=CHECKPOINT,checkpoint_forward_needed=False,
        original95_YOLO_anchor_mismatches=0,original95_support_mismatches=0,
        raw_YOLO_anchor_source='PF INPUTS.json.points; native143 raw_points is N3+SubPix and is NEVER used as anchor',
        no_numeric_reference_or_success_label_reads=True)
    save(out/'POSEFIX_INPUTS238.jsonl.gz',prepared)
    write(out/'SOURCE_AND_INPUT_BINDINGS.json',bindings)
    protocol=dict(schema='POSEFIX_192_TWO_FIXED_COMPARATORS_PROTOCOL_LOCK_V1',
        before_new95_or192_SubPix_F=True,created_UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        comparator_decision='TWO_FIXED_COMPARATORS_NO_GLOBAL_WINNER',
        paths=list(METHODS),COMMON95=dict(CAL=46,TEST=49),PRIMARY192=dict(TEST=49,REMAINDER=143),CAL46_separate=True,
        ids_by_partition={p:[r['id']for r in prepared if r['partition']==p]for p in ('CAL','TEST','REMAINDER')},
        inputs_sha256=sha(out/'POSEFIX_INPUTS238.jsonl.gz'),bindings_sha256=sha(out/'SOURCE_AND_INPUT_BINDINGS.json'),
        checkpoint_sha256=CHECKPOINT,cache_tensor='points[selected_rows,0,1]',SubPix=SUBPIX,
        PF_ANCHOR='Native SubPix on uncapped PF points; cap only additional displacement from PF at .01*raw image diagonal',
        YOLO_ANCHOR='Same native SubPix result; cap total displacement from original YOLO at .01*raw image diagonal',
        cap_fraction=.01,center_and_unsupported_unchanged=True,finite_native8_F_rule_unchanged=True,
        F='Original pose.infer source=False; prediction-only W/D selection, SQPnP+RefineLM; physical Q from selected dimensions',
        labels_or_reference_in_selector=False,new_training=0,new_NN_forwards=0,
        allowed_new_operations='RGB grayscale + inherited cornerSubPix and existing F solves only',
        evaluation_process='separate; verify prediction and protocol seals before any reference decoding',
        no_primary_result_anchor_selection=True,no_threshold_or_weight_search=True,
        asymmetry='Synthetic-only PF and CAL-adapted e are different complete systems; all real sessions seen in development')
    write(private/'POSEFIX_192_PROTOCOL_LOCK.json',protocol)
    write(out/'PREPARATION_RECEIPT.json',dict(rows=238,counts=dict(Counter(r['partition']for r in prepared)),
        protocol_sha256=sha(private/'POSEFIX_192_PROTOCOL_LOCK.json'),timing=timing(start,cpu)))
    print(json.dumps(dict(status='LOCKED',rows=238,path=str(out),protocol_sha256=sha(private/'POSEFIX_192_PROTOCOL_LOCK.json'))))


def load_contract(repo,private):
    folder=private/'posefix_inputs_v1'; protocol=read(private/'POSEFIX_192_PROTOCOL_LOCK.json')
    bindings=read(folder/'SOURCE_AND_INPUT_BINDINGS.json')
    if sha(folder/'POSEFIX_INPUTS238.jsonl.gz')!=protocol['inputs_sha256'] or sha(folder/'SOURCE_AND_INPUT_BINDINGS.json')!=protocol['bindings_sha256']:
        raise ValueError('Frozen contract changed')
    for group in ('sources','inputs'):
        for p,digest in bindings[group].items():
            if sha(repo/p)!=digest: raise ValueError('Bound source/input changed: '+p)
    if sha(__file__)!=bindings['replay_source']['sha256']: raise ValueError('Locked replay implementation changed')
    return protocol,bindings,rows(folder/'POSEFIX_INPUTS238.jsonl.gz')


def execute(repo,private,stage):
    start,cpu=time.monotonic(),time.process_time()
    protocol,bindings,inputs=load_contract(repo,private)
    out=private/('posefix_parity95_v1'if stage=='infer95'else 'posefix192_v1');out.mkdir()
    cv2.setNumThreads(1)
    armset=PARITY_ARMS if stage=='infer95'else METHODS
    selected=[r for r in inputs if stage!='infer95'or r['partition']!='REMAINDER']
    if stage!='infer95':
        parity=read(private/'posefix_parity95_v1/POSEFIX_PARITY95.json')
        if not parity['passed']:raise ValueError('Successful95 parity prerequisite')
    output=[]; calls=Counter(); io_log=None
    with input_boundary(repo,out,protocol)as boundary:
        io_log=boundary
        sys.path.insert(0,str(repo))
        from challenge.evaluation_v2.pnp_selector import select_pnp_hypotheses
        sn=functions(repo/SOURCES[2],['cuboid','solve'],dict(np=np,cv2=cv2))
        pn=functions(repo/SOURCES[0],['rotations','infer'],dict(np=np,cv2=cv2,cuboid=sn['cuboid'],solve=sn['solve'],select_pnp_hypotheses=select_pnp_hypotheses))
        sub=functions(repo/SOURCES[1],['_inputs','cap_points','correct'],dict(np=np,cv2=cv2,Counter=Counter,METHODS=('SUBPIX','CVRANK'),_configuration=lambda:{'SUBPIX':SUBPIX}))
        for n,f in enumerate(selected):
            image_path=repo/f['image']
            if sha(image_path)!=f['image_sha256']:raise ValueError('RGB changed: '+f['id'])
            image=cv2.imread(str(image_path));calls['RGB_image_reads']+=1
            if image is None or list(image.shape[:2])!=f['raw_hw']:raise ValueError('RGB decode/canvas mismatch')
            boundary['decoded_input_paths'].append(f['image'])
            gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY);q0=np.asarray(f['YOLO_points'],float);pf=np.asarray(f['PoseFix_points'],float);support=np.asarray(f['support'],bool)
            native,diag=sub['correct'](gray,pf,support,'SUBPIX');calls['SubPix_images']+=1;calls['SubPix_corner_calls']+=diag['algorithm_corner_calls']
            allq=dict(POSEFIX_RAW=pf,POSEFIX_SUBPIX_PF_ANCHOR=sub['cap_points'](pf,native,image.shape[1],image.shape[0],support),
                POSEFIX_SUBPIX_YOLO_ANCHOR=sub['cap_points'](q0,native,image.shape[1],image.shape[0],support))
            if stage=='infer95':allq.update(NATIVE_BASE=q0,INITIAL_N3_SUBPIX=np.asarray(f['N3_points'],float))
            for method in armset:
                q=allq[method]
                if not np.array_equal(q[8],q0[8],equal_nan=True) or not np.array_equal(q[~support],q0[~support],equal_nan=True):raise ValueError('Native9 invariants violated')
                actual=pn['infer'](q,np.asarray(f['K']),np.asarray(f['xyz']),source=False);calls['F_calls']+=1
                output.append(dict(id=f['id'],partition=f['partition'],session=f['session'],method=method,
                    actual_pose=actual,native_points=q.tolist(),image=f['image'],image_sha256=f['image_sha256'],
                    K=f['K'],physical_dimensions_xyz_m=f['xyz'],support=support.tolist(),
                    native_SubPix_points=native.tolist()if method.startswith('POSEFIX_SUBPIX')else None,
                    SubPix=diag if method.startswith('POSEFIX_SUBPIX')else None,
                    status='AVAILABLE'if actual['available']else 'ORIGINAL_F_UNAVAILABLE',
                    fallback=False,new_model_forwards=0,new_training=0,GT_inputs=False,
                    source_tensor='points[selected_rows,0,1]',checkpoint_sha256=CHECKPOINT))
            if n%25==0:print('POSEFIX_PREDICTED',n+1,len(selected),flush=True)
        save(out/'PREDICTIONS.jsonl.gz',output)
        if stage!='infer95':
            save(out/'POSEFIX_192_PREDICTIONS.jsonl.gz',[r for r in output if r['partition']!='CAL'])
            save(out/'POSEFIX_CAL46_PREDICTIONS.jsonl.gz',[r for r in output if r['partition']=='CAL'])
        write(out/'IO_BOUNDARY.json',boundary)
    write(out/'PREDICTION_SEAL.json',dict(schema='POSEFIX_FAIR_PREDICTION_SEAL_V1',before_reference_decode=True,
        complete=True,before_scoring=True,stage=stage,rows=len(output),images=len(selected),methods=list(armset),
        ordered_ids=[r['id']for r in selected],
        artifacts={name:sha(out/name)for name in ('PREDICTIONS.jsonl.gz','IO_BOUNDARY.json')+(()if stage=='infer95'else ('POSEFIX_192_PREDICTIONS.jsonl.gz','POSEFIX_CAL46_PREDICTIONS.jsonl.gz'))},
        counts=dict(Counter(r['partition']for r in selected)),predictions_sha256=sha(out/'PREDICTIONS.jsonl.gz'),
        primary192_predictions_sha256=sha(out/'POSEFIX_192_PREDICTIONS.jsonl.gz')if stage!='infer95'else None,
        CAL46_predictions_sha256=sha(out/'POSEFIX_CAL46_PREDICTIONS.jsonl.gz')if stage!='infer95'else None,
        protocol_sha256=sha(private/'POSEFIX_192_PROTOCOL_LOCK.json'),input_bindings_sha256=protocol['bindings_sha256'],
        IO_boundary_sha256=sha(out/'IO_BOUNDARY.json'),operation_counts=dict(calls),timing=timing(start,cpu),
        original_F_failure_ids=[dict(id=r['id'],method=r['method'])for r in output if not r['actual_pose']['available']],
        accuracy_evaluation_executed=False,reference_inputs=False))
    print(json.dumps(dict(status='PREDICTIONS_SEALED',rows=len(output),path=str(out),timing=timing(start,cpu))))


def parity(repo,private):
    start,cpu=time.monotonic(),time.process_time();out=private/'posefix_parity95_v1'
    seal=read(out/'PREDICTION_SEAL.json')
    if sha(out/'PREDICTIONS.jsonl.gz')!=seal['predictions_sha256']:raise ValueError('Prediction seal changed')
    predictions=rows(out/'PREDICTIONS.jsonl.gz');old_path=repo/OLD/'PREDICTIONS.jsonl.gz';oldseal=read(repo/OLD/'PREDICTION_SEAL.json')
    if sha(old_path)!=oldseal['predictions_sha256']:raise ValueError('Legacy predictions changed')
    old={(r['id'],r['method']):r for r in rows(old_path)if r['method']in PARITY_ARMS}
    if len(old)!=475 or len(predictions)!=475:raise ValueError('Exact95x5 parity required')
    maxima=Counter();status=[];differences=[];exact_points=0;exact_pose=0
    fields=('R_cf','R_physical','centroid','cf_extents','reprojection_px')
    for r in predictions:
        o=old[r['id'],r['method']];a,b=r['actual_pose'],o['actual_pose']
        if a['available']!=b['available']:status.append(dict(id=r['id'],method=r['method']))
        pointdelta=float(np.nanmax(np.abs(np.asarray(r['native_points'],float)-np.asarray(o['native_points'],float))))
        maxima[r['method']+'/points_px']=max(maxima[r['method']+'/points_px'],pointdelta)
        if np.array_equal(np.asarray(r['native_points'],float),np.asarray(o['native_points'],float),equal_nan=True):exact_points+=1
        if a==b:exact_pose+=1
        for key in fields:
            if a.get('available')and b.get('available'):
                d=float(np.max(np.abs(np.asarray(a[key],float)-np.asarray(b[key],float))))
                maxima[r['method']+'/'+key]=max(maxima[r['method']+'/'+key],d)
                if d>=1e-10:differences.append(dict(id=r['id'],method=r['method'],field=key,max_abs=d))
        if a.get('selected_hypothesis')!=b.get('selected_hypothesis'):status.append(dict(id=r['id'],method=r['method'],field='selected_hypothesis'))
    # Independent additional143 RAW F cache parity is required after extend192.
    result=dict(schema='POSEFIX_RESTORED_COMMON95_REFERENCE_FREE_PARITY_V1',passed=not status and not differences and all(v<1e-10 for v in maxima.values()),
        images=95,methods=list(PARITY_ARMS),prediction_rows=475,pose_components_absolute_tolerance=1e-10,
        points_absolute_tolerance_px=1e-10,exact_point_rows=exact_points,exact_pose_dictionary_rows=exact_pose,
        maxima=dict(maxima),status_mismatches=status,numeric_mismatches=differences,
        original_prediction_sha256=sha(old_path),current_prediction_sha256=sha(out/'PREDICTIONS.jsonl.gz'),
        original_prediction_seal_sha256=sha(repo/OLD/'PREDICTION_SEAL.json'),
        no_reference_or_accuracy_reads=True,accuracy_counts_checked_by_separate_evaluator=True,timing=timing(start,cpu))
    write(out/'POSEFIX_PARITY95.json',result)
    if not result['passed']:raise ValueError('95 parity failed; preserve artifacts and inspect')
    print(json.dumps(dict(status='PARITY95_PASSED',exact_points=exact_points,exact_poses=exact_pose,max_abs=max(maxima.values(),default=0))))


def parity143(repo,private):
    out=private/'posefix192_v1';seal=read(out/'PREDICTION_SEAL.json')
    if sha(out/'PREDICTIONS.jsonl.gz')!=seal['predictions_sha256']:raise ValueError('Prediction seal changed')
    current={r['id']:r for r in rows(out/'PREDICTIONS.jsonl.gz')if r['partition']=='REMAINDER'and r['method']=='POSEFIX_RAW'}
    old={r['id']:r for r in rows(repo/RAW143)}
    if set(current)!=set(old)or len(old)!=143:raise ValueError('Exact143 PF RAW cache IDs')
    maxima=Counter();mismatches=[]
    for fid,r in current.items():
        for field in ('R_cf','R_physical','centroid','cf_extents','reprojection_px'):
            d=float(np.max(np.abs(np.asarray(r['actual_pose'][field])-np.asarray(old[fid]['actual_pose'][field]))))
            maxima[field]=max(maxima[field],d)
            if d>=1e-10:mismatches.append(dict(id=fid,field=field,difference=d))
    result=dict(schema='EXTENDED143_ORIGINAL_RAW_F_CACHE_PARITY_V1',passed=not mismatches,rows=143,maxima=dict(maxima),mismatches=mismatches,
        old_prediction_sha256=sha(repo/RAW143),new_prediction_sha256=seal['predictions_sha256'],no_reference_reads=True)
    write(out/'RAW143_CACHE_PARITY.json',result)
    if not result['passed']:raise ValueError('Original143 RAW F parity mismatch')
    print(json.dumps(result))


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('prepare','infer95','parity95','extend192','parity143'))
    p.add_argument('--repo',type=Path,required=True);p.add_argument('--private',type=Path,required=True);a=p.parse_args()
    repo,private=a.repo.resolve(),a.private.resolve()
    if a.stage=='prepare':prepare(repo,private)
    elif a.stage=='parity95':parity(repo,private)
    elif a.stage=='parity143':parity143(repo,private)
    else:execute(repo,private,a.stage)


if __name__=='__main__':main()
