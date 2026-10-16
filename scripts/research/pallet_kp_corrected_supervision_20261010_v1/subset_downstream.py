"""Scope the frozen original observation/pose pipeline to a sealed cohort.

Original head/66-way decoder/geometry algorithms are unchanged. The evaluator
is derived from its original AST with population constants only; GT access
starts after all selected geometry rows have been sealed. No work on import.
"""
import argparse
import ast
from collections import Counter
import contextlib
import copy
import gzip
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import textwrap
import time

sys.dont_write_bytecode=True
REPO=Path(__file__).resolve().parents[3]
DOC=REPO/'_docs/experiments/pallet_kp_corrected_supervision_20261010_v1'
OLD=REPO/'_docs/experiments/pallet_observation_refiner_20261009_v1'
REPAIR=REPO/'_docs/experiments/pallet_kp_supervision_repair_20261010_v1'
CODE=REPO/'scripts/research/pallet_observation_refiner_20261009_v1'
ARMS=('GEOMETRY_ONLY','IMAGE_NO_ROLE','IMAGE_ROLE')
METHODS=ARMS+('IMAGE_ROLE_NO_MASK_ROBUST','IMAGE_ROLE_STANDARD','IMAGE_ROLE_POINT_LINE')


def require(condition,message):
    if not condition:raise RuntimeError('SUBSET_GUARD: '+message)


def read(path):return json.loads(Path(path).read_text())


def rows(path):
    with gzip.open(path,'rt') as stream:
        for line in stream:
            if line.strip():yield json.loads(line)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def binding(path):
    path=Path(path).resolve()
    name=str(path.relative_to(REPO)) if path.is_relative_to(REPO) else path.name
    return dict(path=name,origin='public_repository' if path.is_relative_to(REPO) else 'external_readonly_dependency',sha256=sha(path),bytes=path.stat().st_size)


def bound(path,expected,label):
    require(Path(path).is_file(),label+' missing')
    require(sha(path)==expected['sha256'] and Path(path).stat().st_size==expected['bytes'],label+' SHA/bytes differs')


def write_new(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as stream:stream.write(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def prior_snapshot():
    p=DOC/'PRIOR_PUBLICATION_BINDINGS.json';data=read(p)
    require(data['protected_prior_files']==len(data['files'])==331,'original331 protection changed')
    for b in data['files']:
        path=(REPO/b['path']).resolve();require(path.is_relative_to(REPO),'prior path escapes repository');bound(path,b,'protected '+b['path'])
    return data['files']


def preflight(args):
    """Standard-library guards; no imports of models/CV2/Torch or pose calls."""
    cohort=read(args.cohort);frames=cohort['frames'];ids=[f['id'] for f in frames]
    count=cohort.get('count',cohort.get('N',len(ids)))
    require(count==len(ids)==len(set(ids)) and 0<count<319,'cohort population invalid')
    require(not cohort.get('ids') or cohort['ids']==ids,'cohort IDs/order differ')
    authority={f['id']:f for f in read(OLD/'INPUTS.json')['frames']}
    require(len(authority)==319 and set(ids)<=set(authority),'cohort not within frozen319 authority')
    for frame in frames:
        require(frame['session']==authority[frame['id']]['session'],'cohort session mismatch')
        require(frame['label'] in ('clean','moderate'),'excluded grade included')
        require(frame['image']['sha256']==authority[frame['id']]['image_sha256'],'cohort original image hash differs')
        if 'image_sha256' in frame:require(frame['image_sha256']==authority[frame['id']]['image_sha256'],'cohort image hash differs')
    protocol=read(args.protocol);completion_path=Path(args.fits)/'TRAINING_COMPLETION.json'
    require(completion_path.is_file(),'completed corrected training missing')
    completion=read(completion_path)
    require(completion.get('complete') and completion.get('formal_updates')==9000 and completion.get('total_updates')==9000,'authorized9000 completion invalid')
    require(completion['protocol']['sha256']==sha(args.protocol) and completion['input_hashes_unchanged'],'completed training protocol/input binding differs')
    require(completion.get('throwaway_updates')==0 and completion.get('batch')==16 and completion.get('seed')==1,'training settings differ')
    require(completion.get('same_initial_tensor_sha') and completion.get('same_batch_order') and completion.get('same_update_budget') and completion.get('source_scores_model_selection') is False,'fixed replay parity fields differ')
    approved=read(args.authorization);bound(args.authorization,completion['authorization'],'completed authorization')
    require(approved['status']=='APPROVED' and approved['additional_formal_updates']==9000 and bool(approved['user_authorization_evidence']),'approved9000 authorization invalid')
    require(approved['protocol_sha256']==sha(args.protocol) and approved['checks_sha256']==sha(REPAIR/'ZERO_UPDATE_CPU_CHECKS.json'),'approved frozen plan differs')
    for key,b in protocol['fixed_inputs'].items():
        if b['origin']=='public_repository':bound(REPO/b['path'],b,'frozen '+key)
    require([c['arm'] for c in completion['checkpoints']]==list(ARMS),'checkpoint arm/order differs')
    for c in completion['checkpoints']:
        require(c['updates']==3000 and c['exposures']==48000,'checkpoint learning budget differs')
        bound(Path(args.fits)/(c['arm']+'.pt'),c['checkpoint'],c['arm']+' checkpoint')
    if args.scope_protocol:
        scoped=read(args.scope_protocol)
        require(scoped['cohort']['sha256']==sha(args.cohort),'scope protocol cohort differs')
        require(scoped['subset_driver']['sha256']==sha(__file__),'scope protocol driver differs')
        require(scoped['training_protocol']['sha256']==sha(args.protocol),'scope training protocol differs')
    output=Path(args.output).resolve()
    roots=[REPO.resolve(),Path(args.source_root).resolve(),Path(args.baseline_root).resolve(),Path(args.fits).resolve(),
           Path('/dev/shm/pallet-observation-private-20261009'),Path('/dev/shm/pallet-kp-supervision-gate-private-20261010'),
           Path('/dev/shm/pallet-kp-difficulty-private-20261010')]
    require(all(not output.is_relative_to(r) and not r.is_relative_to(output) for r in roots),'output overlaps protected data/code/baseline/checkpoints')
    require(args.stage in ('infer','evaluate','preflight'),'unknown stage')
    require(args.stage!='evaluate' or output.is_dir(),'evaluation requires preceding selected inference directory')
    if output.exists():
        require(output.is_dir(),'output must be a directory')
        if args.stage=='infer':require(not any(output.iterdir()),'inference output must be empty')
        elif args.stage=='evaluate':
            previous=read(output/'SUBSET_INFER_ADAPTER_RECEIPT.json')
            require(previous['schema']=='subset_original_path_supervision_repair_adapter_v1' and previous['stage']=='infer','preceding subset inference receipt invalid')
            require(previous['cohort']['sha256']==sha(args.cohort) and previous['completed_training']['sha256']==sha(completion_path) and previous['training_protocol']['sha256']==sha(args.protocol),'evaluation must follow the same subset inference')
            require(not (output/'LEARNED_GEOMETRY_SEALED.jsonl.gz').exists() and not (output/'LEARNED_PREDICTIONS.jsonl.gz').exists() and not (output/'SUBSET_EVALUATE_STARTED.json').exists(),'preserve completed/interrupted evaluation')
    prior_snapshot()
    return dict(cohort=cohort,ids=ids,count=count,authority=authority,completion_path=completion_path,
                output=output,checkpoints=[binding(Path(args.fits)/(a+'.pt')) for a in ARMS])


def evaluator_function(E,count,scoped_loader):
    """Only original run population literals and the reference loader are scoped."""
    tree=ast.parse(textwrap.dedent(inspect.getsource(E.run)));changes=[]
    class Scope(ast.NodeTransformer):
        def visit_Compare(self,node):
            node=self.generic_visit(node)
            if (isinstance(node.left,ast.Call) and isinstance(node.left.func,ast.Name) and node.left.func.id=='len'
                and len(node.left.args)==1 and isinstance(node.left.args[0],ast.Name) and node.left.args[0].id=='rows'
                and len(node.comparators)==1 and isinstance(node.comparators[0],ast.Constant) and node.comparators[0].value==1914):
                node.comparators[0]=ast.Constant(count*6);changes.append('assert len(rows):1914->6N')
            return node
        def visit_keyword(self,node):
            node=self.generic_visit(node)
            old={'point_solver_paths':1595,'point_line_paths':319}
            if node.arg in old and isinstance(node.value,ast.Constant) and node.value.value==old[node.arg]:
                node.value=ast.Constant(count*(5 if node.arg=='point_solver_paths' else 1));changes.append(node.arg+':'+str(old[node.arg])+'->'+('5N' if node.arg=='point_solver_paths' else 'N'))
            return node
        def visit_Call(self,node):
            node=self.generic_visit(node)
            if isinstance(node.func,ast.Name) and node.func.id=='load_real' and not node.args and not node.keywords:
                node.func=ast.Name('_scoped_load_real',ast.Load());changes.append('load_real()->selected-only cached reference loader')
            return node
    derived=ast.fix_missing_locations(Scope().visit(tree))
    require(sorted(changes)==sorted(['assert len(rows):1914->6N','point_solver_paths:1595->5N','point_line_paths:319->N','load_real()->selected-only cached reference loader']),'unexpected original evaluation AST shape')
    namespace=dict(E.__dict__,_scoped_load_real=scoped_loader)
    exec(compile(derived,str(Path(__file__))+'::scoped_original_evaluation','exec'),namespace)
    return namespace['run'],dict(original_code=binding(CODE/'learned_evaluate.py'),changes=changes,
                                 derived_function_AST_sha256=hashlib.sha256(ast.dump(derived,include_attributes=False).encode()).hexdigest())


def reference_loader(C,scope,audit):
    """Validate the319 authority; open actual cached-frame reference only for N."""
    path=scope['output']/'LEARNED_GEOMETRY_SEALED.jsonl.gz'
    raw=list(rows(path));ids=scope['ids'];selected=set(ids)
    require(len(raw)==scope['count']*6 and Counter(r['method'] for r in raw)=={m:scope['count'] for m in METHODS},'GT requested before complete selected geometry seal')
    require({r['id'] for r in raw}==selected,'geometry seal has excluded IDs')
    from scripts.research.pallet_training_free_compare_20261007_v1 import common as real
    E,baseline_root=real.legacy()
    data=object.__new__(E.Data);data.root=C.ROOT;data.dim=C.ROOT/'data/pallet/results/pallet_dim_conditioned_p_v1'
    # Keep every original per-frame operation, filter the sorted axis list
    # before its cache/annotation loop. No excluded cached frame is opened.
    tree=ast.parse(textwrap.dedent(inspect.getsource(E.Data.real_frames)));changes=[]
    class Frames(ast.NodeTransformer):
        def visit_For(self,node):
            node=self.generic_visit(node)
            if isinstance(node.iter,ast.Call) and isinstance(node.iter.func,ast.Name) and node.iter.func.id=='sorted':
                node.iter=ast.Call(ast.Name('_selected_axis',ast.Load()),[node.iter],[]);changes.append('filter sorted319 axis before cached-frame loop')
            return node
    derived=ast.fix_missing_locations(Frames().visit(tree));require(len(changes)==1,'unexpected immutable Data.real_frames shape')
    def select_axis(axis):
        require(len(axis)==319 and {r['frame_id'] for r in axis}==set(scope['authority']),'original319 reference authority differs')
        filtered=[r for r in axis if r['frame_id'] in selected]
        require({r['frame_id'] for r in filtered}==selected,'selected reference IDs missing');return filtered
    namespace=dict(E.Data.real_frames.__globals__,_selected_axis=select_axis)
    exec(compile(derived,str(Path(__file__))+'::selected_reference_frames','exec'),namespace)
    frames=namespace['real_frames'](data);require(len(frames)==scope['count'],'selected reference frame count differs')
    target_path=C.ROOT/'data/pallet/results/pallet_posefix_replay_diagnosis_v1/TARGETS.json';targets=real.read(target_path)
    baseline_path=baseline_root/'_docs/experiments/pallet_joint_action_handoff_20261006_v1/results/A_REAL_DEV_BASELINES.json';baselines=real.read(baseline_path)
    require(baselines['target_sha256']==sha(target_path),'frozen reference targets differ')
    authority=set(scope['authority'])
    require(all(len(rr)==319 and {r['id'] for r in rr}==authority for rr in baselines['rows'].values()),'frozen319 reference baseline IDs differ')
    require(sum(sum(targets[i]['valid'][:8]) for i in authority)==2499 and sum(targets[i]['matched'] for i in authority)==311,'frozen319 target authority changed')
    require(sum(len(r['corner']['observed_errors']) for r in baselines['rows']['RAW'])==2445,'frozen319 corner baseline count changed')
    baselines=copy.copy(baselines);baselines['rows']={k:[r for r in rr if r['id'] in selected] for k,rr in baselines['rows'].items()}
    audit.update(reference_scope=dict(authority_metadata_frames=319,actual_cached_reference_frames=len(frames),
                 excluded_cached_reference_frames=0,actual_scored_frames=scope['count'],after_geometry_seal=True,
                 targets=binding(target_path),baselines=binding(baseline_path),
                 data_frame_math=binding(inspect.getsourcefile(E.Data.real_frames)),
                 derived_frame_AST_sha256=hashlib.sha256(ast.dump(derived,include_attributes=False).encode()).hexdigest(),changes=changes,
                 reference='GEOMETRIC_PROXY; not independent physically measured truth'))
    return E,frames,{fid:targets[fid] for fid in ids},baselines,baseline_path


@contextlib.contextmanager
def primitives(cv2,point_line=None):
    counts={'real':Counter(),'unit_checks':Counter()};state={'phase':'real'};saved={}
    for name in ('solvePnP','solvePnPGeneric','solvePnPRefineLM'):
        fn=getattr(cv2,name);saved[name]=fn
        def wrap(*a,_name=name,_fn=fn,**kw):counts[state['phase']][_name]+=1;return _fn(*a,**kw)
        setattr(cv2,name,wrap)
    least=None
    if point_line is not None:
        least=point_line.least_squares
        def optimizer(fun,*a,**kw):
            phase=state['phase'];counts[phase]['scipy_least_squares_calls']+=1
            def callback(*x,**y):counts[phase]['optimizer_residual_callback_calls']+=1;return fun(*x,**y)
            result=least(callback,*a,**kw);counts[phase]['scipy_reported_nfev']+=int(result.nfev);return result
        point_line.least_squares=optimizer
    try:yield counts,state
    finally:
        for name,fn in saved.items():setattr(cv2,name,fn)
        if point_line is not None:point_line.least_squares=least


def run(args):
    scope=preflight(args);before=prior_snapshot();output=scope['output'];count=scope['count'];start=time.monotonic()
    output.mkdir(parents=True,exist_ok=True)
    write_new(output/('SUBSET_'+args.stage.upper()+'_STARTED.json'),dict(stage=args.stage,cohort=binding(args.cohort),configured_frames=count,actual_pipeline_calls=0))
    os.environ['PALLET_SOURCE_ROOT']=str(Path(args.source_root).resolve());os.environ['PALLET_BASELINE_ROOT']=str(Path(args.baseline_root).resolve())
    from scripts.research.pallet_observation_refiner_20261009_v1 import common as C
    require(C.ROOT==Path(args.source_root).resolve(),'imported original sourceRoot differs')
    saved={name:getattr(C,name) for name in ('DOC','SCRATCH','read','iter_rows')};audit={};restore=[]
    selected=set(scope['ids']);frozen_views={
        'INPUTS.json':dict(schema='selected_original_frozen_inputs_v1',GT_input=False,frames=[scope['authority'][fid] for fid in scope['ids']]),
        'TRAINING_COMPLETION.json':read(scope['completion_path'])}
    def input_read(path):
        p=Path(path)
        return frozen_views[p.name] if p.parent.resolve()==output and p.name in frozen_views else saved['read'](p)
    def input_rows(path):
        p=Path(path)
        if p.parent.resolve()==output and p.name in ('OBSERVATIONS.jsonl.gz','FIXED_CONTROLS.jsonl.gz'):
            return (r for r in saved['iter_rows'](OLD/p.name) if r['id'] in selected)
        return saved['iter_rows'](p)
    C.DOC=output;C.SCRATCH=output.parent/'scratch';C.read=input_read;C.iter_rows=input_rows
    try:
        import cv2
        if args.stage=='infer':
            from scripts.research.pallet_observation_refiner_20261009_v1 import learned_infer as L
            fits=L.FITS;restore.append(lambda:setattr(L,'FITS',fits));L.FITS=Path(args.fits).resolve()
            original_argv=sys.argv;sys.argv=[sys.argv[0],'--output','LEARNED_OBSERVATIONS.jsonl.gz'];restore.append(lambda:setattr(sys,'argv',original_argv))
            with primitives(cv2) as (calls,state):L.main()
            seal=read(output/'OBSERVATION_SEAL.json');raw=list(rows(output/'LEARNED_OBSERVATIONS.jsonl.gz'))
            require(seal['frames']==count and seal['rows']==3*count and seal['detector_forward']==count,'selected inference counts differ')
            require(Counter(r['method'] for r in raw)=={a:count for a in ARMS} and {r['id'] for r in raw}==selected,'selected observation IDs differ')
            require(all(r['GT_input'] is False and r['detector_available'] for r in raw),'selected detector/parity/GT contract failed')
            audit.update(detector_forwards=count,head_forwards=3*count,feature_initial_pose_calls=count,
                         observation_seal=binding(output/'OBSERVATION_SEAL.json'),observations=binding(output/'LEARNED_OBSERVATIONS.jsonl.gz'))
        else:
            from scripts.research.pallet_observation_refiner_20261009_v1 import learned_evaluate as E,point_line as P
            original_tests=E.line_tests
            with primitives(cv2,P) as (calls,state):
                def unit_tests():
                    state['phase']='unit_checks'
                    try:return original_tests()
                    finally:state['phase']='real'
                E.line_tests=unit_tests;restore.append(lambda:setattr(E,'line_tests',original_tests))
                fn,proof=evaluator_function(E,count,lambda:reference_loader(C,scope,audit));fn()
            execution=read(output/'LEARNED_POSE_EXECUTION.json');sealed=list(rows(output/'LEARNED_GEOMETRY_SEALED.jsonl.gz'));scored=list(rows(output/'LEARNED_PREDICTIONS.jsonl.gz'))
            require(execution['rows']==6*count and execution['point_solver_paths']==5*count and execution['point_line_paths']==count,'selected geometry path counts differ')
            require(Counter(r['method'] for r in sealed)=={m:count for m in METHODS} and {r['id'] for r in sealed}==selected and len(scored)==6*count,'selected geometry/scoring population differs')
            require(not any(k in r for r in sealed for k in ('pose','corner','mask_audit','baseline_pose','baseline_corner')),'geometry seal contains posthocGT fields')
            audit.update(derived_evaluator=proof,point_solver_paths=5*count,point_line_paths=count,
                         geometry=binding(output/'LEARNED_GEOMETRY_SEALED.jsonl.gz'),predictions=binding(output/'LEARNED_PREDICTIONS.jsonl.gz'),pose_execution=binding(output/'LEARNED_POSE_EXECUTION.json'))
        require(prior_snapshot()==before,'original331 protected files changed')
        write_new(output/('SUBSET_'+args.stage.upper()+'_ADAPTER_RECEIPT.json'),dict(
            schema='subset_original_path_supervision_repair_adapter_v1',complete=True,stage=args.stage,
            cohort=binding(args.cohort),training_protocol=binding(args.protocol),protocol=binding(args.protocol),completed_training=binding(scope['completion_path']),
            scope_protocol=binding(args.scope_protocol) if args.scope_protocol else None,driver=binding(__file__),
            frames=count,sessions=len({f['session'] for f in scope['cohort']['frames']}),models=list(ARMS),methods=list(METHODS),
            original319_authority_verified=True,excluded_frames_executed=0,protected_prior_files_preserved=331,
            primitive_counts={k:dict(v) for k,v in calls.items()},primitive_count_definition='actual OpenCV solver entry calls and SciPy optimizer/residual callbacks; callbacks exclude precheck evaluations outside SciPy; scipy_reported_nfev excludes finite difference callbacks',
            wall_seconds=time.monotonic()-start,runtime_benchmark=False,**audit))
        print('SUBSET_'+args.stage.upper()+'_COMPLETE',count,'excluded_calls=0',flush=True)
    finally:
        for fn in reversed(restore):fn()
        for name,value in saved.items():setattr(C,name,value)


def parser():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=('preflight','infer','evaluate'))
    p.add_argument('--cohort',default=str(DOC/'COHORT.json'));p.add_argument('--scope-protocol',default=str(DOC/'SUBSET_PROTOCOL.json'))
    p.add_argument('--protocol',default=str(REPAIR/'RETRAINING_PROTOCOL.json'))
    p.add_argument('--fits',default='/dev/shm/pallet-kp-supervision-repair-private-20261010/learned_fits')
    p.add_argument('--authorization',default=str(DOC/'APPROVED_ADDITIONAL_9000.json'))
    p.add_argument('--output',default='/dev/shm/pallet-kp-corrected-supervision-private-20261010/downstream')
    p.add_argument('--source-root',default=os.environ.get('PALLET_SOURCE_ROOT'));p.add_argument('--baseline-root',default=os.environ.get('PALLET_BASELINE_ROOT'))
    return p


if __name__=='__main__':
    arguments=parser().parse_args()
    if arguments.stage=='preflight':
        result=preflight(arguments);print(json.dumps(dict(frames=result['count'],ids=result['ids'],completed_training=binding(result['completion_path']),no_models_imported=True)))
    else:run(arguments)
