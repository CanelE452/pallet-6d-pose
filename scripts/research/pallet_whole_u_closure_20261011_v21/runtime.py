"""Fixed fresh V21 point closure, with inner pipeline closed before outer context.

ROOT alone freezes and measures. Frozen V7 detector/N3/ROLE/CAL/H-union-k
selection supplies the actual selected in-memory point bank. One point closure
is timed; endpoint/line solves and saved accuracy inputs are excluded.
"""
import argparse
import ast
from collections import Counter
from contextlib import contextmanager
import copy
import hashlib
import importlib
import importlib.util
import io
import builtins
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

HERE=Path(__file__).resolve().parent
PRIMARY='HYBRID_POINT_CONSENSUS_WHOLE_U_CLOSURE'
PRIMITIVES=('solvePnP','solvePnPGeneric','solvePnPRefineLM','projectPoints','Rodrigues','cornerSubPix')
STAGES=('detector','N3_correction','initial_pose','boundary_observation','hybrid_selection_and_point_consensus','final_pose_reprojection_and_metadata')
SOURCE_SHA=dict(deployment='3d03d1073fd16629514665c5fe990e6f6b8826b8cd8ac9989705aac978485fe7',runtime='0465082112b0ceda8d00b204c7585d5c12218ada1118086b90abc061f28fa6dd')
PRIVATE_SHA=dict(solver='1603401ed1f7b1cde301b272bfc1f2f6d3c3935f371ba9c7bafd56e2a893d099',
 pipeline='570ff5feece0a10cb81ba1685141af8f0334751a3b0d903f7514a382f242774f')
CLASS_METHODS=('__init__','torch','runtime','old','heads','models','bindings','capture_point_banks','packet','close')

def require(value,message):
    if not value:raise RuntimeError('V21_RUNTIME: '+message)

def binding(path):
    path=Path(path);h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return dict(path=str(path.resolve()),bytes=path.stat().st_size,sha256=h.hexdigest())

def read(path):return json.loads(Path(path).read_bytes())

def same(left,right):return all(left[k]==right[k] for k in ('bytes','sha256'))

def write(path,value):
    with Path(path).open('x') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())

def source_paths(a):
    root=Path(a.repo)/'scripts/research'
    return dict(deployment=root/'pallet_hybrid_factor_subset_consensus_20261011_v16/deployment.py',
                runtime=root/'pallet_consensus_decision_repair_20261011_v18/runtime.py')

def source_nodes(a):
    paths=source_paths(a);result={}
    for key,names in (('deployment',('assemble_result','Pipeline')),('runtime',('schedule','instruments','delta'))):
        require(binding(paths[key])['sha256']==SOURCE_SHA[key],'immutable source '+key)
        tree=ast.parse(paths[key].read_text());nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
        require({n.name for n in nodes}==set(names),'complete reused source '+key);result[key]=nodes
    return paths,result

def bookkeeping(a):
    paths,nodes=source_nodes(a)
    env=dict(__name__='isolated_v21_bookkeeping',__package__='scripts.research.pallet_consensus_decision_repair_20261011_v18',
             contextmanager=contextmanager,R=SimpleNamespace(PRIMITIVES=PRIMITIVES))
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes['runtime'],type_ignores=[])),str(paths['runtime']),'exec'),env)
    return env,{k:{n.name:hashlib.sha256(ast.dump(n,include_attributes=False).encode()).hexdigest() for n in v} for k,v in nodes.items()}

def protect(a):
    result={}
    for key,root in (('research',a.repo),('original',a.source_root)):
        def git(*v):return subprocess.check_output(['git','-C',str(root),*v])
        result[key]=dict(HEAD=git('rev-parse','HEAD').decode().strip(),index=hashlib.sha256(git('ls-files','--stage','-z')).hexdigest(),
            diff=hashlib.sha256(git('diff','--binary')).hexdigest(),status=hashlib.sha256(git('status','--porcelain=v1','-uall')).hexdigest())
    return result

def setup(a):
    sys.path.insert(0,str(a.repo));sys.dont_write_bytecode=True
    C=importlib.import_module('scripts.research.pallet_sparse_local_line_20261010_v9.common')
    V7=importlib.import_module('scripts.research.pallet_three_head_observation_20261010_v7.common')
    helpers=importlib.import_module('scripts.research.pallet_sparse_local_line_20261010_v9.runtime_review')
    previous=importlib.import_module('scripts.research.pallet_three_head_observation_20261010_v7.runtime')
    book,witness=bookkeeping(a);return C,V7,helpers,previous,book,witness

def live_repair(point, native, selected, camera, dimensions, observation, H, contract, bank, size, source):
    require(bank.known_dimension_index is None and bank.image_size == size, 'unconstrained same-size actual selected bank')
    require(source.P16.np.array_equal(bank.points, selected, equal_nan=True)
        and source.P16.np.array_equal(bank.K, camera) and source.P16.np.array_equal(bank.xyz, dimensions), 'same q/K/xyz bank')
    require(point['hidden_initial'] == H and point['selected_corner_ids'] == contract['hybrid_boundary_corner_ids'], 'actual frozen corner selection and H')
    before=copy.deepcopy(bank.ledger);U=tuple(k for k in bank.eligible if k not in H)
    clean=copy.deepcopy(observation);clean['lines']=[]
    subject=source.S21.PointClosureSolver(bank)
    solved=subject.solve(U,hidden=H,line_blocks=(),observation=clean,initial_pose=None,point_result=None)
    after=copy.deepcopy(bank.ledger);delta={k:after[k]-before[k] for k in after}
    ordinary=solved['closure_ledger'];extra=ordinary.get('bank_operation_counts',{k:0 for k in after})
    raw4={k:delta[k]-extra[k] for k in delta}
    require(raw4['generic_calls']==raw4['generic_cache_misses']==raw4['lm_calls']==0,
        'actual identical four-ID roots must reuse the original fresh bank')
    require(extra['generic_calls']<=len(bank.dims) and extra['lm_calls']<=1, 'dedicated whole-U closure extra bound')
    require(solved['line_edges']==solved['fit_line_edges']==solved['consumed_edges']==[] and 8 not in solved['fit_input_ids']
        and set(H).isdisjoint(solved['fit_input_ids']), 'point-only actual fit excludes H/center')
    ledger=dict(schema='fresh_hybrid_whole_U_closure_ledger_v21',actual_solver_calls=1,reported_output_views=1,
        used=list(U),initial_H=H,point_solver=copy.deepcopy(solved),point_solver_counts=copy.deepcopy(subject.counts),
        local_operation_counts=copy.deepcopy(solved['operation_counts']),point_bank_before=before,point_bank_after=after,
        point_bank_operation_counts=delta,raw_four_ID_bank_operation_counts=raw4,
        whole_U_closure_bank_operation_counts=extra,closure_ledger=copy.deepcopy(ordinary),
        new_endpoint_validation_calls=0,new_line_solve_calls=0,point_available_verdict_not_a_root_filter=True,
        policy=copy.deepcopy(source.POLICY),source_proof=copy.deepcopy(source.SOURCE_PROOF),
        saved_accuracy_row_used=False,extra_independent_observations=0)
    return solved,ledger

def assemble_fresh(point,native,selected,initial,H,contract,solved,ledger,source):
    row={k:copy.deepcopy(point[k]) for k in ('id','session','method','K','xyz','raw_hw','selected_index','fixed_metadata',
        'head_arm','head_mode','head_checkpoint_sha256','calibration_binding') if k in point}
    row.update(source.P16.assemble(native,selected,initial,H,solved,'N3_VALIDATED_ROLE',contract))
    row.update(method=PRIMARY,parent_method=source.PARENT_METHOD,observation_supply='CORNERWISE_HYBRID',
        selected_corner_ids=list(contract['hybrid_boundary_corner_ids']),original_native_N3_points=source.P16.np.asarray(native).tolist(),
        predicted_initial_N3_hidden=list(H),original_self_hidden_initial=list(H),initial_H_applied=True,
        hybrid_consensus_policy=copy.deepcopy(source.POLICY),cornerwise_selection=copy.deepcopy(contract['cornerwise_selection']),
        whole_U_closure_pipeline_source_proof=copy.deepcopy(source.SOURCE_PROOF),
        observation_raw_logits_sha256=point.get('observation_raw_logits_sha256'),pose_estimation_kind=PRIMARY,
        native_N3_is_numeric_final_pose_observation=True,missing_sparse_filled_for_numeric_fit=False,
        partial_lines_not_pose_inputs=True,local_point_line_refinement=False,
        initial_pose_may_supply_local_optimizer_start=False,final_numeric_pose_has_initial_prior=False,
        numerical_NEW_is_not_pose_accuracy_success=True,global_uniqueness_proven=False,
        fresh_complete_capture=True,fresh_hybrid_corner_selection_recomputed_inside_call=True,
        fresh_point_solver_recomputed_inside_call=True,fresh_endpoint_LOO_recomputed_inside_call=False,
        fresh_raw_selected_point_roots_reused_inside_call=True,old_point_control_computed_by_original_selection=True,
        paired_point_accuracy_arm_executed_inside_primary_call=False,actual_fit_incidence_guard_applied_per_candidate=True,
        parent_completion_receipt_used_as_deployment_input=False,saved_accuracy_coordinates_or_pose_as_input=False,
        cached_accuracy_row_used=False,solver_replay_latency_claim=False,runtime_only_fresh_metadata=True,oracle=False)
    return row

def fresh_predict(self,image,K,xyz,metadata=None,method=PRIMARY,stage_callback=None):
    P.require(method==P.PRIMARY,'single fresh point closure primary')
    self.last_capture=self.last_point_result=self.last_point_bank_ledger=self.last_repair_ledger=None
    self.last_point_bank_reuse=None;self.last_execution_events=[]
    self.counts['fresh_predict_calls_started']+=1;self.counts[method+':started']+=1
    with self.old.no_truth_reads(),self.torch.no_grad():
        capture=self.base.capture(image,K,xyz,metadata,need_arms=('IMAGE_ROLE',),need_n3=True,need_base_pose=False,stage_callback=stage_callback)
        self.last_capture=capture;self.counts['fresh_hybrid_point_paths_started']+=1
        with self.capture_point_banks() as banks:rows,point_ledger=self.base.outcomes(capture,(P.PARENT_METHOD,))
        P.require(len(rows)==1 and 1<=len(banks)<=2,'one original hybrid point route and actual bank registry')
        point=rows[0];self.last_point_result,self.last_point_bank_ledger=point,point_ledger
        selected=np.asarray(point['input_points'],float);native=np.asarray(capture['native_N3_points'],float)
        size=(image.shape[1],image.shape[0]);camera=np.asarray(K,float);dimensions=np.asarray(xyz,float)
        def matches(bank,points):return np.array_equal(bank.points,points,equal_nan=True) and np.array_equal(bank.K,camera) and np.array_equal(bank.xyz,dimensions) and bank.image_size==size and bank.known_dimension_index is None
        selected_banks=[b for b in banks if matches(b,selected)];native_banks=[b for b in banks if matches(b,native)]
        P.require(len(selected_banks)==len(native_banks)==1,'unique actual selected and native bank identity')
        shared,native_bank=selected_banks[0],native_banks[0];same_bank=shared is native_bank
        native_before=copy.deepcopy(native_bank.ledger);selected_cache_before=set(shared.numeric_cache)
        self.counts['fresh_hybrid_point_paths_complete']+=1
        observation=capture['observations']['IMAGE_ROLE'];contract=point['observation_contract'];H=list(capture['hidden'])
        self.counts['fresh_repair_paths_started']+=1;self.counts['fresh_point_closure_paths_started']+=1
        self.last_execution_events.append(dict(phase='whole_U_point_closure_call_started'))
        solved,ledger=live_repair(point,native,selected,camera,dimensions,observation,H,contract,shared,size,SOURCE)
        native_after=copy.deepcopy(native_bank.ledger);selected_cache_after=set(shared.numeric_cache)
        new_keys=sorted(selected_cache_after-selected_cache_before);whole=ledger['whole_U_closure_bank_operation_counts']
        P.require(selected_cache_before<=selected_cache_after and whole['generic_cache_misses']==len(new_keys), 'same point problem cache preservation')
        P.require(all(tuple(ids)==tuple(ledger['used']) for _,ids in new_keys),'only whole-U new numeric problems permitted')
        reuse=dict(schema='fresh_selected_point_closure_bank_reuse_v21',actual_constructor_count=len(banks),constructor_restored=True,
            same_coordinate_bank_reused=True,selected_native_bank_is_same_object=same_bank,
            selected_input_hash=shared.digest,native_input_hash=native_bank.digest,
            selected_bank_before=copy.deepcopy(ledger['point_bank_before']),selected_bank_after=copy.deepcopy(ledger['point_bank_after']),
            selected_bank_delta=copy.deepcopy(ledger['point_bank_operation_counts']),
            raw_four_ID_bank_delta=copy.deepcopy(ledger['raw_four_ID_bank_operation_counts']),whole_U_closure_bank_delta=copy.deepcopy(whole),
            native_bank_before=native_before,native_bank_after=native_after,
            native_bank_delta={k:native_after[k]-native_before[k] for k in native_after},
            selected_numeric_cache_before=[dict(dimension_index=d,point_ids=list(ids)) for d,ids in sorted(selected_cache_before)],
            selected_numeric_cache_after=[dict(dimension_index=d,point_ids=list(ids)) for d,ids in sorted(selected_cache_after)],
            newly_requested_selected_numeric_problems=[dict(dimension_index=d,point_ids=list(ids)) for d,ids in new_keys],
            additional_raw_four_ID_Generic_calls=ledger['raw_four_ID_bank_operation_counts']['generic_calls'],
            additional_whole_U_Generic_calls=whole['generic_calls'],additional_whole_U_LM_calls=whole['lm_calls'],
            new_endpoint_validation_calls=0,new_line_solve_calls=0,point_available_verdict_not_a_root_filter=True,
            cache_reconstructed_from_serialized_Rt=False,original_cornerwise_policy=copy.deepcopy(P.S4.POLICY),
            fresh_cornerwise_selection=copy.deepcopy(contract['cornerwise_selection']))
        P.require(reuse['native_bank_delta']==(reuse['selected_bank_delta'] if same_bank else {k:0 for k in native_after}), 'unused native bank remains unchanged')
        self.last_point_bank_reuse=reuse;ledger['fresh_actual_point_bank_reuse']=copy.deepcopy(reuse)
        self.last_repair_ledger=ledger;self.local_operation_counts.update(ledger['local_operation_counts'])
        self.direct_point_operation_counts.update(reuse['raw_four_ID_bank_delta']);self.whole_U_operation_counts.update(whole)
        self.last_execution_events.append(dict(phase='whole_U_point_closure_call_returned'))
        self.counts['fresh_point_closure_paths_complete']+=1;self.counts['fresh_repair_paths_complete']+=1
        result=self.packet(capture['metadata'],capture,assemble_fresh(point,native,selected,capture['initial_pose'],H,contract,solved,ledger,SOURCE))
        result.update(fresh_point_solver=copy.deepcopy(point['solver']),fresh_point_coordinate_bank_ledger=copy.deepcopy(point_ledger),
            fresh_repair_ledger=copy.deepcopy(ledger),fresh_actual_point_bank_reuse=copy.deepcopy(reuse))
        self.counts['reported_primary_views_from_single_point_closure']+=1
        if stage_callback:stage_callback('hybrid_selection_and_point_consensus')
        prediction=copy.deepcopy(capture['prediction']);index=prediction['selected_index']
        if index is not None:prediction['candidates'][index]['keypoints_xy']=result['native_points']
        preserve_prediction(capture['raw'],prediction)
        P.require(np.array_equal(np.asarray(result['native_points'])[8],native[8],equal_nan=True),'detector center changed')
        if stage_callback:stage_callback('final_pose_reprojection_and_metadata')
        result.update(prediction=prediction,original_base_points=capture['original_base_points'],native_N3_points=capture['native_N3_points'],
            observations=capture['observations'],observation=capture['observation'],fresh_capture_input_only=True)
    self.counts['fresh_predict_calls_complete']+=1;self.counts[method+':complete']+=1
    return result

def pipeline_class(a):
    parent=importlib.import_module('scripts.research.pallet_hybrid_factor_subset_consensus_20261011_v16.deployment')
    paths,nodes=source_nodes(a);path=HERE/'pipeline.py';require(binding(path)['sha256']==PRIVATE_SHA['pipeline'],'CPU-tested point packet adapter')
    require(binding(HERE/'solver.py')['sha256']==PRIVATE_SHA['solver'],'CPU-tested point closure')
    name='private_v21_runtime_point_pipeline';spec=importlib.util.spec_from_file_location(name,path)
    require(spec is not None and spec.loader is not None,'explicit sibling pipeline source');current=importlib.util.module_from_spec(spec)
    previous=sys.modules.get(name);sys.modules[name]=current
    try:spec.loader.exec_module(current)
    except BaseException:
        if previous is None:sys.modules.pop(name,None)
        else:sys.modules[name]=previous
        raise
    original=next(n for n in nodes['deployment'] if isinstance(n,ast.ClassDef) and n.name=='Pipeline')
    cls=copy.deepcopy(original);cls.body=[copy.deepcopy(n) for n in original.body if isinstance(n,ast.FunctionDef) and n.name in CLASS_METHODS]
    require({n.name for n in cls.body}==set(CLASS_METHODS),'complete unchanged frontend/capture/packet/close methods')
    prediction=next(n for n in ast.parse(Path(__file__).read_bytes()).body if isinstance(n,ast.FunctionDef) and n.name=='fresh_predict');prediction=copy.deepcopy(prediction);prediction.name='predict';cls.body.append(prediction)
    facade=SimpleNamespace(PRIMARY=current.PRIMARY,PARENT_METHOD=current.PARENT_METHOD,POLICY=current.POLICY,
        S4=current.P16.S4,PoseBank=current.P16.PoseBank,require=current.require)
    env=dict(vars(parent));env.update(P=facade,PRIMARY=PRIMARY,SOURCE=current,live_repair=live_repair,assemble_fresh=assemble_fresh,__name__='isolated_v21_fresh_point_deployment')
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls],type_ignores=[])),str(paths['deployment'])+':unchanged frontend plus point dispatch','exec'),env)
    result=env['Pipeline'];init=result.__init__
    def initialize(self,args):init(self,args);self.whole_U_operation_counts=Counter()
    result.__init__=initialize
    result._private_loader_previous=previous;result._private_loader_name=name
    result.source_isolation_proof=dict(unchanged_frontend_class_methods=list(CLASS_METHODS),
        frontend_method_AST_changes=0,predict_replaced_by_live_point_only_dispatch=True,
        unused_endpoint_methods_not_loaded=True,parent_class_or_module_mutated=False,
        inherited_frontend_source=binding(paths['deployment']),private_point_pipeline=binding(path))
    return result,current.PRIMARY

def gate(a):
    g,v=Path(a.geometry),Path(a.validation)
    core,start,r=[read(g/n) for n in ('GEOMETRY_PROTOCOL.json','GEOMETRY_STARTED.json','GEOMETRY_RECEIPT.json')]
    own,started,check=[read(v/n) for n in ('VALIDATION_PROTOCOL.json','VALIDATION_STARTED.json','VALIDATION_CHECKS.json')]
    require(core['schema']=='whole_U_closure_geometry_protocol_v21' and core['methods']==[PRIMARY]
        and core['frames']==core['rows']==core['ledger_rows']==245 and core['limits']['GT_allowed'] is False,'fixed point geometry')
    require(r['schema']=='whole_U_closure_geometry_receipt_v21' and r['complete'] is True and r['error'] is None
        and not r['cleanup_errors'] and r['inputs_unchanged'] is r['environment_restored'] is r['counter_dispatch_closed'] is True
        and all(value is True for value in r['restoration'].values())
        and r['inputs']==core['inputs']==start['inputs']
        and r['actual_execution']==dict(frames_started=245,geometry_rows=245,ledger_rows=245,frames_complete=245),'completed restored245')
    for key,name in (('geometry','GEOMETRY.jsonl.gz'),('ledgers','LEDGERS.jsonl.gz'),('protocol','GEOMETRY_PROTOCOL.json')):
        require(same(r[key],binding(g/name)),'geometry authority '+key)
    require(own['schema']=='whole_U_closure_point_scalar_protocol_v21' and check['schema']=='whole_U_closure_point_scalar_checks_v21'
        and check['complete'] is check['passed'] is True and check['failure_count']==0 and check['exception'] is None
        and check['new_production_calls']==started['new_production_calls']==0
        and check['old_full_V16_validation_passed'] is False and check['limits']['old_full_V16_564_FAIL_remains_FAIL'] is True
        and all(check['actual_counts'][k]==245 for k in ('frames','geometry_rows','ledger_rows')),'own actual full point scalar PASS')
    require(check['inputs']==own['inputs']==started['inputs'] and same(check['own_protocol'],binding(v/'VALIDATION_PROTOCOL.json'))
        and same(started['own_protocol'],check['own_protocol']) and same(check['checker'],binding(a.validator)),'actual independent envelope')
    for key,name in (('geometry','GEOMETRY.jsonl.gz'),('ledgers','LEDGERS.jsonl.gz'),('geometry_protocol','GEOMETRY_PROTOCOL.json'),
        ('geometry_started','GEOMETRY_STARTED.json'),('geometry_receipt','GEOMETRY_RECEIPT.json')):
        require(same(check['inputs'][key],binding(g/name)),'independent current stream '+key)
    return r,check

def prerequisites(a,C,V7,helpers):
    gate(a);eligible,normalization=helpers.normalized_frames(a);panel=read(C.CORRECTED/'RUNTIME_PANEL.json')['frames'];ids={r['frame_id'] for r in panel}
    require(len(panel)==len(ids)==26 and len({r['session_id'] for r in panel})==13
        and Counter(r['session_id'] for r in panel)=={s:2 for s in {r['session_id'] for r in panel}},'fixed26 panel/13sessions')
    references={};seen=set();fields=('native_points','actual_pose','hidden_initial','output_status','new_pose_estimated','fallback_used','fixed_metadata')
    for row in C.rows(Path(a.geometry)/'GEOMETRY.jsonl.gz'):
        require(row['method']==PRIMARY and row['id'] in eligible and row['id'] not in seen,'full unique primary geometry');seen.add(row['id'])
        if row['id'] in ids:references[row['id']]={k:copy.deepcopy(row[k]) for k in fields}
    require(len(seen)==245 and set(references)==ids and all(eligible[p['frame_id']]['session']==p['session_id'] for p in panel),'fixed full245/26 parity references')
    return panel,eligible,references,normalization,PRIMARY

def inputs(a,V7):
    paths=dict(code=Path(__file__),validator=Path(a.validator),**source_paths(a))
    for name in ('solver.py','pipeline.py','run_cpu.py','test_solver.py','run_geometry.py'):paths['private_source:'+name]=HERE/name
    for name in ('GEOMETRY_PROTOCOL.json','GEOMETRY_STARTED.json','GEOMETRY_RECEIPT.json','GEOMETRY.jsonl.gz','LEDGERS.jsonl.gz'):paths['geometry:'+name]=Path(a.geometry)/name
    for name in ('VALIDATION_PROTOCOL.json','VALIDATION_STARTED.json','VALIDATION_CHECKS.json'):paths['validation:'+name]=Path(a.validation)/name
    own=read(Path(a.validation)/'VALIDATION_PROTOCOL.json')
    for key,expected in read(Path(a.geometry)/'GEOMETRY_PROTOCOL.json')['inputs'].items():
        path=Path(own['inputs']['geometry_input:'+key]['path']);require(same(binding(path),expected),'actual frozen closure '+key);paths['tested:'+key]=path
    result={key:binding(path) for key,path in paths.items()};result.update({'fresh_v7:'+key:value for key,value in V7.inputs(a).items()})
    old=read(a.old_runtime_receipt)
    require(old['complete'] is False and old['status']=='FAILED_RUNTIME_CLEANUP' and old['statistics_official'] is False
        and old['serialized_rows']==old['execution']['calls_complete']==150,'historical V19 runtime failure preserved')
    result['historical_failed_V19_runtime_receipt']=binding(a.old_runtime_receipt)
    return result

@contextmanager
def no_geometry_reads(a):
    old=[(builtins,'open',builtins.open),(io,'open',io.open)];roots=(Path(a.geometry).resolve(),Path(a.validation).resolve())
    def wrap(fn):
        def call(path,*args,**kw):
            if isinstance(path,(str,Path)):require(not any(Path(path).resolve().is_relative_to(root) for root in roots),'saved geometry entered inference')
            return fn(path,*args,**kw)
        return call
    try:
        for owner,key,fn in old:setattr(owner,key,wrap(fn))
        yield
    finally:
        for owner,key,fn in old:setattr(owner,key,fn)

def namespace_snapshot(a):
    import scripts.research as research
    environment={k:os.environ.get(k) for k in ('PALLET_SOURCE_ROOT','PALLET_BASELINE_ROOT')}
    os.environ.update(PALLET_SOURCE_ROOT=str(a.source_root),PALLET_BASELINE_ROOT=str(a.baseline_root))
    try:
        old=importlib.import_module('scripts.research.pallet_observation_refiner_20261009_v1.common')
        learned=importlib.import_module('scripts.research.pallet_observation_refiner_20261009_v1.learned_infer')
    finally:
        for key,value in environment.items():
            if value is None:os.environ.pop(key,None)
            else:os.environ[key]=value
    return dict(old=old,learned=learned,attributes={key:getattr(old,key) for key in ('DOC','SCRATCH','read','iter_rows','binding')},
        fits=learned.FITS,research=research,namespace=list(research.__path__),environment=environment)

def namespace_restored(saved):
    values={key:getattr(saved['old'],key)==value if key in ('DOC','SCRATCH') else getattr(saved['old'],key) is value
        for key,value in saved['attributes'].items()}
    values.update(learned_FITS=saved['learned'].FITS==saved['fits'],research_path=list(saved['research'].__path__)==saved['namespace'],
        environment=saved['environment']=={key:os.environ.get(key) for key in saved['environment']})
    return values

def main():
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=('freeze','preflight','measure'))
    for key in ('repo','private','geometry','validation','validator','source-root','baseline-root','fits','cohort','calibration','calibration-root','prior-bindings'):
        parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--old-runtime-receipt',type=Path,default=Path('/tmp/pallet-finite-support-orthogonal-private-20261011-v19/runtime/RUNTIME_RECEIPT.json'));parser.add_argument('--base-weights');parser.add_argument('--N3-weights');parser.add_argument('--remaining-seconds',type=float,default=3600.)
    a=parser.parse_args()
    for key in ('repo','private','geometry','validation','validator','source_root','baseline_root','fits','cohort','calibration','calibration_root','prior_bindings'):setattr(a,key,getattr(a,key).resolve())
    require(a.private==HERE,'fixed private21 runtime root');a.output=str(a.private/'runtime');out=Path(a.output);proto=out/'RUNTIME_PROTOCOL.json'
    require(not any(p.is_symlink() for p in (out,*out.parents)) and not any(out.is_relative_to(p) or p.is_relative_to(out) for p in (a.repo,a.source_root,a.baseline_root,a.fits,a.geometry,a.validation)),'new disjoint runtime output')
    C,V7,H,old,book,witness=setup(a);panel,eligible,refs,normalization,primary=prerequisites(a,C,V7,H);bound=inputs(a,V7)
    if a.phase=='freeze':
        require(not out.exists(),'fresh absent runtime folder');out.mkdir()
        write(proto,dict(schema='fresh_whole_U_point_closure_runtime_protocol_v21',inputs=bound,method=primary,stages=list(STAGES),schedule=book['schedule'](),
            warmup=20,measured=130,configured_calls=150,panel=panel,normalization=normalization,bookkeeping_AST=witness,protection=protect(a),
            geometry_used_only_after_return=True,one_point_only_closure_path_timed=True,historical_V19_runtime_cleanup_failed_retained=True,automatic_retry=False,new_RGB=0,new_training=0,
            first_predict_detector_forwards=2,other_predict_detector_forwards=1,N3_forwards_per_prediction=1,no_artificial_pre_forward=True,
            included='fresh detector/N3/SubPix/initial and Base-feature poses/ROLE+CAL/H-union-k/old point control/one point closure/all-U Generic+LM/H projection/metadata',
            excluded=['model loading','file decode','saved parity','resource and protection probes','logging','GT scoring']))
        print('V21_RUNTIME_FROZEN',flush=True);return
    protocol=read(proto);require(protocol['inputs']==bound and protocol['schedule']==book['schedule']() and protocol['method']==primary and protocol['bookkeeping_AST']==witness and protocol['protection']==protect(a),'frozen runtime inputs/protection')
    if a.phase=='preflight':
        resource=C.quiet_snapshot();write(out/'RUNTIME_PREFLIGHT.json',dict(protocol=binding(proto),passed=bool(resource['quiet'] and resource['temperature_under_80']),resource=resource,actual_model_fit_decode_calls=0));require(resource['quiet'] and resource['temperature_under_80'],'quiet preflight');return
    pre=read(out/'RUNTIME_PREFLIGHT.json');require(pre['passed'] is True and same(pre['protocol'],binding(proto)),'own preflight PASS')
    for name in ('RUNTIME_STARTED.json','RUNTIME_RECEIPT.json','RUNTIME_ROWS.jsonl.gz','PENDING_RUNTIME_ROWS.jsonl.gz','INTERRUPTED_RUNTIME_ROWS.jsonl.gz','INTERRUPTED_RUNTIME_PACKET.json','RUNTIME_RESOURCES.json'):require(not(out/name).exists(),'preserve '+name)
    import cv2
    import numpy as np
    from scripts.research.pallet_three_head_observation_20261010_v7.run import RowWriter
    result=dict(schema='fresh_whole_U_point_closure_runtime_receipt_v21',complete=False,status='FAILED_RUNTIME',protocol=binding(proto),method=primary,configured_calls=150,normalization=normalization)
    write(out/'RUNTIME_STARTED.json',dict(protocol=binding(proto),inputs=bound,status='STARTED'));before=protect(a);state=Counter();cv=Counter();opt=Counter();svd=Counter();head=Counter();resources=[];cleanup={};errors=[];rows=[];pipeline=writer=active=None;written=False;hooks=[];began=time.monotonic();environment={k:os.environ.get(k) for k in ('PALLET_SOURCE_ROOT','PALLET_BASELINE_ROOT')};saved_namespace=None;cv_initial_threads=cv2.getNumThreads();torch_options=None
    private_prefix='private_v21_runtime_point_pipeline';saved_private_modules={k:v for k,v in sys.modules.items() if k==private_prefix or k.startswith(private_prefix+'.')}
    def probe(phase):
        r=C.quiet_snapshot();r['phase']=phase;resources.append(r);require(r['quiet'] and r['temperature_under_80'],'quiet/thermal resource guard')
    try:
        saved_namespace=namespace_snapshot(a)
        probe('before_decode');images=[]
        for p in panel:
            f=eligible[p['frame_id']];path=a.source_root/f['image'];require(V7.sha(path)==f['image_sha256'],'original RGB SHA');image=cv2.imread(str(path));require(image is not None and list(image.shape[:2])==f['raw_hw'],'original RGB shape');images.append(image);state['image_decode_calls']+=1
        writer=RowWriter(out/'RUNTIME_ROWS.jsonl.gz',interrupted_path=out/'INTERRUPTED_RUNTIME_ROWS.jsonl.gz')
        with C.legacy_context(a):
            try:
                cls,label=pipeline_class(a);require(label==primary,'single new primary')
                import torch
                torch_options=dict(threads=torch.get_num_threads(),matmul_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_tf32=torch.backends.cudnn.allow_tf32,cudnn_benchmark=torch.backends.cudnn.benchmark,cudnn_deterministic=torch.backends.cudnn.deterministic)
                with V7.inference_canary():pipeline=cls(a)
                torch=pipeline.torch;require(torch.cuda.is_available(),'fixed GPU path');state['model_initializations']=1;loaded=Counter(detector=pipeline.models.detector_forwards,N3=pipeline.models.n3_forwards);require(loaded==dict(detector=0,N3=0),'no loading forwards')
                module=pipeline.heads['IMAGE_ROLE'];hooks=[module.register_forward_pre_hook(lambda *v:head.update(entries=1)),module.register_forward_hook(lambda *v:head.update(returns=1))];probe('after_models');torch.cuda.reset_peak_memory_stats();last_probe=time.monotonic()
                with book['instruments'](cv,opt,svd,cleanup),V7.inference_canary(),no_geometry_reads(a),pipeline.old.no_truth_reads(),torch.no_grad():
                    for job in book['schedule']():
                        require(time.monotonic()-began<a.remaining_seconds,'time budget, preserve prefix');active=None;written=False
                        if time.monotonic()-last_probe>=5 or job['phase']=='measured' and job['image_index']==0:probe('during_paths');last_probe=time.monotonic()
                        frame=eligible[panel[job['image_index']]['frame_id']];state['calls_started']+=1;old.durable_owned(out/'RUNTIME_STARTED.json',dict(protocol=binding(proto),inputs=bound,status='RUNNING',active_job=job,execution=dict(state)))
                        groups=dict(cv=cv,optimizer=opt,svd=svd,head=head,base=pipeline.base.counts,fresh=pipeline.counts,local=pipeline.local_operation_counts,direct=pipeline.direct_point_operation_counts,whole_U=pipeline.whole_U_operation_counts,bank_capture=pipeline.point_bank_capture_counts);counters={k:v.copy() for k,v in groups.items()};model_before=Counter(detector=pipeline.models.detector_forwards,N3=pipeline.models.n3_forwards);ticks=[]
                        def mark(stage):torch.cuda.synchronize();ticks.append((stage,time.perf_counter_ns()))
                        torch.cuda.synchronize();t0=time.perf_counter_ns();output=pipeline.predict(images[job['image_index']],frame['K'],frame['xyz'],{k:frame[k] for k in ('id','session','object_type')},method=primary,stage_callback=mark);torch.cuda.synchronize();t1=time.perf_counter_ns()
                        times={};previous=t0
                        for stage,tick in ticks:times[stage+'_ms']=(tick-previous)/1e6;previous=tick
                        times['return_ms']=(t1-previous)/1e6;d={k:book['delta'](v,counters[k]) for k,v in groups.items()};d['model']=book['delta'](Counter(detector=pipeline.models.detector_forwards,N3=pipeline.models.n3_forwards),model_before);require([s for s,_ in ticks]==list(STAGES) and d['head']==dict(entries=1,returns=1),'stage/head count');require(d['model']==dict(detector=2 if state['calls_started']==1 else 1,N3=1),'first2/rest1 detector, N3=1')
                        active=dict(**job,id=frame['id'],session=frame['session'],method=primary,t0_ns=t0,t1_ns=t1,stage_ticks_ns=ticks,full_ms=(t1-t0)/1e6,**times,full_fresh_output=output,original_fresh_detector_prediction=copy.deepcopy(pipeline.last_capture['raw']),fresh_point_bank_ledger=copy.deepcopy(pipeline.last_point_bank_ledger),fresh_point_bank_reuse=copy.deepcopy(pipeline.last_point_bank_reuse),fresh_repair_ledger=copy.deepcopy(pipeline.last_repair_ledger),execution_events=copy.deepcopy(pipeline.last_execution_events),counter_deltas=d,parity_status='PENDING')
                        reference=refs[frame['id']];active['points_parity']=pipeline.runtime._point_parity(output['native_points'],reference['native_points'],1e-7,(primary,frame['id']));active['pose_parity']=H.pose_parity(output['actual_pose'],reference['actual_pose'])
                        require(all(C.finite(output[k])==C.finite(reference[k]) for k in ('hidden_initial','output_status','new_pose_estimated','fallback_used','fixed_metadata')),'outcome parity')
                        for arm,key in (('BASE','original_base_points'),('N3_SUBPIX','native_N3_points')):active[arm+'_parity']=pipeline.runtime._point_parity(output[key],frame['points'][arm],1e-7,(arm,frame['id']))
                        ledger=pipeline.last_repair_ledger;reuse=pipeline.last_point_bank_reuse;operations=ledger['local_operation_counts']
                        require(ledger['actual_solver_calls']==ledger['reported_output_views']==1 and ledger['new_endpoint_validation_calls']==ledger['new_line_solve_calls']==0,'one primary point-only closure')
                        require(d['local']==operations and d['direct']==reuse['raw_four_ID_bank_delta'] and d['whole_U']==reuse['whole_U_closure_bank_delta']
                            and reuse['constructor_restored'] and reuse['same_coordinate_bank_reused'] and reuse['additional_raw_four_ID_Generic_calls']==0,'raw4/cache and separate whole-U ledger')
                        require(d['optimizer'].get('entries',0)==operations['refit_optimizer_calls']==operations['local_optimizer_calls']
                            and d['optimizer'].get('returns',0)==operations['refit_optimizer_returns']==operations['local_optimizer_completions']
                            and d['optimizer'].get('exceptions',0)==operations['refit_optimizer_exceptions']
                            and d['optimizer'].get('residual_entries',0)==operations['optimizer_residual_calls']
                            and d['optimizer'].get('jacobian_entries',0)==operations['optimizer_jacobian_calls'],'actual SciPy and callback ledger')
                        if output['new_pose_estimated']:require(set(output['hidden_initial']).isdisjoint(output['solver']['fit_input_ids']),'H fit leak');np.testing.assert_allclose(np.asarray(output['native_points'])[output['hidden_initial']],np.asarray(output['solver']['projected'])[output['hidden_initial']],rtol=0,atol=1e-12)
                        active.update(parity_status='PASS',saved_geometry_used_only_after_return=True);written=True;writer.write(active);rows.append({k:active[k] for k in ('phase','full_ms',*(s+'_ms' for s in STAGES),'return_ms')});active=None;state['calls_complete']+=1
                        if state['calls_complete']%25==0:print('V21_FRESH_RUNTIME',state['calls_complete'],150,flush=True)
                probe('after_fresh_paths');require(state['calls_complete']==150 and head==dict(entries=150,returns=150),'actual150 complete')
                for k in ('detector_calls','N3_route_calls','initial_pose_calls','feature_initial_pose_calls','IMAGE_ROLE_head_calls','head_calls'):require(pipeline.base.counts[k]==150,'base path '+k)
                for k in ('fresh_predict_calls','fresh_hybrid_point_paths','fresh_repair_paths','fresh_point_closure_paths'):require(pipeline.counts[k+'_started']==pipeline.counts[k+'_complete']==150,'fresh path '+k)
                require(pipeline.models.detector_forwards==151 and pipeline.models.n3_forwards==150 and pipeline.counts['reported_primary_views_from_single_point_closure']==150,'actual model/output totals')
                require(opt['entries']==opt['returns']+opt['exceptions'] and svd['entries']==svd['returns']+svd['exceptions'] and all(cv[n+':entries']==cv[n+':returns']+cv[n+':exceptions'] for n in PRIMITIVES),'actual entry/return/exception totals')
                require(all(opt[k+'_entries']==opt[k+'_returns']+opt[k+'_exceptions'] for k in ('residual','jacobian')),'actual callback accounting')
                result.update(complete=True,status='COMPLETE',peak_GPU_bytes=torch.cuda.max_memory_allocated(),model_forwards_after_loading=dict(loaded))
            finally:
                if pipeline is not None:
                    try:pipeline.close();cleanup['pipeline_closed']=pipeline.closed;cleanup['pipeline_closed_before_outer_legacy_restore']=True
                    except BaseException as exc:errors.append(dict(phase='close',type=type(exc).__name__,message=str(exc)))
    except BaseException as exc:result.update(complete=False,status='FAILED_RUNTIME',error=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()))
    finally:
        for hook in hooks:
            try:hook.remove()
            except BaseException as exc:errors.append(dict(phase='hook_remove',type=type(exc).__name__,message=str(exc)))
        cleanup['head_hooks_removed']=not any(e['phase']=='hook_remove' for e in errors)
        if pipeline is not None:
            result.update(model_bindings=pipeline.bindings,model_forwards=dict(detector=pipeline.models.detector_forwards,N3=pipeline.models.n3_forwards),actual_base_counts=dict(pipeline.base.counts),actual_fresh_counts=dict(pipeline.counts),local_operation_counts=dict(pipeline.local_operation_counts),whole_U_operation_counts=dict(pipeline.whole_U_operation_counts),direct_point_operation_counts=dict(pipeline.direct_point_operation_counts),point_bank_capture_counts=dict(pipeline.point_bank_capture_counts),packet_compatibility_counts=dict(pipeline.packet_compatibility_counts),deployment_source_isolation=pipeline.source_isolation_proof)
            if not result['complete']:
                try:write(out/'INTERRUPTED_RUNTIME_PACKET.json',C.finite(dict(capture=pipeline.last_capture,point=pipeline.last_point_result,point_ledger=pipeline.last_point_bank_ledger,reuse=pipeline.last_point_bank_reuse,repair=pipeline.last_repair_ledger,active_row=active)))
                except BaseException as exc:errors.append(dict(phase='interrupted_packet',type=type(exc).__name__,message=str(exc)))
        for name in tuple(sys.modules):
            if name==private_prefix or name.startswith(private_prefix+'.'):sys.modules.pop(name,None)
        sys.modules.update(saved_private_modules)
        cleanup['private_pipeline_registration_restored']=({k:v for k,v in sys.modules.items() if k==private_prefix or k.startswith(private_prefix+'.')}==saved_private_modules)
        try:
            cv2.setNumThreads(cv_initial_threads);cleanup['original_opencv_threads_restored']=cv2.getNumThreads()==cv_initial_threads
            if torch_options is not None:
                torch.set_num_threads(torch_options['threads']);torch.backends.cuda.matmul.allow_tf32=torch_options['matmul_tf32']
                torch.backends.cudnn.allow_tf32=torch_options['cudnn_tf32'];torch.backends.cudnn.benchmark=torch_options['cudnn_benchmark'];torch.backends.cudnn.deterministic=torch_options['cudnn_deterministic']
                cleanup['original_torch_options_restored']=(torch.get_num_threads()==torch_options['threads'] and torch.backends.cuda.matmul.allow_tf32==torch_options['matmul_tf32'] and torch.backends.cudnn.allow_tf32==torch_options['cudnn_tf32'] and torch.backends.cudnn.benchmark==torch_options['cudnn_benchmark'] and torch.backends.cudnn.deterministic==torch_options['cudnn_deterministic'])
            cleanup['legacy_namespace_restoration']=namespace_restored(saved_namespace) if saved_namespace is not None else dict(snapshot_completed=False)
            require(all(cleanup['legacy_namespace_restoration'].values()),'complete nested context namespace restoration')
            require(cleanup['original_opencv_threads_restored'] and (torch_options is None or cleanup['original_torch_options_restored']), 'original numerical thread/backend options restoration')
            require(cleanup['private_pipeline_registration_restored'] and (pipeline is None or pipeline.closed and cleanup.get('pipeline_closed_before_outer_legacy_restore')), 'pipeline closed in LIFO order and private registrations restored')
            cleanup['environment_restored']=environment=={k:os.environ.get(k) for k in environment};require(cleanup['environment_restored'],'environment restoration');require(inputs(a,V7)==bound and protect(a)==before,'protected source/input bytes');require(not cleanup.get('instrument_restoration_errors') and (not cleanup.get('instruments_entered') or cleanup.get('instruments_restored') and cleanup.get('opencv_threads_restored')) and (pipeline is None or cleanup.get('pipeline_closed') and cleanup['head_hooks_removed']),'instrument/pipeline restoration');probe('after_cleanup')
        except BaseException as exc:errors.append(dict(phase='restoration',type=type(exc).__name__,message=str(exc)))
        if writer is not None:
            try:
                if active is not None and not written:writer.write(active)
                writer.close();require(not writer.close_errors,'stream close');require(not result['complete'] or writer.count==150,'serialized150')
                if result['complete'] and not errors:writer.promote()
                else:writer.preserve_interrupted()
            except BaseException as exc:errors.append(dict(phase='stream',type=type(exc).__name__,message=str(exc)));writer.preserve_interrupted()
            errors.extend(writer.close_errors+writer.preservation_errors)
        if errors:result.update(complete=False,status='FAILED_RUNTIME_CLEANUP')
        result.update(cleanup=cleanup,cleanup_errors=errors,execution=dict(state),live_cv_counts=dict(cv),live_optimizer_counts=dict(opt),live_SVD_counts=dict(svd),live_head_counts=dict(head),resource_snapshots=resources,all_recorded_resource_snapshots_quiet=bool(resources and all(r['quiet'] and r['temperature_under_80'] for r in resources)),serialized_rows=0 if writer is None else writer.count,statistics_official=result['complete'],protected_before=before,protected_after=protect(a))
        raw=out/('RUNTIME_ROWS.jsonl.gz' if result['complete'] else 'INTERRUPTED_RUNTIME_ROWS.jsonl.gz');result['raw_rows']=binding(raw) if raw.is_file() else None
        result['summaries']={s:old.describe([r[s+'_ms'] for r in rows if r['phase']=='measured']) for s in ('full',*STAGES,'return')} if result['complete'] else None
        write(out/'RUNTIME_RESOURCES.json',dict(protocol=binding(proto),snapshots=resources,observed_snapshots_only=True));result['resources']=binding(out/'RUNTIME_RESOURCES.json');write(out/'RUNTIME_RECEIPT.json',C.finite(result));old.durable_owned(out/'RUNTIME_STARTED.json',dict(protocol=binding(proto),inputs=bound,status=result['status'],execution=dict(state),receipt=binding(out/'RUNTIME_RECEIPT.json')))
    require(result['complete'],'fresh runtime incomplete, first prefix/failure preserved')

if __name__=='__main__':main()
