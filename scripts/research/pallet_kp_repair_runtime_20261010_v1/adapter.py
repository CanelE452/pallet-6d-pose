"""Prepare or execute the original 600-path benchmark for corrected IMAGE_ROLE.

Readiness/schedule stages use the standard library only. ``measure`` checks
authorized completion and sealed geometry before importing Torch or the old
benchmark. The old benchmark/decoder/solver remain unmodified.
"""
import argparse
import ast
from collections import Counter
import contextlib
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import sys

sys.dont_write_bytecode=True
REPO=Path(__file__).resolve().parents[3]
DOC=REPO/'_docs/experiments/pallet_kp_repair_runtime_20261010_v1'
ORIGINAL=REPO/'_docs/experiments/pallet_observation_refiner_20261009_v1'
REPAIR=REPO/'_docs/experiments/pallet_kp_supervision_repair_20261010_v1'
OLD_CODE=REPO/'scripts/research/pallet_observation_refiner_20261009_v1'
ARMS=('BASE','N3_SUBPIX','N3_SUBPIX_GEOM_NOSELF_ROBUST','IMAGE_ROLE')
MODEL_ARMS=('GEOMETRY_ONLY','IMAGE_NO_ROLE','IMAGE_ROLE')
METHODS=MODEL_ARMS+('IMAGE_ROLE_NO_MASK_ROBUST','IMAGE_ROLE_STANDARD','IMAGE_ROLE_POINT_LINE')
REDIRECT_NAMES={'SOLVER_CHECKS.json','CONTRACT_AUDIT.json','POSE_DIAGNOSTICS.jsonl.gz'}
PRIVATE_ROOTS=tuple(Path('/dev/shm')/x for x in (
    'pallet-observation-private-20261009','pallet-kp-difficulty-private-20261010',
    'pallet-kp-supervision-gate-private-20261010','pallet-kp-supervision-repair-private-20261010'))


def require(condition,message):
    if not condition:raise RuntimeError('RUNTIME_PENDING: '+message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def binding(path):
    path=Path(path).resolve()
    name,origin=(str(path.relative_to(REPO)),'public_repository') if path.is_relative_to(REPO) else (path.name,'external_readonly_dependency')
    return dict(path=name,origin=origin,sha256=sha(path),bytes=path.stat().st_size)


def read(path):return json.loads(Path(path).read_text())


def write_new(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:f.write(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def rows(path):
    with gzip.open(path,'rt') as f:
        for line in f:
            if line.strip():yield json.loads(line)


def bound_file(path,expected,label):
    require(Path(path).is_file(),label+' file missing')
    require(sha(path)==expected['sha256'] and Path(path).stat().st_size==expected['bytes'],label+' SHA/bytes mismatch')


def prior_snapshot(path):
    data=read(path);require(data['protected_prior_files']==len(data['files'])==311,'prior311 protection population changed')
    result=[]
    for b in data['files']:
        p=(REPO/b['path']).resolve();require(p.is_relative_to(REPO),'prior binding path escapes repository')
        bound_file(p,b,'protected prior '+b['path']);result.append(b)
    return result


def original_schedule():
    """Execute only the exact original pure-stdlib schedule AST, never import it."""
    source=(OLD_CODE/'benchmark.py').read_text();tree=ast.parse(source)
    function=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='schedules')
    original_constants={}
    for node in tree.body:
        if not isinstance(node,ast.Assign):continue
        if len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id=='ARMS':
            original_constants['ARMS']=ast.literal_eval(node.value)
        if len(node.targets)==1 and isinstance(node.targets[0],ast.Tuple):
            names=[n.id for n in node.targets[0].elts]
            if names==['WARMUP','REPEATS','FRAMES']:original_constants.update(zip(names,ast.literal_eval(node.value)))
    require(original_constants==dict(ARMS=ARMS[:3],WARMUP=20,REPEATS=5,FRAMES=26),'original schedule constants changed')
    namespace=dict(original_constants,Counter=Counter)
    isolated=ast.Module(body=[function],type_ignores=[])
    exec(compile(isolated,str(OLD_CODE/'benchmark.py')+':schedules_only','exec'),namespace)
    warm,measured=namespace['schedules'](ARMS)
    require(len(warm)==80 and len(measured)==520,'original600 schedule size differs')
    require(Counter(x['arm'] for x in warm)=={a:20 for a in ARMS},'warmup20 per arm differs')
    require(Counter(x['arm'] for x in measured)=={a:130 for a in ARMS},'measured130 per arm differs')
    require(len({(x['arm'],x['repeat'],x['image_index']) for x in measured})==520,'measured schedule duplicated')
    return warm,measured,dict(source=binding(OLD_CODE/'benchmark.py'),
                             isolated_function_AST_sha256=hashlib.sha256(ast.dump(function,include_attributes=False).encode()).hexdigest(),
                             model_imports=0,old_module_imported=False)


def fixed_paths():
    result={name:OLD_CODE/name for name in ['benchmark.py','learned_infer.py','model.py','inference.py','solver.py','common.py']}
    result.update({name:ORIGINAL/name for name in sorted(REDIRECT_NAMES)})
    result.update(repair_protocol=REPAIR/'RETRAINING_PROTOCOL.json',repair_checks=REPAIR/'ZERO_UPDATE_CPU_CHECKS.json',
                  prior_bindings=DOC/'PRIOR_PUBLICATION_BINDINGS.json',adapter=Path(__file__).resolve(),
                  readiness_smoke=Path(__file__).with_name('readiness_smoke.py'))
    return result


def freeze(args):
    protected=prior_snapshot(args.prior_bindings)
    warm,measured,proof=original_schedule()
    value=dict(schema='corrected_image_role_original_runtime_protocol_v1',status='PREPARED_NOT_MEASURED',
               training_authorization='stillPENDING; actual corrected last-checkpoint completion must exist before measurement',
               implementation='unchanged original observation_refiner benchmark.run + benchmark_adapter; only corrected L.FITS and new C.DOC/C.SCRATCH paths',
               arms=list(ARMS),frames=26,sessions=13,warmup_per_arm=20,repeats=5,measured_per_arm=130,total_pipeline_calls=600,
               fixed_inputs={key:binding(path) for key,path in fixed_paths().items()},schedule_source=proof,
               sealed_reference='corrected LEARNED_GEOMETRY_SEALED.jsonl.gz only; GT-scored LEARNED_PREDICTIONS is forbidden as the learned parity input',
               startup='before benchmark/model import: completed authorized9000 replay, exact protocol/checks,all3last-checkpointSHA,957observationseal,1914geometryseal,319IMAGE_ROLEobservation+geometry+originalinput identities; then CPU-check all3checkpoint metadata before detector creation',
               timing='original fresh600 complete routes: native RAM RGB, shared detector neck, original66wayMAP correction/feature-role pose, original historical initial pose, fresh finite4subset robustPnP, hidden replacement/display/metadata copy',
               cached_coordinate_timing=False,partial_stage_sum_timing=False,parameters_changed=False,
               outside_intervals=['model/checkpoint load','RGB file/decode','reference/GT scoring','coordinate/pose parity checks','journals','resource snapshots'],
               parity='original per-call native point/pose/status/hidden tolerance1e-7, metadata/center preservation, GT canary; official runtime only if all600 finish and resource checks remain quiet',
               output_protection='new exact runtimeDOC or isolated external directory; new scratch; preserve complete or interrupted runtime files/journal/claim; old3diagnostic paths redirected for reads/bindings only and globals restored',
               original_protected_files=len(protected),executed_pipeline_calls=0,head_forwards=0,detector_forwards=0,
               additional_training_updates=0,measured_time_results_available=False)
    write_new(args.protocol,value)
    write_new(DOC/'RUNTIME_SCHEDULE.json',dict(schema='exact_original600_runtime_schedule_v1',proof=proof,
                                             arms=list(ARMS),warmup=warm,measured=measured,total=600))
    print('RUNTIME_ADAPTER_FROZEN',sha(args.protocol),flush=True)


def check_frozen(args):
    p=read(args.protocol)
    require(p['arms']==list(ARMS) and p['total_pipeline_calls']==600,'runtime protocol changed')
    require({key:binding(path) for key,path in fixed_paths().items()}==p['fixed_inputs'],'runtime frozen input/code changed')
    prior_snapshot(args.prior_bindings)
    return p


def protected_location(path,source_root=None,baseline_root=None,fits=None,allow_doc=False):
    path=Path(path).resolve()
    if allow_doc and path==DOC.resolve():return path
    roots=[REPO.resolve(),*(p.resolve() for p in PRIVATE_ROOTS)]
    roots.extend(Path(p).resolve() for p in [source_root,baseline_root,fits] if p)
    require(all(not path.is_relative_to(root) and not root.is_relative_to(path) for root in roots),'output overlaps a protected experiment/source directory')
    return path


def output_guards(args):
    output=protected_location(args.output,args.source_root,args.baseline_root,args.fits,allow_doc=True)
    scratch=protected_location(args.scratch,args.source_root,args.baseline_root,args.fits)
    require(output!=scratch and not output.is_relative_to(scratch) and not scratch.is_relative_to(output),'runtime output/scratch must be disjoint')
    for name in ['RUNTIME.json','RUNTIME_ROWS.jsonl.gz','RUNTIME_ADAPTER_STARTED.json','RUNTIME_ADAPTER_RECEIPT.json']:
        require(not (output/name).exists(),'preserve completed/interrupted '+name)
    require(not (scratch/'RUNTIME_STARTED.json').exists(),'preserve interrupted runtime journal')
    require(not scratch.exists() or (scratch.is_dir() and not any(scratch.iterdir())),'runtime scratch must be new or empty')
    require(output==DOC.resolve() or not output.exists() or (output.is_dir() and not any(output.iterdir())),'external runtime output must be new or empty')
    return output,scratch


def prereqs(args):
    """Pure stored-data guards. No Torch/CV2/model imports or pose calculation."""
    require(Path(args.completion).is_file(),'corrected TRAINING_COMPLETION missing; no models imported')
    completion=read(args.completion);protocol=read(REPAIR/'RETRAINING_PROTOCOL.json')
    require(completion.get('complete') is True and completion.get('formal_updates')==9000 and completion.get('total_updates')==9000,'corrected authorized9000 completion invalid')
    require(completion.get('throwaway_updates')==0 and completion.get('batch')==16 and completion.get('seed')==1,'corrected training settings changed')
    require(completion.get('protocol',{}).get('sha256')==sha(REPAIR/'RETRAINING_PROTOCOL.json'),'corrected training protocol binding differs')
    require(Path(args.authorization).is_file(),'additional9000 approved authorization missing')
    approved=read(args.authorization)
    require(approved.get('status')=='APPROVED' and approved.get('additional_formal_updates')==9000 and bool(approved.get('user_authorization_evidence')),'additional9000 authorization is not approved')
    require(approved.get('protocol_sha256')==sha(REPAIR/'RETRAINING_PROTOCOL.json') and approved.get('checks_sha256')==sha(REPAIR/'ZERO_UPDATE_CPU_CHECKS.json'),'approved protocol/checks binding differs')
    bound_file(args.authorization,completion['authorization'],'completed training authorization')
    require(completion.get('input_hashes_unchanged') and completion.get('same_initial_tensor_sha') and completion.get('same_batch_order') and completion.get('same_update_budget') and completion.get('source_scores_model_selection') is False,'corrected training parity fields invalid')
    checkpoints=completion['checkpoints']
    require(len(checkpoints)==3 and [x['arm'] for x in checkpoints]==list(MODEL_ARMS),'last-checkpoint arm population differs')
    checkpoint_bindings={}
    for row in checkpoints:
        require(row.get('updates')==3000 and row.get('exposures')==48000,'checkpoint update/exposure count differs')
        path=Path(args.fits)/(row['arm']+'.pt');bound_file(path,row['checkpoint'],row['arm']+' last checkpoint')
        checkpoint_bindings[row['arm']]=binding(path)
    require(Path(args.observation_seal).is_file() and Path(args.pose_execution).is_file(),'corrected observation/geometry completion missing')
    seal=read(args.observation_seal);execution=read(args.pose_execution)
    require(seal.get('complete') and seal.get('frames')==319 and seal.get('rows')==957 and seal.get('models')==list(MODEL_ARMS) and seal.get('GT_open_allowed') is False,'corrected noGT957 observation seal invalid')
    bound_file(args.observations,seal['records'],'corrected learned observations')
    require(len(seal['checkpoints'])==3,'observation checkpoint population differs')
    for arm,b in zip(MODEL_ARMS,seal['checkpoints']):bound_file(Path(args.fits)/(arm+'.pt'),b,'observation '+arm+' checkpoint')
    require(execution.get('complete') and execution.get('rows')==1914 and execution.get('models')==3 and execution.get('GT_canary') is True,'corrected1914 geometry completion invalid')
    require(execution.get('point_solver_paths')==1595 and execution.get('point_line_paths')==319,'corrected original6path geometry counts differ')
    require(Path(args.geometry).name=='LEARNED_GEOMETRY_SEALED.jsonl.gz','learned runtime reference must be the GT-free geometry seal')
    bound_file(args.geometry,execution['final_sealed'],'corrected geometry seal')
    bound_file(args.observations,execution['raw_observations'],'corrected geometry observation linkage')
    for path,stage in [(args.infer_receipt,'infer'),(args.evaluate_receipt,'evaluate')]:
        require(Path(path).is_file(),'corrected '+stage+' adapter receipt missing')
        receipt=read(path)
        require(receipt.get('schema')=='original_path_supervision_repair_adapter_v1' and receipt.get('stage')==stage and receipt.get('protocol',{}).get('sha256')==sha(REPAIR/'RETRAINING_PROTOCOL.json'),'corrected '+stage+' adapter protocol differs')
        bound_file(args.completion,receipt['completed_training'],'corrected '+stage+' training linkage')
    expected={x['id']:x for x in read(ORIGINAL/'INPUTS.json')['frames']}
    require(len(expected)==319 and len({x['session'] for x in expected.values()})==13,'frozen original319 identity invalid')
    observations={}
    for r in rows(args.observations):
        key=(r['id'],r['method']);require(key not in observations,'duplicate corrected observation')
        require(r['method'] in MODEL_ARMS and r['id'] in expected and r['session']==expected[r['id']]['session'] and r.get('GT_input') is False,'corrected observation identity/noGT differs')
        observations[key]=r
    require(len(observations)==957 and all((fid,arm) in observations for fid in expected for arm in MODEL_ARMS),'corrected957 observation population differs')
    geometry={}
    for r in rows(args.geometry):
        key=(r['id'],r['method']);require(key not in geometry,'duplicate corrected geometry')
        require(r['method'] in METHODS and r['id'] in expected and r['session']==expected[r['id']]['session'],'corrected geometry identity differs')
        require(not any(k in r for k in ['pose','corner','mask_audit','baseline_pose','baseline_corner']),'GT-scored fields appeared in geometry seal')
        require(r['K']==expected[r['id']]['K'] and r['xyz']==expected[r['id']]['xyz'],'corrected input camera/geometry changed')
        source_arm=r['method'] if r['method'] in MODEL_ARMS else 'IMAGE_ROLE';obs=observations[(r['id'],source_arm)]
        require(r.get('observation_raw_logits_sha256')==obs.get('raw_logits_sha256'),'corrected geometry observation-logit linkage differs')
        require(r.get('reprojections_reused_as_observations') is False and r.get('correspondence_absence_not_used_to_fill_pose') is True,'corrected observation contract changed')
        geometry[key]=r
    require(len(geometry)==1914 and all((fid,method) in geometry for fid in expected for method in METHODS),'corrected1914 geometry population differs')
    original_initial={r['id']:r['initial_pose'] for r in rows(ORIGINAL/'POSE_DIAGNOSTICS.jsonl.gz') if r['method']=='BASE_NO_MASK_STANDARD'}
    for fid in expected:
        r=geometry[(fid,'IMAGE_ROLE')];obs=observations[(fid,'IMAGE_ROLE')]
        base=expected[fid]['points']['BASE'];H=r['hidden_initial'];new=r['new_pose_estimated']
        require(r['initial_pose']==original_initial[fid],'corrected runtime final-mask historical Base pose changed')
        available=bool(r['actual_pose'].get('available'));fallback=(not new and bool(r['initial_pose'].get('available')))
        require(new==bool(r['solver']['available']) and r['pose_available']==available and r['fallback_used']==fallback and r['no_pose']==(not available),'corrected new/fallback/failure flags disagree')
        require(r['output_status']==('NEW_POSE' if new else 'BASELINE_FALLBACK' if fallback else 'POSE_FAILURE'),'corrected pose output status disagrees')
        require(len(r['native_points'])==9 and r['native_points'][8]==base[8],'corrected center metadata changed')
        require(len(H)==len(set(H)) and set(H)<=set(range(8)) and r['excluded']==H,'corrected self/exclusion IDs invalid')
        require(r['hidden_reprojected']==bool(new and H) and r['reprojected_ids']==(H if new else []),'corrected hidden replacement flags disagree')
        require(set(H).isdisjoint(r['solver'].get('fit_input_ids',[])),'excluded self corner entered corrected fit')
        selected={c['id']:c['xy'] for c in obs['corners']}
        require(len(selected)==len(obs['corners']) and set(selected)<=set(range(8)),'corrected observation corner IDs invalid')
        sparse=[selected.get(i,[None,None]) for i in range(8)]+[base[8]]
        require(r['input_points']==sparse,'corrected missing correspondence was filled or selected input changed')
        if new:
            require(available,'new corrected pose unavailable')
            for key in ['R_cf','R_physical','centroid','cf_extents','selected_hypothesis','reprojection_px']:
                require(r['actual_pose'][key]==r['solver'][key],'corrected actualPose disagrees with fitted solver '+key)
            projected=r['solver']['projected']
            require(all(r['native_points'][k]==projected[k] for k in H),'corrected hidden coordinates disagree with sealed solver reprojections')
            for c in obs['corners']:
                if c['id'] not in H:require(r['native_points'][c['id']]==c['xy'],'corrected selected visible output differs from sealed observation')
            require(all(r['native_points'][k]==base[k] for k in range(8) if k not in H and k not in selected),'corrected unselected displayed point changed')
        else:
            require(r['actual_pose']==r['initial_pose'] and r['native_points']==base,'corrected fallback/failure is not the entire original Base output')
    return dict(completion=binding(args.completion),authorization=binding(args.authorization),checkpoints=checkpoint_bindings,
                observation_seal=binding(args.observation_seal),observations=binding(args.observations),
                geometry=binding(args.geometry),pose_execution=binding(args.pose_execution),
                original319_input_identity=True,corrected_role_reference_rows=319,geometry_has_GT_scoring_fields=False,
                imports_before_this_guard=dict(torch=False,cv2=False,models=False),new_pose_calculations=0)


def checkpoint_metadata(args):
    """Future measurement only: inspect trusted frozen checkpoints before detector."""
    import torch
    protocol=read(REPAIR/'RETRAINING_PROTOCOL.json');results=[]
    for arm in MODEL_ARMS:
        ck=torch.load(Path(args.fits)/(arm+'.pt'),map_location='cpu',weights_only=False)
        require(ck['arm']==arm and ck['steps']==3000 and ck['config']==protocol['model'],'last checkpoint metadata/config differs')
        require(ck['initial_state_sha256']==protocol['initial_state_sha256'] and ck['batch_order_sha256']==protocol['batch_order_sha256'],'last checkpoint init/order differs')
        require(ck['protocol_sha256']==sha(REPAIR/'RETRAINING_PROTOCOL.json') and ck['repaired_target_sha256']==protocol['fixed_inputs']['repaired_targets']['sha256'],'last checkpoint protocol/targets differs')
        require(sum(v.numel() for v in ck['model'].values())==5890,'last checkpoint parameter count differs')
        results.append(dict(arm=arm,metadata_passed=True,parameters=5890,checkpoint=binding(Path(args.fits)/(arm+'.pt'))))
        del ck
    return results


@contextlib.contextmanager
def original_context(args,output,scratch):
    import scripts.research
    namespace=list(scripts.research.__path__)
    environment={name:os.environ.get(name) for name in ['PALLET_SOURCE_ROOT','PALLET_BASELINE_ROOT']}
    saved={};C=None;L=None;fits=None
    os.environ['PALLET_SOURCE_ROOT']=str(Path(args.source_root).resolve())
    os.environ['PALLET_BASELINE_ROOT']=str(Path(args.baseline_root).resolve())
    try:
        from scripts.research.pallet_observation_refiner_20261009_v1 import common as C,learned_infer as L
        require(C.ROOT.resolve()==Path(args.source_root).resolve(),'sourceRoot differs from imported original context')
        saved={name:getattr(C,name) for name in ['DOC','SCRATCH','read','iter_rows','binding']};fits=L.FITS
        C.source_modules()
        from scripts.research.pallet_pose_target_6d_20261006_v1 import baseline as baseline_module
        require(baseline_module.BASELINE_ROOT.resolve()==Path(args.baseline_root).resolve(),'imported immutable baselineRoot differs from requested root')
        require(baseline_module.BASE=='a22fb14beb5e8df08076385000e0d53503c1ae29','original immutable a22 baseline identifier changed')
        def redirect(path):
            path=Path(path)
            return ORIGINAL/path.name if path.parent.resolve()==output and path.name in REDIRECT_NAMES else path
        C.DOC,C.SCRATCH=output,scratch
        C.read=lambda path:saved['read'](redirect(path))
        C.iter_rows=lambda path:saved['iter_rows'](redirect(path))
        C.binding=lambda path:saved['binding'](redirect(path))
        L.FITS=Path(args.fits).resolve()
        yield C,L
    finally:
        if C is not None:
            for name,value in saved.items():setattr(C,name,value)
        if L is not None and fits is not None:L.FITS=fits
        scripts.research.__path__=namespace
        for name,value in environment.items():
            if value is None:os.environ.pop(name,None)
            else:os.environ[name]=value


def measure(args):
    check_frozen(args);output,scratch=output_guards(args);proof=prereqs(args)
    require(args.source_root and Path(args.source_root).is_dir(),'original sourceRoot missing')
    require(args.baseline_root and Path(args.baseline_root).is_dir(),'immutable original baselineRoot missing')
    before=prior_snapshot(args.prior_bindings)
    write_new(output/'RUNTIME_ADAPTER_STARTED.json',dict(schema='corrected_runtime_attempt_start_v1',prerequisites=proof,
              protocol=binding(args.protocol),configured_pipeline_calls=600,actual_pipeline_calls=0))
    scratch.mkdir(parents=True,exist_ok=True)
    metadata=checkpoint_metadata(args)
    with original_context(args,output,scratch):
        from scripts.research.pallet_observation_refiner_20261009_v1 import benchmark as B
        result=B.run('scripts.research.pallet_observation_refiner_20261009_v1.benchmark:benchmark_adapter',
                     learned_reference=Path(args.geometry).resolve(),remaining_seconds=args.remaining_seconds)
    after=prior_snapshot(args.prior_bindings);require(after==before,'protected old311 files changed')
    write_new(output/'RUNTIME_ADAPTER_RECEIPT.json',dict(schema='corrected_image_role_original_runtime_adapter_v1',
              complete=result['complete'],status=result['status'],protocol=binding(args.protocol),prerequisites=proof,
              checkpoint_metadata_checks=metadata,original_prior_files_preserved=311,
              old_context_restored=True,cached_coordinate_timing=False,new_training_updates=0,
              runtime=binding(output/'RUNTIME.json'),raw_rows=binding(output/'RUNTIME_ROWS.jsonl.gz')))
    return result


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['freeze','schedule','preflight','measure'])
    p.add_argument('--protocol',default=str(DOC/'RUNTIME_ADAPTER_PROTOCOL.json'))
    p.add_argument('--prior-bindings',default=str(DOC/'PRIOR_PUBLICATION_BINDINGS.json'))
    p.add_argument('--output',default=str(DOC));p.add_argument('--scratch',default='/dev/shm/pallet-kp-repair-runtime-private-20261010')
    p.add_argument('--fits',default='/dev/shm/pallet-kp-supervision-repair-private-20261010/learned_fits')
    p.add_argument('--completion');p.add_argument('--authorization',default=str(REPAIR/'APPROVED_ADDITIONAL_9000.json'))
    p.add_argument('--observation-seal',default=str(REPAIR/'OBSERVATION_SEAL.json'))
    p.add_argument('--observations',default=str(REPAIR/'LEARNED_OBSERVATIONS.jsonl.gz'))
    p.add_argument('--geometry',default=str(REPAIR/'LEARNED_GEOMETRY_SEALED.jsonl.gz'))
    p.add_argument('--pose-execution',default=str(REPAIR/'LEARNED_POSE_EXECUTION.json'))
    p.add_argument('--infer-receipt',default=str(REPAIR/'REPAIR_INFER_ADAPTER_RECEIPT.json'))
    p.add_argument('--evaluate-receipt',default=str(REPAIR/'REPAIR_EVALUATE_ADAPTER_RECEIPT.json'))
    p.add_argument('--source-root',default=os.environ.get('PALLET_SOURCE_ROOT'))
    p.add_argument('--baseline-root',default=os.environ.get('PALLET_BASELINE_ROOT'))
    p.add_argument('--remaining-seconds',type=float,default=3600.)
    return p


def main():
    args=parser().parse_args();args.completion=args.completion or str(Path(args.fits)/'TRAINING_COMPLETION.json')
    if args.stage=='freeze':freeze(args)
    elif args.stage=='schedule':
        w,m,proof=original_schedule();print(json.dumps(dict(arms=list(ARMS),warmup=len(w),measured=len(m),total=len(w+m),proof=proof)))
    elif args.stage=='preflight':
        check_frozen(args);output_guards(args);print(json.dumps(prereqs(args),indent=2))
    else:measure(args)


if __name__=='__main__':main()
