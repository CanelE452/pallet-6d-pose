"""Gated WOOD125 cached-pose CI and fixed-checkpoint cross-type 2D evaluation."""
import os
import sys
sys.dont_write_bytecode=True
os.environ['PYTHONDONTWRITEBYTECODE']='1'
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
from pathlib import Path
import csv
import json
import math
import hashlib
import datetime as dt
import importlib.util
import subprocess
import traceback

OUT=Path(__file__).resolve().parents[1]
REPO=OUT.parents[2]
BASE=Path('/home/minjae/Documents/pallet-research')
OLD=BASE/'dope-thesis-analysis-batch2-v1'
FLIP=BASE/'dope-loo-flip-per-type-20261026-v1'
W80=BASE/'dope-wood640-eval80-20261027-v1'
MERGED=W80/'combined372'
POSE=MERGED/'evaluation/final_v1/supplemental-pose'
PRIMARY=Path('/home/minjae/Documents/github/pallet-pose')
ADAPTER=PRIMARY/'scripts/research/pallet_dope_refiner_20261001_v1/dope_adapter.py'
MODEL_SOURCE=PRIMARY/'Deep_Object_Pose/common/models.py'
PREFIX='_docs/experiments/pallet_dope_wood125_pose_cross_type_20261029_v1'
COMMITTED={
 '7a_pose_frames':('7a865ce','_docs/experiments/pallet_dope_thesis_combined372_20261029_v1/results/supplemental_pose/FRAME_ERRORS.csv'),
 '7a_pose_summary':('7a865ce','_docs/experiments/pallet_dope_thesis_combined372_20261029_v1/results/supplemental_pose/MODEL_SUMMARY.csv'),
 '7a_main_ci':('7a865ce','_docs/experiments/pallet_dope_thesis_combined372_20261029_v1/results/main_comparisons/PAIRED_CI.csv'),
 'a4_pose_ci':('a4c8904','_docs/experiments/pallet_dope_pose_examples_20261027_v1/results/PAIRED_CI.csv'),
 'a4_manifest':('a4c8904','_docs/experiments/pallet_dope_pose_examples_20261027_v1/MANIFEST.json'),
 'a4_results_qa':('a4c8904','_docs/experiments/pallet_dope_pose_examples_20261027_v1/results/RESULTS_QA.json')}
CP={
 'WOOD_R1':FLIP/'WOOD/corrected_LOO_flip_R1.pth',
 'WOOD_R2':FLIP/'WOOD/corrected_LOO_flip_R2.pth',
 'PLASTIC_R2':FLIP/'PLASTIC/corrected_LOO_flip_R2.pth',
 'GREEN_R2':FLIP/'GREEN/corrected_LOO_flip_R2.pth',
 'SYNTHETIC_CONTROL_R1':FLIP/'SYNTHETIC_CONTROL/synthetic_control_R1.pth',
 'SYNTHETIC_CONTROL_R2':FLIP/'SYNTHETIC_CONTROL/synthetic_control_R2.pth'}
METRICS=('translation_error_m','relative_translation_error','rotation_indexed_deg','rotation_symmetry_deg')

def require(ok,msg):
    if not ok:raise RuntimeError(msg)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def csvread(p):
    with Path(p).open(newline='') as f:return list(csv.DictReader(f))
def save(p,obj):
    p=Path(p);require(p.resolve().is_relative_to(OUT.resolve()),'Write outside new prefix')
    require(not p.exists(),'Preserve existing output: '+str(p));p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def csvsave(p,rows):
    require(rows and not p.exists(),'Empty/existing CSV');p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
def log(event,**kw):
    with (OUT/'logs/RUN_LOG.jsonl').open('a') as f:
        f.write(json.dumps(dict(utc=dt.datetime.now(dt.timezone.utc).isoformat(),event=event,**kw),ensure_ascii=False,allow_nan=False)+'\n')
def git(args):
    require(REPO.name=='isolated_publish_checkout' and not REPO.is_relative_to(PRIMARY),'Unsafe Git repository')
    env={k:v for k,v in os.environ.items() if not k.startswith('GIT_')};env['GIT_TERMINAL_PROMPT']='0'
    return subprocess.run(['git',*args],cwd=REPO,env=env,check=True,capture_output=True).stdout
def frozen_module(p,name):
    before=read(OUT/'reproducibility/INPUT_HASHES_BEFORE.json')['files']
    require(str(p) in before and sha(p)==before[str(p)],'Unbound/changed source module')
    s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def bootstrap_module():
    code=OLD/'code';sys.path.insert(0,str(code))
    return frozen_module(code/'b_pose_metrics.py','frozen_original_pose_bootstrap')
def score_module():return frozen_module(W80/'code/combined_metrics.py','frozen_original_combined_metrics')
def committed(name):
    rev,path=COMMITTED[name];return git(['show',rev+':'+path])
def local_commit(name):return OUT/'reproducibility/committed_inputs'/(name+('.csv' if name!='a4_manifest' and name!='a4_results_qa' else '.json'))
def unchanged():
    before=read(OUT/'reproducibility/INPUT_HASHES_BEFORE.json')['files']
    mismatches=[dict(path=p,before=h,after=sha(p)) for p,h in before.items() if sha(p)!=h]
    return dict(passed=not mismatches,checked_count=len(before),mismatches=mismatches)
def require_unchanged():require(unchanged()['passed'],'Existing input SHA changed')

def preflight():
    require(not (OUT/'gates/PREFLIGHT.json').exists(),'Preflight already exists')
    sources=[POSE/'FRAME_ERRORS.csv',POSE/'MODEL_SUMMARY.csv',
        MERGED/'comparisons/main_v1/PAIRED_CI.csv',
        OLD/'tables/B_SUPPLEMENTAL_MIN_RMSE_FRAME_ERRORS.csv',
        OLD/'tables/B_SUPPLEMENTAL_MIN_RMSE_PAIRED_CI.csv',
        OLD/'cache/B_SUPPLEMENTAL_MIN_RMSE_BOOTSTRAP.json',
        *[OLD/'code'/n for n in ['b_pose_metrics.py','common.py','pose_geometry.py']],
        *[W80/'code'/n for n in ['combined_metrics.py','evaluate80.py']],
        MODEL_SOURCE,ADAPTER,FLIP/'PREDICTIONS.json',FLIP/'SCORES.json',
        W80/'inference/WOOD__full__R2.json',W80/'MODEL_CATALOG.json',
        W80/'MANIFEST.json',W80/'ALLOWLIST.json',W80/'AUTHORIZATION.json',
        MERGED/'MERGED_MANIFEST.json',MERGED/'MERGED_REFERENCES.json',MERGED/'COMBINED_PROTOCOL.json',
        *CP.values()]
    require(all(p.is_file() for p in sources),'Gate1 missing required local input')
    frames80=read(W80/'MANIFEST.json')['records'];allow=read(W80/'ALLOWLIST.json')['input_files']
    frames=[f for f in read(MERGED/'MERGED_MANIFEST.json')['records'] if f['kind']=='WOOD']
    require(len(frames)==125 and len(frames80)==80 and len({f['id'] for f in frames})==125,'WOOD population differs')
    require({f['id'] for f in frames80}.issubset({f['id'] for f in frames}),'80 IDs outside fixed125')
    for f in frames80:
        p=Path(f['image_path']);a=allow.get(str(p))
        require(a and a['category']=='authorized_WOOD80_RGB' and a['frame_id']==f['source_frame_id'],'Unapproved RGB')
        require(p.is_file() and sha(p)==f['image_sha256'],'Approved80 RGB changed')
        sources.append(p)
    bindings={str(p):sha(p) for p in sources}
    save(OUT/'reproducibility/INPUT_HASHES_BEFORE.json',dict(files=bindings,raw_annotations_in_inputs=False,
        RGB_scope='Approved existing WOOD80 only; no original45 forward needed'))
    commits={}
    for name,(rev,path) in COMMITTED.items():
        b=committed(name);p=local_commit(name);p.parent.mkdir(parents=True,exist_ok=True)
        with p.open('xb') as f:f.write(b)
        commits[name]=dict(commit=git(['rev-parse',rev+'^{commit}']).decode().strip(),path=path,
            sha256=hashlib.sha256(b).hexdigest(),bytes=len(b))
    require(local_commit('7a_pose_frames').read_bytes()==(POSE/'FRAME_ERRORS.csv').read_bytes(),'7a pose frames local mismatch')
    require(local_commit('7a_pose_summary').read_bytes()==(POSE/'MODEL_SUMMARY.csv').read_bytes(),'7a pose summary local mismatch')
    require(local_commit('7a_main_ci').read_bytes()==(MERGED/'comparisons/main_v1/PAIRED_CI.csv').read_bytes(),'7a main CI local mismatch')
    require(local_commit('a4_pose_ci').read_bytes()==(OLD/'tables/B_SUPPLEMENTAL_MIN_RMSE_PAIRED_CI.csv').read_bytes(),'a4 CI local mismatch')
    oldqa=read(local_commit('a4_results_qa'))
    require(oldqa['source_sha256_before'][str(OLD/'tables/B_SUPPLEMENTAL_MIN_RMSE_FRAME_ERRORS.csv')]==bindings[str(OLD/'tables/B_SUPPLEMENTAL_MIN_RMSE_FRAME_ERRORS.csv')],'Old frame source provenance differs')
    require(read(OLD/'cache/B_SUPPLEMENTAL_MIN_RMSE_BOOTSTRAP.json')['code_sha256']==bindings[str(OLD/'code/b_pose_metrics.py')],'Original bootstrap source not certified')
    rows=csvread(local_commit('7a_pose_summary'));cp_expected={}
    for name,p in CP.items():
        groups=[r for r in rows if r['model']==name]
        require(groups and len({r['checkpoint_sha256'] for r in groups})==1,'Missing/ambiguous checkpoint model '+name)
        expected=groups[0]['checkpoint_sha256'];require(bindings[str(p)]==expected,'Actual checkpoint SHA differs: '+name)
        cp_expected[name]=dict(path=str(p),sha256=expected)
    old_manifest=read(local_commit('a4_manifest'));old_wood={s:set() for s in ['R1','R2']}
    for c in old_manifest['cases']:
        if c['type']=='WOOD':
            for s in old_wood:old_wood[s].add(c['models']['flip_'+s]['checkpoint_sha256'])
    for s,hs in old_wood.items():require(hs=={cp_expected['WOOD_'+s]['sha256']},'GateA3 old/current WOOD checkpoint mismatch')
    pose_rows=csvread(local_commit('7a_pose_frames'))
    require(set(METRICS).issubset(pose_rows[0]),'Missing A metric columns')
    require({r['model'] for r in pose_rows if r['kind']=='WOOD'}.issuperset({'WOOD_R1','WOOD_R2','SYNTHETIC_CONTROL_R1','SYNTHETIC_CONTROL_R2'}),'Missing A models')
    require(all(sum(r['kind']=='WOOD' and r['model']==m for r in pose_rows)==125 for m in ['WOOD_R1','WOOD_R2','SYNTHETIC_CONTROL_R1','SYNTHETIC_CONTROL_R2']),'A125 counts differ')
    save(OUT/'reproducibility/COMMIT_INPUTS.json',commits)
    save(OUT/'reproducibility/CHECKPOINTS.json',cp_expected)
    require_unchanged()
    save(OUT/'gates/PREFLIGHT.json',dict(passed=True,gate1_structure=True,gateA3_actual_checkpoint_SHA=True,
        gate4_existing_output_search='No requested WOOD125 pose CI or PLASTIC/GREEN R2 WOOD125 result found in named committed artifacts and all paired-CI/cross-type result filename candidates under pallet-research.',
        training=False,forward_count=0,output_prefix=PREFIX,files_hashed=len(bindings)))
    log('preflight_pass',input_files=len(bindings),actual_checkpoint_SHAs=6,forward_count=0)

def float_pose(rows):
    return [{**r,**{m:None if r[m]=='' else float(r[m]) for m in METRICS}} for r in rows]
def pose_groups(kind,round_idx,old45=False):
    rows=float_pose(csvread(local_commit('7a_pose_frames')))
    if old45:
        original=csvread(OLD/'tables/B_SUPPLEMENTAL_MIN_RMSE_FRAME_ERRORS.csv')
        ids={r['id'] for r in original if r['kind']=='WOOD'}
        rows=[r for r in rows if r['id'] in ids]
    a=[r for r in rows if r['kind']==kind and r['condition']=='full' and int(r['round'])==round_idx]
    b=[r for r in rows if r['kind']==kind and r['condition']=='control' and int(r['round'])==round_idx]
    require(len(a)==len(b)==(45 if old45 else {'PLASTIC':128,'WOOD':125,'GREEN':119}[kind]),'Pose population differs')
    require([r['id'] for r in a]==[r['id'] for r in b],'Pose pair row order differs')
    return a,b
def identity(model):
    cp=read(OUT/'reproducibility/CHECKPOINTS.json')[model]
    return dict(group='WOOD__cross__'+model,kind='WOOD',condition='cross_type_evaluation',round=2,
        model=model,status='available_frozen_checkpoint',available=True,checkpoint_sha256=cp['sha256'])
def cached_wood_predictions(model):
    allpred=read(FLIP/'PREDICTIONS.json')['predictions'][model]
    frames=[f for f in read(MERGED/'MERGED_MANIFEST.json')['records'] if f['kind']=='WOOD']
    wanted={f['id'] for f in frames};old={r['id']:r for r in allpred if r['id'] in wanted}
    require(len(old)==45,'OriginalWOOD45 prediction population differs')
    if model=='WOOD_R2':
        added=read(W80/'inference/WOOD__full__R2.json')['records']
        require(len(added)==80 and not set(old)&{r['id'] for r in added},'Cached WOOD80 overlap differs')
        old.update({r['id']:r for r in added})
    return frames,old
def score_predictions(model,predictions,frames):
    # This is the only stage that reads cached reference coordinates; they go only to scoring.
    cm=score_module();references=read(MERGED/'MERGED_REFERENCES.json')['references'];g=identity(model)
    require(sum(len(references[f['id']]) for f in frames) in [346,617,963],'Scoring denominator differs')
    rows=[cm.added_row(g,f,references[f['id']],predictions[f['id']]) for f in frames]
    summary=cm.aggregate(g,rows);summary['n_correct10']=sum(r['correct10'] for r in rows)
    return summary,rows

def reproduce():
    require(read(OUT/'gates/PREFLIGHT.json')['passed'],'Preflight required')
    bm=bootstrap_module();expected=csvread(local_commit('a4_pose_ci'));checks=[]
    for kind in ['PLASTIC','GREEN','WOOD']:
        for round_idx in [1,2]:
            a,b=pose_groups(kind,round_idx,old45=(kind=='WOOD'))
            values,receipts=bm.bootstrap_comparison(a,b,kind,'flip_R'+str(round_idx),'control_R'+str(round_idx),'supplemental')
            for v in values:
                e=next(x for x in expected if x['analysis']=='supplemental' and x['kind']==kind and x['candidate_role']==v['candidate_role'] and x['comparator_role']==v['comparator_role'] and x['metric']==v['metric'])
                exact=['n_common','decision','replicates','seed','undefined_replicates','multiplicity_correction','ordered_ids_sha256','sampling_weight_sha256']
                require(all(str(v[k])==e[k] for k in exact),'GateA reproduction metadata/RNG mismatch '+kind+'/'+v['metric'])
                diffs={k:abs(v[k]-float(e[k])) for k in ['mean_difference_candidate_minus_comparator','ci95_low','ci95_high']}
                require(all(d<=1e-12 for d in diffs.values()),'GateA numeric reproduction mismatch '+kind+'/'+v['metric']+'/'+str(diffs))
                checks.append(dict(kind=kind,round=round_idx,metric=v['metric'],n_common=v['n_common'],max_absolute_difference=max(diffs.values()),ordered_ids_and_sampling_SHA_exact=True))
    csvsave(OUT/'gates/A_REPRODUCTION.csv',checks)
    frames,preds=cached_wood_predictions('WOOD_R2');s,rows=score_predictions('WOOD_R2',preds,frames)
    main=csvread(local_commit('7a_main_ci'))
    published=next(x for x in main if x['kind']=='WOOD' and x['candidate_group']=='WOOD__full__R2' and x['metric']=='PCK10')
    require(s['n_DEV']==125 and s['n_reference_points']==963 and s['n_correct10']==600,'GateB WOOD125 exact counts mismatch')
    require(s['PCK10']==float(published['candidate_value']) and abs(s['PCK10']-.6230529595)<5e-11,'GateB WOOD125 PCK mismatch')
    old_scores=read(FLIP/'SCORES.json')['models'];old_checks=[]
    for model in ['PLASTIC_R2','GREEN_R2','WOOD_R2']:
        ff,pp=cached_wood_predictions(model);ff=[f for f in ff if f['id'] in {r['id'] for r in read(FLIP/'PREDICTIONS.json')['predictions'][model]}]
        old,rr=score_predictions(model,pp,ff)
        e=old_scores[model]['groups']['type/wood_small_80x59x14']['all_annotated_reference']
        require(old['n_DEV']==45 and old['n_reference_points']==346,'B old45 population mismatch')
        for key in ['PCK10','coverage']:
            require(old[key]==e[key],'B old45 '+model+' '+key+' mismatch')
        old_checks.append(dict(model=model,PCK10=old['PCK10'],coverage=old['coverage'],n_reference=346,n_observed=old['n_observed_reference_predictions'],n_correct10=old['n_correct10'],exact_match=True))
    csvsave(OUT/'gates/B_ORIGINAL45_REPRODUCTION.csv',old_checks)
    save(OUT/'gates/B_WOOD125_REPRODUCTION.json',dict(passed=True,summary=s,checkpoint_sha256=identity('WOOD_R2')['checkpoint_sha256'],
        n_reference=963,n_correct10=600,expected_committed_PCK10=float(published['candidate_value']),forward_count=0,
        original45_predictions_reused=45,approved80_predictions_reused=80,GT_only_used_in_added_row=True))
    require_unchanged()
    save(OUT/'gates/REPRODUCTION_PASS.json',dict(passed=True,A_rows=24,A_max_absolute_difference=max(x['max_absolute_difference'] for x in checks),
        A_ordered_ids_and_sampling_SHA_exact=True,A_original_WOOD45=True,A_checkpoint_SHA_verified=True,
        B_WOOD125_PCK10=s['PCK10'],B_nref=963,B_correct10=600,B_old45_models=3,
        threshold_or_sample_change=False,NN_forwards=0,GT_used_for_checkpoint_or_threshold_selection=False))
    log('all_reproductions_pass',pose_CI_rows=24,WOOD125_correct10=600,WOOD125_reference_points=963,forward_count=0)

def task_a():
    require(read(OUT/'gates/REPRODUCTION_PASS.json')['passed'],'All reproductions required before new A')
    bm=bootstrap_module();out=[];allreceipts=[]
    for rnd in [1,2]:
        a,b=pose_groups('WOOD',rnd)
        values,receipts=bm.bootstrap_comparison(a,b,'WOOD','flip_R'+str(rnd),'control_R'+str(rnd),'supplemental')
        for x in values:
            x['source_candidate_model']='WOOD_R'+str(rnd)
            x['source_comparator_model']='SYNTHETIC_CONTROL_R'+str(rnd)
            x['evaluation_frames']=125
            if x['metric']=='translation_error_m':
                x['mean_difference_cm']=100*x['mean_difference_candidate_minus_comparator']
                x['ci95_low_cm']=100*x['ci95_low'];x['ci95_high_cm']=100*x['ci95_high']
            else:x.update(mean_difference_cm=None,ci95_low_cm=None,ci95_high_cm=None)
            out.append(x)
        allreceipts.extend(receipts)
    csvsave(OUT/'results/A_WOOD125_POSE_PAIRED_CI.csv',out)
    save(OUT/'reproducibility/A_BOOTSTRAP_RECEIPTS.json',dict(complete=True,source_function='b_pose_metrics.bootstrap_comparison',
        function_source_sha256=sha(OLD/'code/b_pose_metrics.py'),sampling_receipts=allreceipts,rows=8))
    require_unchanged();log('task_A_complete',rows=8)

def clean(value):
    if hasattr(value,'tolist'):return clean(value.tolist())
    if isinstance(value,float):return value if math.isfinite(value) else None
    if isinstance(value,(list,tuple)):return [clean(x) for x in value]
    return value
def forward_b():
    require(read(OUT/'gates/REPRODUCTION_PASS.json')['passed'],'All reproductions required before forward')
    require(not (OUT/'gates/B_FORWARD_COMPLETE.json').exists(),'Do not repeat completed forward')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    require(os.environ['CUBLAS_WORKSPACE_CONFIG']==':4096:8','Canonical CUBLAS setting differs')
    import fcntl
    import torch
    import cv2
    import numpy as np
    torch.set_num_threads(4);cv2.setNumThreads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    require(torch.cuda.is_available(),'GPU unavailable')
    lock=BASE/'dope-thesis-delivery-20261029-v1/GPU_QUEUE.lock'
    require(lock.is_file() and not lock.is_symlink(),'Existing GPU queue lock missing')
    handle=lock.open('r');fcntl.flock(handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        model_code=frozen_module(MODEL_SOURCE,'wood125_frozen_DopeNetwork')
        adapter=frozen_module(ADAPTER,'wood125_frozen_adapter');require(adapter.THRESHOLD==.3,'Decoder threshold changed')
        frames=read(W80/'MANIFEST.json')['records'];allow=read(W80/'ALLOWLIST.json')['input_files'];checkpoints=read(OUT/'reproducibility/CHECKPOINTS.json')
        require(len(frames)==80,'Forward outside approved80 population')
        for name in ['PLASTIC_R2','GREEN_R2']:
            cp=CP[name];require(sha(cp)==checkpoints[name]['sha256'],'Checkpoint changed')
            state=torch.load(cp,map_location='cpu',weights_only=True)
            net=model_code.DopeNetwork(pretrained=False,stop_at_stage=6)
            net.load_state_dict({k.removeprefix('module.'):v for k,v in state.items()},strict=True)
            net.float().requires_grad_(False).eval().to('cuda');records=[]
            for frame in frames:
                p=Path(frame['image_path']);item=allow.get(str(p))
                require(item and item['category']=='authorized_WOOD80_RGB' and item['frame_id']==frame['source_frame_id'],'Forbidden RGB read')
                require(sha(p)==frame['image_sha256'],'RGB changed')
                image=cv2.imread(str(p),cv2.IMREAD_COLOR);require(image is not None and image.shape[:2]==(480,640),'Native image differs')
                prepared=adapter.prepare(image,source_pre_padded=False)
                log('forward_attempt',model=name,id=frame['id'])
                with torch.inference_mode(),torch.autocast(device_type='cuda',enabled=False):
                    beliefs,_=net(prepared['tensor'].unsqueeze(0).to(device='cuda',dtype=torch.float32))
                maps=beliefs[-1][0].detach().cpu().numpy()
                require(maps.dtype==np.float32 and maps.shape[0]==9 and np.isfinite(maps).all(),'Invalid FP32 belief output')
                pred=adapter.decode(maps,prepared)
                records.append(dict(id=frame['id'],image_sha256=frame['image_sha256'],checkpoint_sha256=checkpoints[name]['sha256'],
                    points=clean(pred['points_original']),valid=clean(pred['valid']),confidence=clean(pred['confidence'])))
                log('forward_complete',model=name,id=frame['id'])
            save(OUT/'.private'/f'{name}_WOOD80_PREDICTIONS.json',dict(complete=True,records=records,checkpoint_sha256=checkpoints[name]['sha256'],
                runtime=dict(batch=1,dtype='float32',decoder_threshold=.3,autocast=False,TF32=False),private_prediction_coordinates_do_not_publish=True))
            del net,state;torch.cuda.empty_cache()
        require_unchanged();save(OUT/'gates/B_FORWARD_COMPLETE.json',dict(passed=True,models=['PLASTIC_R2','GREEN_R2'],
            frames_per_model=80,NN_forwards=160,GT_coordinate_reads_during_forward=0,new_PnP=0,additional_training=False))
    finally:fcntl.flock(handle.fileno(),fcntl.LOCK_UN);handle.close()

def score_b():
    require(read(OUT/'gates/B_FORWARD_COMPLETE.json')['passed'],'New80 forward completion required')
    output=[];framescores=[]
    for model in ['PLASTIC_R2','GREEN_R2']:
        frames,preds=cached_wood_predictions(model);added=read(OUT/'.private'/f'{model}_WOOD80_PREDICTIONS.json')['records']
        require(len(added)==80 and not set(preds)&{r['id'] for r in added},'Cross predictions overlap')
        preds.update({r['id']:r for r in added});require(set(preds)=={f['id'] for f in frames},'Cross prediction population differs')
        s,rr=score_predictions(model,preds,frames)
        require(s['n_DEV']==125 and s['n_reference_points']==963,'Cross evaluation denominator changed')
        output.append(s)
        for r in rr:framescores.append({k:v for k,v in r.items() if k not in ['errors_px','manual_errors_px']})
    csvsave(OUT/'results/B_CROSS_TYPE_WOOD125.csv',output)
    csvsave(OUT/'results/B_FRAME_COUNTS.csv',framescores)
    require_unchanged();save(OUT/'gates/B_SCORE_COMPLETE.json',dict(passed=True,models=2,frames_per_model=125,n_reference_points=963,CI_calculated=False))
    log('task_B_complete',models=2,frames_per_model=125,n_reference_points=963)

def main():
    phase=sys.argv[1];log('phase_start',phase=phase,command=[sys.executable,*sys.argv],code_sha256=sha(__file__),seed=20261026)
    try:
        {'preflight':preflight,'reproduce':reproduce,'a':task_a,'b-forward':forward_b,'b-score':score_b}[phase]()
        log('phase_complete',phase=phase)
        print('PHASE_PASS',phase,flush=True)
    except BaseException as e:
        proof=dict(stopped=True,phase=phase,error=str(e),traceback=traceback.format_exc())
        if (OUT/'reproducibility/INPUT_HASHES_BEFORE.json').exists():proof['input_hash_check']=unchanged()
        if not (OUT/'GATE_STOP.json').exists():save(OUT/'GATE_STOP.json',proof)
        log('gate_stop',phase=phase,error=str(e));raise

if __name__=='__main__':main()
