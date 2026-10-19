"""New-campaign PF complete-action union; immutable original D1 solves are inputs.

No real reference reader, model forward, optimizer, or old campaign gate is used.
Stored scalar features are losslessly retained; only bank-relative gaps change.
The expm1/log1p round trip is numerical reuse, with a declared 1e-10 feature
tolerance (bitwise identity is reported separately), never symmetry equality.
"""
from __future__ import annotations
import argparse, ast, gzip, hashlib, json, math, os, time
from collections import Counter
from copy import deepcopy
from pathlib import Path
import cv2
import numpy as np

CHECKPOINT = 'b9670d1a42db8d4e80ae1efe55eb237f05f81f7bd089cefe4051bbedd7d0a581'
METHODS = ('POSEFIX_SUBPIX_PF_ANCHOR', 'POSEFIX_SUBPIX_YOLO_ANCHOR')
MODES = ('fixed', 'A', 'B', 'AB', 'edge')
FEATURE_ATOL = 1e-10
SUBPIX = dict(winSize=[5,5], zeroZone=[-1,-1], minimum_image_extent=15,
              criteria=dict(maxCount=40, epsilon=.001))

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''): h.update(b)
    return h.hexdigest()

def read(path): return json.loads(Path(path).read_text())
def rows(path):
    with gzip.open(path,'rt') as f:
        for x in f:
            if x.strip(): yield json.loads(x)
def canonical(value): return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)
def payload_sha(value): return hashlib.sha256(canonical(value).encode()).hexdigest()
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    text=json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n'
    if path.exists():
        if path.read_text()!=text: raise ValueError('Immutable artifact changed: '+str(path))
        return
    path.write_text(text)
def save(path,records):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.pending')
    with open(temp,'wb') as raw:
        with gzip.GzipFile(fileobj=raw,mode='wb',mtime=0) as z:
            for r in records:z.write((canonical(r)+'\n').encode())
    if path.exists():
        if sha(path)!=sha(temp):raise ValueError('Immutable rows changed: '+str(path))
        temp.unlink()
    else:temp.replace(path)
def finite(value):
    if isinstance(value,dict):return {str(k):finite(v)for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [finite(v)for v in value]
    if isinstance(value,np.ndarray):return finite(value.tolist())
    if isinstance(value,np.generic):return finite(value.item())
    if isinstance(value,float)and not math.isfinite(value):return None
    return value

def primitive(repo,path,names,namespace):
    tree=ast.parse((Path(repo)/path).read_text())
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name in names]
    if {n.name for n in nodes}!=set(names):raise ValueError('Original primitive unavailable')
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(Path(repo)/path),'exec'),namespace)
    return namespace

def load_primitives(repo):
    from challenge.evaluation_v2.pnp_selector import select_pnp_hypotheses
    sn=primitive(repo,'scripts/paper/pose_metric_closure_v1/run_pose_evaluation.py',
                 ['cuboid','solve'],dict(np=np,cv2=cv2))
    pn=primitive(repo,'scripts/research/pallet_dim_conditioned_p_v1/pose.py',
                 ['rotations','infer'],dict(np=np,cv2=cv2,cuboid=sn['cuboid'],solve=sn['solve'],select_pnp_hypotheses=select_pnp_hypotheses))
    sub=primitive(repo,'scripts/research/pallet_training_free_compare_20261007_v1/methods.py',
                  ['_inputs','cap_points','correct'],dict(np=np,cv2=cv2,Counter=Counter,METHODS=('SUBPIX','CVRANK'),_configuration=lambda:{'SUBPIX':SUBPIX}))
    return pn,sub

def paths(d1,domain):
    name='REAL238'if domain=='real'else'SYNTHETIC1003'
    return Path(d1)/f'd1_upstream_v1/{name}_UPSTREAM.jsonl.gz',Path(d1)/f'd1_{domain}_candidates_v1'

def prepare(repo,d1,output):
    """Read-only source/cache/pixel contract checks; no pose production."""
    start=time.monotonic();repo,d1,out=map(Path,(repo,d1,output));out.mkdir(parents=True,exist_ok=True)
    cache=repo/'data/pallet/results/pallet_posefix_replay_diagnosis_v1'
    ip=cache/'INPUTS.json';inputs={r['id']:r for r in read(ip)}
    cp=repo/'data/pallet/results/pallet_sensors_submission_v1/runs/PRIOR1/last.pt'
    if sha(cp)!=CHECKPOINT:raise ValueError('Frozen PF checkpoint differs')
    zp=cache/'predictions/seed1_heldout.npz';z=np.load(zp,allow_pickle=False)
    if str(z['checkpoint_sha256'].item())!=CHECKPOINT:raise ValueError('PF cache checkpoint differs')
    zi={str(x):i for i,x in enumerate(z['ids'])};data=list(rows(paths(d1,'synthetic')[0]));prepared=[]
    pixels=Counter();pads=Counter()
    for f in data:
        r=inputs[f['id']];i=zi[f['id']];pad=int(r['pad']);pads[str(pad)]+=1
        q0=np.asarray(r['points'],float)-pad;pf=z['points'][i,0,1];support=np.asarray(r['valid'],bool)
        if not np.array_equal(z['points'][i,0,0],q0,equal_nan=True):raise ValueError('RAW pass0 anchor differs')
        if not np.array_equal(pf[8],q0[8],equal_nan=True)or not np.array_equal(pf[~support],q0[~support],equal_nan=True):raise ValueError('PF native slot contract differs')
        raw=cv2.imread(f['image']);im=cv2.imread(str(repo/r['image']))
        if raw is None or im is None:raise ValueError('PF source RGB unavailable')
        if sha(f['image'])!=f['image_sha256']or sha(repo/r['image'])!=r['image_sha256']:raise ValueError('Source RGB bytes changed')
        cropped=im[pad:pad+raw.shape[0],pad:pad+raw.shape[1]]
        if list(raw.shape[:2])!=f['raw_hw']or not np.array_equal(raw,cropped):raise ValueError('Prepared PF crop differs from raw RGB')
        if list(raw.shape[:2])!=r['raw_hw']:raise ValueError('Raw PF canvas differs')
        pixels['exact_raw_prepared_crop']+=1
        prepared.append(dict(id=f['id'],YOLO_points=finite(q0),PoseFix_points=finite(pf),support=support.tolist(),
                             cache_row=i,prepared_pad_removed_px=pad,usable=bool(r['usable']),
                             input_metadata_sha256=payload_sha(r),source_image=r['image'],source_image_sha256=r['image_sha256']))
    save(out/'PF_SYNTHETIC_INPUTS1003.jsonl.gz',prepared)
    sources=['scripts/research/pallet_bi_full_execution_v1/s1_union.py',
        'scripts/research/pallet_statewise_end_to_end_recovery_v1/solver.py',
        'scripts/research/pallet_boundary_action_repair_v1/action_data.py',
        'scripts/research/pallet_boundary_action_repair_v1/d1_batch.py',
        'scripts/research/pallet_dim_conditioned_p_v1/pose.py',
        'scripts/research/pallet_training_free_compare_20261007_v1/methods.py',
        'scripts/paper/pose_metric_closure_v1/run_pose_evaluation.py','challenge/evaluation_v2/pnp_selector.py']
    bind=dict(schema='FULL_CAMPAIGN_S1_INPUT_BINDINGS_V1',synthetic_rows=len(data),PF_NN_forward_needed=False,
        PF_cache_tensor='points[selected_rows,0,1]',checkpoint_sha256=CHECKPOINT,
        artifacts={str(p):sha(p)for p in (ip,zp,cp,out/'PF_SYNTHETIC_INPUTS1003.jsonl.gz')},
        code_root=str(Path(__file__).resolve().parents[3]),
        sources={s:sha(Path(__file__).resolve().parents[3]/s)for s in sources},pixel_parity=dict(pixels),padding_counts=dict(pads),
        production_GT_inputs=False,scientific_fit_or_pose_forward_executed=False,
        real_completed_pose_source=str(d1.parent/'pallet-accuracy-closeout-20261017-v1/posefix192_v1/PREDICTIONS.jsonl.gz'))
    write(out/'INPUT_BINDINGS.json',bind)
    write(out/'PREPARATION_RECEIPT.json',dict(complete=True,rows=len(data),wall_seconds=time.monotonic()-start,
          new_PF_forwards=0,new_pose_solves=0,new_H1_fits=0,inputs_sha256=sha(out/'PF_SYNTHETIC_INPUTS1003.jsonl.gz')))
    return bind

def check_plan(run):
    plan=Path(run)/'CAMPAIGN_PLAN.json'
    if not plan.exists():raise PermissionError('Root campaign plan must be frozen before new pose production')
    return sha(plan)

def synthetic_poses(repo,d1,run,prepared,output):
    plan_sha=check_plan(run);start=time.monotonic();cpu=time.process_time();cv2.setNumThreads(1)
    pn,sub=load_primitives(repo);data=list(rows(paths(d1,'synthetic')[0]));pf={r['id']:r for r in rows(Path(prepared)/'PF_SYNTHETIC_INPUTS1003.jsonl.gz')}
    out=Path(output);out.mkdir(parents=True,exist_ok=True);counts=Counter();allrows=[]
    for i,f in enumerate(data):
        folder=out/'frames'/f'{i:04d}';folder.mkdir(parents=True,exist_ok=True);dest=folder/'PREDICTIONS.jsonl.gz'
        if dest.exists():
            s=read(folder/'SEAL.json')
            if s['input_sha256']!=payload_sha(f)or s['prediction_sha256']!=sha(dest):raise ValueError('Completed PF cache differs')
            allrows.extend(rows(dest));counts['reused_completed_frames']+=1;continue
        r=pf[f['id']];image=cv2.imread(f['image'])
        if image is None or sha(f['image'])!=f['image_sha256']:raise ValueError('RGB binding differs')
        q0=np.asarray(r['YOLO_points'],float);qpf=np.asarray(r['PoseFix_points'],float);support=np.asarray(r['support'],bool)
        native,diag=sub['correct'](cv2.cvtColor(image,cv2.COLOR_BGR2GRAY),qpf,support,'SUBPIX')
        both=(sub['cap_points'](qpf,native,image.shape[1],image.shape[0],support),sub['cap_points'](q0,native,image.shape[1],image.shape[0],support))
        result=[]
        for method,q in zip(METHODS,both):
            # Synthetic native registration is the historical fixed Rx(pi).
            actual=pn['infer'](q,np.asarray(f['K']),np.asarray(f['known_physical_dimensions_xyz_m']),source=True)
            result.append(dict(id=f['id'],method=method,actual_pose=actual,native_points=finite(q),support=support.tolist(),
                image=f['image'],image_sha256=f['image_sha256'],SubPix=diag,original_F_available=actual['available'],
                PF_tensor_exact_reuse=True,PF_checkpoint_sha256=CHECKPOINT,GT_inputs=False))
        save(dest,result);write(folder/'SEAL.json',dict(id=f['id'],input_sha256=payload_sha(f),prediction_sha256=sha(dest),
             SubPix_corner_calls=diag['algorithm_corner_calls'],F_calls=2,new_PF_forwards=0,plan_sha256=plan_sha))
        allrows.extend(result);counts['new_frames']+=1;counts['new_F_calls']+=2;counts['new_SubPix_corner_calls']+=diag['algorithm_corner_calls']
        if i%100==0:print(canonical(dict(stage='S1_PF_SYNTHETIC',done=i+1,total=len(data),wall=time.monotonic()-start)),flush=True)
    save(out/'PREDICTIONS.jsonl.gz',allrows)
    seal=dict(complete=True,rows=len(allrows),images=len(data),predictions_sha256=sha(out/'PREDICTIONS.jsonl.gz'),
              plan_sha256=plan_sha,actual_calls=dict(counts),new_PF_forwards=0,wall_seconds=time.monotonic()-start,CPU_seconds=time.process_time()-cpu,
              source_native_basis='historical fixed Rx(pi), not reference conversion',accuracy_evaluation_executed=False)
    write(out/'PREDICTION_SEAL.json',seal);return seal

def pose_valid(pose,family):
    if not pose or not pose.get('available'):return False,'ORIGINAL_F_UNAVAILABLE'
    R=np.asarray(pose['R_physical'],float);t=np.asarray(pose['centroid'],float)
    if R.shape!=(3,3)or t.shape!=(3,)or not np.isfinite(R).all()or not np.isfinite(t).all():return False,'NONFINITE_OR_SHAPE'
    if not np.allclose(R.T@R,np.eye(3),atol=1e-6,rtol=0)or abs(np.linalg.det(R)-1)>=1e-6:return False,'IMPROPER_ROTATION'
    xyz=np.asarray(family['native_xyz']);Q=np.asarray(family['physical_registration_Q']);depth=((xyz-xyz.mean(0))@(R@Q.T).T+t)[:,2]
    if depth.min()<=1e-5:return False,'NONPOSITIVE_DEPTH'
    return True,'COMMON_COMPLETE_ACTION_VALID'

def decode(x):return float(np.sign(x)*np.expm1(abs(x)))
def encode(x):return float(np.sign(x)*np.log1p(abs(x)))
def recover_minima(candidates):
    """Recover cached bank minimum from its exact defined cost-minus-gap pair."""
    minima={};spread={}
    for j,mode in enumerate(MODES):
        values=[decode(c['features'][j])-decode(c['features'][26+j]) for c in candidates
                if c['features'][31+j]and c['features'][57+j]]
        if values:
            # A zero gap avoids cancellation when the global minimum's family
            # also supplies that candidate's fixed-score context.
            zeros=[decode(c['features'][j])for c in candidates if c['features'][31+j]and c['features'][57+j]and c['features'][26+j]==0.]
            conditioned=[(abs(decode(c['features'][j]))+abs(decode(c['features'][26+j])),decode(c['features'][j])-decode(c['features'][26+j]))
                         for c in candidates if c['features'][31+j]and c['features'][57+j]]
            minima[mode]=zeros[0]if zeros else min(conditioned,key=lambda x:x[0])[1];spread[mode]=max(values)-min(values)
        else:minima[mode]=None;spread[mode]=None
    return minima,spread

def update_gaps(old_candidates,minima,pfmin):
    out=deepcopy(old_candidates)
    for c in out:
        for j,mode in enumerate(MODES):
            if minima[mode]is not None and pfmin[mode]is not None and pfmin[mode]<minima[mode]and c['features'][57+j]:
                c['features'][26+j]=encode(decode(c['features'][26+j])+minima[mode]-pfmin[mode])
    return out

def union_frame(frame,erun,old,bundle,pfrows,config):
    from scripts.research.pallet_statewise_end_to_end_recovery_v1 import solver as S
    from scripts.research.pallet_boundary_action_repair_v1 import action_data as A
    inp=dict(frame,packet=erun['packet'],families=erun['families'],fallback=frame['initial_pose'])
    scale=old['object_scale_m'];minima,spread=recover_minima(old['candidates']);models=None
    additions=[];rejections=[];duplicates=[];oldc=deepcopy(old['candidates'])
    for row in pfrows:
        if row.get('image_sha256')not in (None,frame['image_sha256']):raise ValueError('PF/D1 RGB identity differs')
        if 'K'in row and row['K']!=frame['K']:raise ValueError('PF/D1 camera differs')
        pose=row['actual_pose'];valid,reason=pose_valid(pose,erun['families']['ORIGINAL_WD'])
        if not valid:rejections.append(dict(method=row['method'],reason=reason));continue
        match=next((c for c in oldc+additions if A.same_pose(c['actual_pose'],pose,scale)),None)
        if match is not None:
            match['source_aliases']=sorted(set(match.get('source_aliases',[]))|{row['method']})
            match['complete_action_aliases']=sorted(set(match.get('complete_action_aliases',[]))|{row['method']})
            match['eligible']=True;duplicates.append(dict(method=row['method'],candidate_id=match['candidate_id']));continue
        additions.append(dict(candidate_id=row['method'],actual_pose=deepcopy(pose),source_aliases=[row['method']],source_baseline_aliases=[],
             complete_action_aliases=[row['method']],eligible=True,object_scale_m=scale))
    pfscores=[];pfbank=[];pfmin={m:None for m in MODES};old_cost_reconstructions=0
    if additions:
        models=S.compile_frame(inp,config,bundle=bundle)
        if any(v is None for v in minima.values()):
            oldbank=[dict(candidate_id=c['candidate_id'],R_physical=c['actual_pose']['R_physical'],centroid=c['actual_pose']['centroid'])for c in oldc]
            oldscores=[S.candidate_scores(c,models)for c in oldbank];old_cost_reconstructions=len(oldbank)
            for mode in MODES:
                if minima[mode]is None:
                    values=[s[f]['scores'][mode]for s in oldscores for f in models if S._finite_feature(s[f]['scores'][mode])is not None]
                    minima[mode]=min(values)if values else None
        pfbank=[dict(candidate_id=c['candidate_id'],source='PF_COMPLETE_ACTION',R_physical=c['actual_pose']['R_physical'],centroid=c['actual_pose']['centroid'])for c in additions]
        pfscores=[S.candidate_scores(c,models)for c in pfbank]
        for mode in MODES:
            vals=[s[f]['scores'][mode]for s in pfscores for f in models if S._finite_feature(s[f]['scores'][mode])is not None]
            pfmin[mode]=min(vals)if vals else None
        newfeatures=S.features_for_bank(inp,models,pfbank,pfscores)
        for source,nf,score in zip(additions,newfeatures,pfscores):
            nf.update({k:deepcopy(v)for k,v in source.items()if k not in ('features','feature_context_family')});nf['eligible']=True
            nf['admission_rule']='COMMON_COMPLETE_ACTION_VALID';family=nf['feature_context_family']
            for j,mode in enumerate(MODES):
                available=[x for x in (minima[mode],pfmin[mode])if x is not None]
                cost=score[family]['scores'][mode]
                if available and S._finite_feature(cost)is not None:nf['features'][26+j]=encode(cost-min(available));nf['features'][57+j]=1.
                else:nf['features'][26+j]=0.;nf['features'][57+j]=0.
        additions=newfeatures
    result=deepcopy(old);result['candidates']=update_gaps(oldc,minima,pfmin)+additions
    result['s1_union']=dict(original_bank_exact_pose_reuse=True,old_candidate_count=len(oldc),new_unique_PF_actions=len(additions),
        complete_action_duplicates=duplicates,PF_rejections=rejections,old_minima=minima,new_PF_minima=pfmin,
        old_minima_inverse_transform_spread=spread,old_cost_reconstructions_for_missing_context=old_cost_reconstructions,numerical_feature_atol=FEATURE_ATOL,
        admission='N0/E0 unchanged; available PF actions properR+positiveDepth; original OTHER eligibility unchanged',
        old_individual_feature_scalars_bitwise_reused=26,bank_relative_gap_features_recomputed=5,
        normalization_and_188D_eligible_bank_summaries='must be recomputed by original action_data.inputs with new train normalizer')
    A.validate_frame(result)
    return result,dict(new_PF_scores=pfscores,new_PF_bank=pfbank,old_minima=minima,
         union_minima={m:min(x for x in (minima[m],pfmin[m])if x is not None)if any(x is not None for x in (minima[m],pfmin[m]))else None for m in MODES})

def features(d1,old_run,run,pfpath,domain,output):
    plan_sha=check_plan(run);started=time.monotonic();cpu=time.process_time();cv2.setNumThreads(1)
    inpath,bankdir=paths(d1,domain);data=list(rows(inpath));pfs={}
    pfseal=read(Path(pfpath).parent/'PREDICTION_SEAL.json')
    if not pfseal.get('complete')or sha(pfpath)!=pfseal['predictions_sha256']:raise ValueError('PF prediction seal changed')
    for r in rows(pfpath):
        if r.get('method')in METHODS:pfs.setdefault(r['id'],[]).append(r)
    if set(pfs)!={f['id']for f in data}or any(len(x)!=2 for x in pfs.values()):raise ValueError('Exact all-ID two-anchor PF outputs required')
    from scripts.research.pallet_boundary_action_repair_v1.d1_batch import source_bindings
    closure=source_bindings();closure['scripts/research/pallet_bi_full_execution_v1/s1_union.py']=sha(__file__)
    config=read(Path(old_run)/'CONFIG_FROZEN.json');out=Path(output);out.mkdir(parents=True,exist_ok=True);allrows=[];counts=Counter()
    write(out/'SOURCE_DEPENDENCY_CLOSURE.json',closure)
    for i,frame in enumerate(data):
        src=bankdir/'frames'/f'{i:04d}';dest=out/'frames'/f'{i:04d}';dest.mkdir(parents=True,exist_ok=True)
        deps={n:sha(src/n)for n in ('E_RUN.jsonl.gz','FEATURES.jsonl.gz','BOUNDARY_BUNDLE.jsonl.gz')}
        key=dict(original_artifacts=deps,input_row_sha256=payload_sha(frame),PF_rows_sha256=payload_sha(pfs[frame['id']]),config_sha256=sha(Path(old_run)/'CONFIG_FROZEN.json'),source_dependency_closure_sha256=payload_sha(closure))
        fp=dest/'FEATURES.jsonl.gz';sp=dest/'SEAL.json'
        if sp.exists():
            seal=read(sp)
            if seal['dependencies']!=key or seal['features_sha256']!=sha(fp):raise ValueError('Union cache changed')
            allrows.extend(rows(fp));counts['completed_row_cache_hits']+=1;continue
        erun=next(rows(src/'E_RUN.jsonl.gz'));old=next(rows(src/'FEATURES.jsonl.gz'));bundle=next(rows(src/'BOUNDARY_BUNDLE.jsonl.gz'))
        if erun['id']!=frame['id']or old['id']!=frame['id']:raise ValueError('Original ordered IDs differ')
        tick=time.monotonic();result,costs=union_frame(frame,erun,old,bundle,pfs[frame['id']],config)
        # Preserve every actual visited endpoint and original alias, verbatim.
        save(dest/'ORIGINAL_E_RUN_EXACT.jsonl.gz',[erun]);save(dest/'PF_INDIVIDUAL_COSTS.jsonl.gz',[costs]);save(fp,[result])
        seal=dict(id=frame['id'],complete=True,dependencies=key,features_sha256=sha(fp),
             original_E_RUN_payload_sha256=payload_sha(erun),original_E_RUN_exact_reuse=True,
             generated_endpoint_count=len(erun['candidates'])+2,old_unique_count=len(old['candidates']),
             union_unique_count=len(result['candidates']),eligible_count=sum(c['eligible']for c in result['candidates']),
             new_unique_PF_count=result['s1_union']['new_unique_PF_actions'],new_pose_optimizer_calls=0,
             new_individual_cost_evaluations=result['s1_union']['new_unique_PF_actions'],wall_seconds=time.monotonic()-tick,plan_sha256=plan_sha)
        write(sp,seal);allrows.append(result);counts['new_rows']+=1;counts['exact_original_E_RUN_reuses']+=1;counts['new_unique_PF_costs']+=seal['new_unique_PF_count']
        if i%100==0:print(canonical(dict(stage='S1_UNION_FEATURES',domain=domain,done=i+1,total=len(data),wall=time.monotonic()-started)),flush=True)
    save(out/'HEAD_FEATURES.jsonl.gz',allrows)
    seal=dict(schema='NEW_CAMPAIGN_S1_FEATURE_SEAL_V1',complete=True,rows=len(allrows),domain=domain,
        head_features_sha256=sha(out/'HEAD_FEATURES.jsonl.gz'),input_sha256=sha(inpath),PF_predictions_sha256=sha(pfpath),
        plan_sha256=plan_sha,source_sha256=sha(__file__),ordered_ids=[f['id']for f in data],actual_calls=dict(counts),
        new_pose_optimizer_calls=0,new_NN_forwards=0,e0_independently_executed=True,
        old_individual_cost_cache='31 stored signedlog scalars+validbits, algebraic cost-minus-gap reuse; 1e-10 tolerance',
        original_visited_endpoint_artifacts_preserved=True,old_admission_unchanged=True,matched_baseline_refit_required=False,
        wall_seconds=time.monotonic()-started,CPU_seconds=time.process_time()-cpu,before_current_real_reference_decode=True)
    write(out/'FEATURE_SEAL.json',seal);return seal

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','synthetic-poses','features'])
    for k in ('repo','d1','run','old-run','prepared','pfpath','output'):p.add_argument('--'+k,type=Path)
    p.add_argument('--domain',choices=['real','synthetic']);a=p.parse_args()
    if a.stage=='prepare':r=prepare(a.repo,a.d1,a.output)
    elif a.stage=='synthetic-poses':r=synthetic_poses(a.repo,a.d1,a.run,a.prepared,a.output)
    else:r=features(a.d1,a.old_run,a.run,a.pfpath,a.domain,a.output)
    print(canonical(r),flush=True)
if __name__=='__main__':main()
