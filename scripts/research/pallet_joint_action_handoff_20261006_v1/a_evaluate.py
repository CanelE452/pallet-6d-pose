"""Full-denominator locked evaluator; all pose outputs are actual legacy F(q)."""
from pathlib import Path
import json,time,importlib.util,sys,csv,re
from concurrent.futures import ProcessPoolExecutor
import numpy as np,torch,cv2
from .a_common import ROOT,DOC,read,write,sha,hash_value,POSE
from .a_data import network_bank
from .geometry import build_bank as _build_bank,permute_bank
from .scorer import JointActionScorer,decode_bank
from .a_experiment import forward_bank,oracle_job
spec=importlib.util.spec_from_file_location('handoff_existing_eval_math',ROOT/'scripts/research/pallet_dim_conditioned_p_v1/eval_math.py');M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)

def build_bank(q,K,xyz,raw_hw,point_valid=None):
    try:return _build_bank(q,K,xyz,raw_hw,point_valid)
    except (cv2.error,ValueError,FloatingPointError):return dict(points=np.array(q,copy=True)[None],hypotheses=['NoOp'],details=[],reason='INITIALIZATION_FAILED')

def finite_json(v):
    if isinstance(v,dict):return {k:finite_json(x) for k,x in v.items()}
    if isinstance(v,(tuple,list)):return [finite_json(x) for x in v]
    if hasattr(v,'tolist'):return finite_json(v.tolist())
    if isinstance(v,float) and not np.isfinite(v):return None
    return v

def scored(f,q,t,method,selected=None,generating=None):
    if q is None:q=np.full((9,2),np.nan)
    detected=f['q'] is not None and np.isfinite(f['q'][:8]).any()
    corner=M.measure(q,t['gt'],t['valid'],t['permutations'],f['raw_hw'],t['matched'] and detected,detected)
    corner.update(id=f['id'],session=f['session']);pose=POSE.infer(q,f['K'],f['xyz'],f['source']);metric=POSE.metric((f['id'],pose,f['truth']))
    return dict(id=f['id'],session=f['session'],method=method,corner=corner,pose=metric,selected_index=selected,final_hypothesis=pose.get('selected_hypothesis'),generating_hypothesis=generating,raw_error_bin='<=5' if corner.get('frame_mean_px',float('inf'))<=5 else '<=10' if corner.get('frame_mean_px',float('inf'))<=10 else '>10',reference_distance_m=float(np.linalg.norm(f['truth']['t'])))

def summarize(rows):
    cs=M.summary([r['corner'] for r in rows]);ps=[r['pose'] for r in rows if r['pose']['available']]
    pose=dict(total_frames=len(rows),available=len(ps),coverage=len(ps)/len(rows),failures=len(rows)-len(ps))
    for key in ['translation_cm','rotation_deg','ADDsym_m']:
        a=[r[key] for r in ps];pose[key]=dict(median=float(np.median(a)) if a else None,P90=float(np.quantile(a,.9)) if a else None)
    return dict(corner=cs,pose=pose,NoOp=sum(bool(np.all(np.asarray(r['selected_index'])==0)) if r['selected_index'] is not None else False for r in rows),W_D_switches_from_generation=sum(r.get('generating_hypothesis') not in [None,'NoOp','PERM',r['final_hypothesis']] for r in rows))

def load_reference_points(data,split,method,seed):
    if method=='N3':
        payload=read(data.dim/f'predictions/{split}/N3_DIM_SYM_seed{seed}.json')
        return {r['id']:None if r['selected_index'] is None else np.array(r['candidates'][r['selected_index']]['keypoints_xy']) for r in payload['records']}
    prefix='heldout' if split=='SYNTH_HELDOUT' else 'REAL_DEV';z=np.load(data.root/f'data/pallet/results/pallet_posefix_replay_diagnosis_v1/predictions/seed{seed}_{prefix}.npz')
    return {str(fid):z['points'][i,0,1] for i,fid in enumerate(z['ids'])}


def evaluation_oracles(data,store,frames,banks,args,split):
    path=DOC/f'results/A_{split}_ORACLE.json'
    if path.exists():
        old=read(path);assert old['bank_binding']==store.binding;return old
    jobs=[]
    for f,b in zip(frames,banks):
        clean={k:v for k,v in f.items() if k not in ['captured','annotation','axis']}
        for arm in ['GEO','PERM']:jobs.append((clean,b if arm=='GEO' else permute_bank(b,f['id']),arm))
    start=time.monotonic();rows=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i,r in enumerate(pool.map(oracle_job,jobs,chunksize=4)):
            rows.append(r)
            if i%200==0:print('A_EVAL_ORACLE',split,i+1,len(jobs),round(time.monotonic()-start,1),flush=True)
    out=dict(rows=rows,bank_binding=store.binding,seconds=time.monotonic()-start,full_denominator=len(frames),deployment=False)
    write(path,out);return out


def real_tensor(data,frame,bank,device='cuda'):
    batch=data.real_batch([frame],device);q=network_bank(bank,frame,batch['points'][0].cpu().numpy());n=max(2,len(q));pad=np.repeat(q[:1],n,axis=0);pad[:len(q)]=q
    return batch,torch.as_tensor(pad[None],device=device),torch.as_tensor((np.arange(n)<len(q))[None],device=device)

def raw_readout(bank,index,readout,point_support,qraw):
    if readout=='J':q=bank['points'][int(index)].copy()
    else:
        q=qraw.copy()
        for i,j in enumerate(index):q[i]=bank['points'][int(j),i]
    if readout=='I':q[:8][~point_support]=qraw[:8][~point_support]
    q[8:]=qraw[8:]
    return q

@torch.no_grad()
def evaluate(data,store,real,args,protocol,trained=False):
    source_frames=[data.source_frame(r) for r in data.eval_rows];store.populate(data.eval_rows,args.workers)
    # Exact same validated supervision packet used by previous PoseFix/N3 audits.
    targets_path=data.root/'data/pallet/results/pallet_posefix_replay_diagnosis_v1/TARGETS.json';targets=read(targets_path)
    for split,frames in [('SYNTH_HELDOUT',source_frames),('REAL_DEV',real)]:
        if split=='SYNTH_HELDOUT':banks=[store.get(f['cache_row']) for f in frames]
        else:
            banks=[]
            for f in frames:
                if f['q'] is None:b=dict(points=np.full((1,9,2),np.nan),hypotheses=['NoOp'],details=[],reason='NO_DETECTION')
                else:b=build_bank(f['q'],f['K'],f['xyz'],f['raw_hw'],f['point_valid'])
                banks.append(b)
        if not trained:
            baseline_path=DOC/f'results/A_{split}_BASELINES.json'
            if baseline_path.exists():
                existing=read(baseline_path);assert existing['target_sha256']==sha(targets_path) and existing['bank_binding']==store.binding,'Cached baseline inputs changed'
            if not baseline_path.exists():
                refs={f'{name}_seed{s}':load_reference_points(data,split,name,s) for name in ['N3','PoseFix'] for s in [1,2,3]}
                base={name:[] for name in ['RAW',*refs]}
                for f in frames:
                    assert f['id'] in targets;t=targets[f['id']]
                    base['RAW'].append(scored(f,f['q'],t,'RAW'))
                    for name,p in refs.items():base[name].append(scored(f,p[f['id']],t,name))
                write(baseline_path,finite_json(dict(schema='newly_defined_handoff_A_eval_v1',rows=base,summary={k:summarize(v) for k,v in base.items()},target_sha256=sha(targets_path),bank_binding=store.binding)))
            evaluation_oracles(data,store,frames,banks,args,split)
        methods=[(arm,s) for s in [1,2,3] for arm in ['GEO','PERM']] if trained else [('N3',s) for s in [1,2,3]]
        for arm,seed in methods:
            outpath=DOC/f"results/A_{split}_{'FIT' if trained else 'FROZEN'}_{arm}_seed{seed}.json"
            ckpath=Path(args.cache_dir)/f'fits/{arm}_seed{seed}/last.pt' if trained else data.dim/f'runs/N3_DIM_SYM_seed{seed}/last.pt'
            if not ckpath.exists():continue
            cksha=sha(ckpath)
            if outpath.exists():
                old=read(outpath);assert old['checkpoint_sha256']==cksha and old['bank_binding']==store.binding and old['target_sha256']==sha(targets_path);continue
            ck=torch.load(ckpath,map_location='cpu',weights_only=False);assert ck['complete'] and ck['step']==6000
            head=JointActionScorer(5,**protocol['config']).cuda().eval();head.load_state_dict(ck['model_state_dict']);head.requires_grad_(False)
            rows={f'{bankarm}_{r}':[] for bankarm in (['GEO','PERM'] if not trained else [arm]) for r in (['I','J'] if not trained else ['J'])};begin=time.monotonic();max_masked=0
            for bankarm in ['GEO','PERM'] if not trained else [arm]:
                for start in range(0,len(frames),16 if split=='SYNTH_HELDOUT' else 1):
                    fs=frames[start:start+(16 if split=='SYNTH_HELDOUT' else 1)];bs=[b if bankarm=='GEO' else permute_bank(b,f['id']) for f,b in zip(fs,banks[start:start+len(fs)])]
                    if split=='SYNTH_HELDOUT':
                        batch=data.batch([f['cache_row'] for f in fs],supervision=False);q,v=store.tensor_batch([f['cache_row'] for f in fs],bankarm,batch,'cuda')
                    elif fs[0]['q'] is None or len(bs[0]['points'])==1:
                        for r in ['I','J'] if not trained else ['J']:rows[f'{bankarm}_{r}'].append(scored(fs[0],fs[0]['q'],targets[fs[0]['id']],r,0,'NoOp'))
                        continue
                    else:batch,q,v=real_tensor(data,fs[0],bs[0])
                    output=forward_bank(head,batch,q,v)
                    for r in ['I','J'] if not trained else ['J']:
                        _,indices=decode_bank(output,r);ix=indices.cpu().numpy();support=output['point_support'].cpu().numpy()
                        for j,(f,b) in enumerate(zip(fs,bs)):
                            qr=raw_readout(b,ix[j],r,support[j],f['q']);selected=int(ix[j]) if r=='J' else ix[j].tolist();gen=b['hypotheses'][int(ix[j])] if r=='J' else None
                            row=scored(f,qr,targets[f['id']],r,selected,gen);row['raw_error_bin']='<=5' if targets[f['id']].get('raw_mean_px',float('inf'))<=5 else 'UNASSIGNED' # subgroup is recomputed against RAW in final analysis
                            rows[f'{bankarm}_{r}'].append(row)
                    if start%320==0:print('A_EVALUATE',split,'FIT' if trained else 'FROZEN',arm,seed,bankarm,start,len(frames),flush=True)
            result=dict(schema='newly_defined_handoff_A_eval_v1',rows=rows,summary={k:summarize(v) for k,v in rows.items()},seconds=time.monotonic()-begin,checkpoint_sha256=cksha,bank_binding=store.binding,target_sha256=sha(targets_path),GT_inference_access=False,full_denominator=len(frames),selection='all3 seeds/readouts, no result-based choice',reference='renderer exact synthetic;real same2D+geometry reconstructed DEV reference',trained=trained)
            write(outpath,finite_json(result));del head;torch.cuda.empty_cache()


def paired_analysis(new,base,clusters=None):
    assert [r['id'] for r in new]==[r['id'] for r in base]
    damage=M.damage([r['corner'] for r in base],[r['corner'] for r in new]);pairs=[(a['pose'],b['pose']) for a,b in zip(new,base) if a['pose']['available'] and b['pose']['available']]
    pose=dict(common_success=len(pairs),new_only_success=sum(a['pose']['available'] and not b['pose']['available'] for a,b in zip(new,base)),base_only_success=sum(b['pose']['available'] and not a['pose']['available'] for a,b in zip(new,base)),both_failed=sum(not a['pose']['available'] and not b['pose']['available'] for a,b in zip(new,base)))
    for k in ['translation_cm','rotation_deg','ADDsym_m']:
        v=[a[k]-b[k] for a,b in pairs];pose[k]=dict(mean_paired_difference=float(np.mean(v)) if v else None,median_paired_difference=float(np.median(v)) if v else None)
    return dict(corner_damage=damage,pose_paired=pose)

def enrich_oracle_corners(data,store,frames,banks,oracle_result,path):
    targets=read(data.root/'data/pallet/results/pallet_posefix_replay_diagnosis_v1/TARGETS.json')
    by_id={f['id']:(f,b) for f,b in zip(frames,banks)};rows=[];raws={};news={}
    for row in oracle_result['rows']:
        f,geo=by_id[row['id']];bank=geo if row['arm']=='GEO' else permute_bank(geo,f['id']);q=bank['points'][row['index']];t=targets[f['id']]
        raw=M.measure(f['q'] if f['q'] is not None else np.full((9,2),np.nan),t['gt'],t['valid'],t['permutations'],f['raw_hw'],t['matched'],f['q'] is not None and np.isfinite(f['q']).any())
        new=M.measure(q,t['gt'],t['valid'],t['permutations'],f['raw_hw'],t['matched'],f['q'] is not None and np.isfinite(f['q']).any())
        raw.update(id=f['id'],session=f['session']);new.update(id=f['id'],session=f['session']);raws.setdefault(row['arm'],[]).append(raw);news.setdefault(row['arm'],[]).append(new)
        rows.append(dict(id=f['id'],arm=row['arm'],selected_index=row['index'],raw_corner=raw,oracle_corner=new,raw_pose=row['raw'],oracle_pose=row['oracle']))
    result=dict(schema='newly_defined_handoff_A_oracle_corners_v1',rows=rows,summary={a:dict(raw=M.summary(raws[a]),oracle=M.summary(news[a]),damage=M.damage(raws[a],news[a])) for a in raws},new_F_calls=0,selection='reuse identical cached bank and ADDsym_m selected candidate; no second oracle or reference-dependent candidate filter')
    write(path,finite_json(result));return result

def finalize(data,store,args,protocol,oracle):
    out={};contrasts={};comparisons={}
    real=data.real_frames()
    source=[data.source_frame(r) for r in data.oracle_rows]
    enrich_oracle_corners(data,store,source,[store.get(f['cache_row']) for f in source],oracle,DOC/'results/A_SOURCE_ORACLE_CORNERS.json')
    for split in ['SYNTH_HELDOUT','REAL_DEV']:
        basepath=DOC/f'results/A_{split}_BASELINES.json'
        if not basepath.exists():continue
        base=read(basepath)
        eval_oracle=read(DOC/f'results/A_{split}_ORACLE.json')
        oracle_lookup={(r['id'],r['arm']):r for r in eval_oracle['rows']}
        frames=[data.source_frame(r) for r in data.eval_rows] if split=='SYNTH_HELDOUT' else real
        banks=[store.get(f['cache_row']) for f in frames] if split=='SYNTH_HELDOUT' else [build_bank(f['q'],f['K'],f['xyz'],f['raw_hw'],f['point_valid']) if f['q'] is not None else dict(points=np.full((1,9,2),np.nan),hypotheses=['NoOp'],details=[],reason='NO_DETECTION') for f in frames]
        enrich_oracle_corners(data,store,frames,banks,eval_oracle,DOC/f'results/A_{split}_ORACLE_CORNERS.json')
        methods={k:v for k,v in base['rows'].items()};summaries={k:v for k,v in base['summary'].items()}
        for p in sorted(DOC.glob(f'results/A_{split}_*seed*.json')):
            x=read(p);seed=int(p.stem[-1]);prefix='FIT' if x['trained'] else 'FROZEN'
            x['summary']={key:summarize(rows) for key,rows in x['rows'].items()};write(p,x)
            for key,rows in x['rows'].items():name=f'{prefix}_{key}_seed{seed}';methods[name]=rows;summaries[name]=x['summary'][key]
        raw_by_id={r['id']:r for r in methods['RAW']}
        for method,method_rows in methods.items():
            for row in method_rows:
                raw=raw_by_id[row['id']]['corner'].get('frame_mean_px',float('inf'))
                row['raw_error_bin']='<=5' if raw<=5 else '<=10' if raw<=10 else '>10'
        strata={}
        for method,method_rows in methods.items():
            strata[method]={label:summarize([r for r in method_rows if r['raw_error_bin']==label]) for label in ['<=5','<=10','>10'] if any(r['raw_error_bin']==label for r in method_rows)}
        write(DOC/f'results/A_{split}_STRATA.json',dict(schema='newly_defined_handoff_A_strata_v1',raw_frame_mean_bins='reuse prior <=5/<=10/>10 pixel error bins; assignment from RAW only',reference_distance='continuous reference_distance_m per row, not an independent real range measurement',physical_occlusion='NOT_ESTIMATED: prior annotation visibility masks do not establish physical occluder contract',summary=strata))
        gap_rows={};gap_summary={}
        for method,rr in methods.items():
            if not method.startswith(('FIT','FROZEN')):continue
            arm=method.split('_')[1];gr=[]
            for row in rr:
                o=oracle_lookup[(row['id'],arm)];joint=row['pose'];reference=o['oracle_ADDsym_m'];gap=joint['ADDsym_m']-reference if joint['available'] and reference is not None else None
                gr.append(dict(id=row['id'],session=row['session'],method=method,actual_F_available=joint['available'],oracle_F_available=o['oracle']['available'],ADDsym_gap_m=gap,selected_index=row['selected_index'],oracle_selected_index=o['index']))
            finite=[r['ADDsym_gap_m'] for r in gr if r['ADDsym_gap_m'] is not None]
            gap_rows[method]=gr;gap_summary[method]=dict(total_frames=len(gr),common_success=len(finite),chosen_F_failure=sum(not r['actual_F_available'] for r in gr),oracle_F_failure=sum(not r['oracle_F_available'] for r in gr),mean_gap_m=float(np.mean(finite)) if finite else None,median_gap_m=float(np.median(finite)) if finite else None,P90_gap_m=float(np.quantile(finite,.9)) if finite else None,negative_beyond_numeric_tolerance=sum(x < -1e-7 for x in finite),interpretation='joint bank upper bound for J;I has a larger product space and may improve beyond this bank oracle')
            if '_J_' in method:assert gap_summary[method]['negative_beyond_numeric_tolerance']==0,(method,'J prediction must belong to scored bank')
        write(DOC/f'results/A_{split}_ORACLE_GAPS.json',dict(schema='newly_defined_handoff_A_oracle_gap_v1',rows=gap_rows,summary=gap_summary))
        scalar_path=DOC/f'results/A_{split}_METRIC_ROWS.tsv'
        fields=['id','session','method','raw_error_bin','detected','matched','supervised_corners','observed_corners','E_sym','frame_mean_px','pose_available','translation_cm','rotation_deg','ADDsym_m','IoU3D','selected_index','final_hypothesis','generating_hypothesis','reference_distance_m']
        with scalar_path.open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=fields,delimiter='\t');writer.writeheader()
            for method,rr in methods.items():
                for row in rr:
                    c=row['corner'];p=row['pose'];writer.writerow(dict(id=row['id'],session=row['session'],method=method,raw_error_bin=row['raw_error_bin'],detected=c['detected'],matched=c['matched'],supervised_corners=c.get('corners'),observed_corners=len(c.get('observed_errors',[])),E_sym=c.get('E_sym'),frame_mean_px=c.get('frame_mean_px'),pose_available=p['available'],translation_cm=p.get('translation_cm'),rotation_deg=p.get('rotation_deg'),ADDsym_m=p.get('ADDsym_m'),IoU3D=p.get('IoU3D'),selected_index=json.dumps(row['selected_index']) if row['selected_index'] is not None else '',final_hypothesis=row['final_hypothesis'],generating_hypothesis=row['generating_hypothesis'],reference_distance_m=row['reference_distance_m']))

        out[split]=summaries;comparisons[split]={}
        names=sorted({n.rsplit('_seed',1)[0] for n in methods if n.startswith(('FIT','FROZEN'))})
        for name in names:
            for ref in ['RAW','N3','PoseFix']+({'FIT_GEO_J':['FIT_PERM_J'],'FROZEN_GEO_J':['FROZEN_GEO_I'],'FROZEN_PERM_J':['FROZEN_PERM_I']}.get(name,[])):
                per=[];a=[];b=[]
                for seed in [1,2,3]:
                    nn=f'{name}_seed{seed}';bb=ref if ref=='RAW' else f'{ref}_seed{seed}'
                    if nn not in methods:continue
                    new,old=methods[nn],methods[bb];per.append(dict(seed=seed,**paired_analysis(new,old)))
                    a.append([r['corner'].get('E_sym',np.nan) for r in new]);b.append([r['corner'].get('E_sym',np.nan) for r in old])
                if len(a)==3:
                    arrays=np.array(a);bas=np.array(b);valid=np.isfinite(arrays).all(0)&np.isfinite(bas).all(0);clusters=np.array([r['session'] for r in methods[nn]])[valid] if split=='REAL_DEV' else None
                    contrast=M.contrast(arrays[:,valid],bas[:,valid],clusters);safety=all(p['corner_damage']['good5_to_bad10']<=0 for p in per) and all(summaries[f'{name}_seed{s}']['corner']['gross20']<=summaries[ref if ref=='RAW' else f'{ref}_seed{s}']['corner']['gross20'] and summaries[f'{name}_seed{s}']['pose']['available']==summaries[ref if ref=='RAW' else f'{ref}_seed{s}']['pose']['available'] for s in [1,2,3])
                    status='UNRESOLVED'
                    if contrast['CI95'][0]>0:status='WORSENED'
                    elif contrast['CI95'][1]<0 and contrast['improved_seeds']>=2 and safety:status='SUPPORTED'
                    pose_contrasts={}
                    success=np.ones(len(methods[nn]),bool)
                    for ss in [1,2,3]:
                        ref_name=ref if ref=='RAW' else f'{ref}_seed{ss}'
                        success &= np.array([r['pose']['available'] for r in methods[f'{name}_seed{ss}']]) & np.array([r['pose']['available'] for r in methods[ref_name]])
                    for metric_key in ['translation_cm','rotation_deg','ADDsym_m']:
                        aa=[];bbpose=[]
                        for ss in [1,2,3]:
                            ref_name=ref if ref=='RAW' else f'{ref}_seed{ss}'
                            aa.append([r['pose'][metric_key] for i,r in enumerate(methods[f'{name}_seed{ss}']) if success[i]])
                            bbpose.append([r['pose'][metric_key] for i,r in enumerate(methods[ref_name]) if success[i]])
                        if success.any():
                            pose_clusters=np.array([r['session'] for r in methods[nn]])[success] if split=='REAL_DEV' else None
                            pose_contrasts[metric_key]=dict(common_success_all3seed_frames=int(success.sum()),**M.contrast(np.array(aa),np.array(bbpose),pose_clusters))
                    comparisons[split][f'{name}_minus_{ref}']=dict(per_seed=per,E_sym=contrast,pose_seed_mean_paired=pose_contrasts,contract_safety_pass=safety,exploratory_verdict=status,interpretation_scope='same-bank decoder contrast only' if ref.startswith('FROZEN') else 'finite coupling control only;does not establish benefit over practical baseline' if ref.startswith('FIT') else 'whole-method versus practical baseline;exploratory DEV evidence')
        write(DOC/f'results/A_{split}_COMPARISONS.json',comparisons[split])
    fits=[read(p) for p in sorted((DOC/'A_fits').glob('*.json'))] if (DOC/'A_fits').exists() else []
    smoke=[read(p) for p in sorted((DOC/'A_smoke').glob('*.json'))] if (DOC/'A_smoke').exists() else []
    summary=dict(schema='newly_defined_handoff_A_summary_v1',oracle=oracle['summary'],summaries=out,comparisons=comparisons,fits=fits,formal_updates=sum(r['updates'] for r in fits),formal_exposures=sum(r['exposures'] for r in fits),smoke_updates=sum(r['updates'] for r in smoke),checkpoint_reuse='legacy N3 all3 seeds;prior actual N3/PoseFix outputs; frozen source features',independent_confirmation=False,bootstrap_executed=dict(resamples=10000,seed=20260917,real_primary_unit='session',synthetic_primary_unit='frame',synthetic_scenario_secondary='NOT_RUN: prior optional secondary descriptor retained;only fixed frame primary implemented'))
    write(DOC/'results/A_summary.json',summary)
    status=dict(component='A',status='DONE' if len(fits)==6 else 'DONE_NEGATIVE' if not oracle['headroom_gate'] else 'NOT_RUN_DEPENDENCY',oracle='DONE',frozen='DONE' if out else 'NOT_RUN_DEPENDENCY',formal_fits=len(fits),formal_updates=summary['formal_updates'],formal_exposures=summary['formal_exposures'],smoke_updates=summary['smoke_updates'],scientific_verdict='Use exploratory per-contrast CIs and damage/coverage;independent6D confirmation unavailable',remaining_independent_reference='BLOCKED_DATA')
    write(DOC/'A_status.json',status)
    initialization_pairs=[]
    for seed in [1,2,3]:
        pair=[r for r in fits if r['seed']==seed]
        if len(pair)!=2:continue
        assert {r['arm'] for r in pair}=={'GEO','PERM'}
        assert pair[0]['first_step']['initial_state_sha256']==pair[1]['first_step']['initial_state_sha256'],'Paired arms must start from identical weights'
        assert pair[0]['order_sha256']==pair[1]['order_sha256'] and pair[0]['params']==pair[1]['params']
        initialization_pairs.append(dict(seed=seed,initial_state_sha256=pair[0]['first_step']['initial_state_sha256'],order_sha256=pair[0]['order_sha256'],params=pair[0]['params'],updates_per_arm=[r['updates'] for r in pair]))
    offline_results=[oracle]+[read(DOC/f'results/A_{split}_ORACLE.json') for split in out]
    execution_accounting=dict(formal_updates=summary['formal_updates'],formal_exposures=summary['formal_exposures'],formal_fit_seconds=sum(r['seconds'] for r in fits),formal_failed_or_discarded_updates=0,smoke_updates=summary['smoke_updates'],smoke_seconds=sum(r['seconds'] for r in smoke),smoke_weights_discarded_before_formal=True,offline_oracle_F_calls=sum(r['actions'] for x in offline_results for r in x['rows']),offline_oracle_F_failures=sum(r['failures'] for x in offline_results for r in x['rows']),offline_oracle_seconds=sum(x['seconds'] for x in offline_results),oracle_corner_enrichment_new_F_calls=0,bank_precompute_seconds='NOT_RECORDED: precompute retries retained valid banks before any optimizer update;fit/oracle/evaluation times have individual receipts')
    evaluated_files=[read(p) for split in out for p in sorted(DOC.glob(f'results/A_{split}_*seed*.json'))]
    baseline_files=[read(DOC/f'results/A_{split}_BASELINES.json') for split in out]
    execution_accounting['method_evaluation_F_calls_from_stored_rows']=sum(len(rows) for x in evaluated_files+baseline_files for rows in x['rows'].values())
    execution_accounting['frozen_and_fitted_evaluation_seconds']=sum(x['seconds'] for x in evaluated_files)
    execution_accounting['baseline_evaluation_seconds']='NOT_RECORDED: baseline outputs reused or evaluated once,actual F calls counted by rows'
    execution_accounting['source_full_partition_counts']={name:sum(r['partition']==name for r in data.source['records']) for name in ['train','calibration','selection','heldout']}
    execution_accounting['source_cache_partition_counts']={name:int((data.partitions==name).sum()) for name in ['train','calibration','selection','heldout']}
    execution_accounting['source_cache_omitted_rows']=len(data.source['records'])-len(data.indices)
    assert execution_accounting['source_full_partition_counts']==execution_accounting['source_cache_partition_counts'],'Source denominator changed'
    execution_accounting['train_excluded_by_prior_matched_target_rules']=execution_accounting['source_full_partition_counts']['train']-len(data.train_rows)
    owned_result=r'A_(SOURCE_ORACLE(?:_CORNERS)?|ID_MANIFEST|NUMERICAL_PARITY|INDEPENDENT_PRELIMINARY_QA|summary)\.json|A_(SYNTH_HELDOUT|REAL_DEV)_(BASELINES|ORACLE(?:_CORNERS|_GAPS)?|STRATA|COMPARISONS|FROZEN_N3_seed[123]|FIT_(GEO|PERM)_seed[123])\.json|A_(SYNTH_HELDOUT|REAL_DEV)_METRIC_ROWS\.tsv'
    # Auxiliary subgroup/count audits run afterwards and have parent-manifest
    # bindings; only files produced by this A evaluator are bound here.
    artifacts=[dict(path=str(p.relative_to(ROOT)),sha256=sha(p),bytes=p.stat().st_size) for p in sorted(DOC.glob('results/A*')) if p.is_file() and re.fullmatch(owned_result,p.name)]
    write(DOC/'A_manifest.json',dict(schema='newly_defined_handoff_A_manifest_v1',protocol_sha256=sha(DOC/'A_protocol.json'),bank_binding=store.binding,cache_path=str(args.cache_dir),cache_files=[dict(name=p.name,sha256=sha(p),bytes=p.stat().st_size) for p in sorted(Path(args.cache_dir).glob('*.npy'))],artifacts=artifacts,command=f'{sys.executable} -m scripts.research.pallet_joint_action_handoff_20261006_v1.a_experiment --source-root {args.source_root} --cache-dir {args.cache_dir} --stage {args.stage} --workers {args.workers}',full_reproduction_stage='all;matching completed fits/inference/oracle receipts are reused',analysis_reaggregation='finalizing process reaggregates existing per-frame rows under its recorded analysis code;NoOp/corner/oracle-gap/bootstrap descriptor corrections add no optimizer updates',paired_initialization=initialization_pairs,execution_accounting=execution_accounting,generation_code_guard=dict(path=str(Path(args.cache_dir)/'GENERATION_CODE_BINDINGS.json'),sha256=sha(Path(args.cache_dir)/'GENERATION_CODE_BINDINGS.json')),environment=dict(torch=torch.__version__,cuda=torch.version.cuda,device=torch.cuda.get_device_name() if torch.cuda.is_available() else None),fits=fits,original_data_read_only=True,input_dependencies=dict(path=str((DOC/'INPUT_DEPENDENCIES.json').relative_to(ROOT)),sha256=sha(DOC/'INPUT_DEPENDENCIES.json')) if (DOC/'INPUT_DEPENDENCIES.json').exists() else None,generation_code_binding=read(Path(args.cache_dir)/'GENERATION_CODE_BINDINGS.json'),source_cache_full_byte_audit=dict(path=str((DOC/'SOURCE_CACHE_HASHES.json').relative_to(ROOT)),sha256=sha(DOC/'SOURCE_CACHE_HASHES.json')) if (DOC/'SOURCE_CACHE_HASHES.json').exists() else None,attempts=[dict(stage='all_source_bank_precompute',status='STOPPED_AND_CORRECTED',reason='Repeated npz member materialization per frame caused excess CPU I/O; all arrays materialized once without changing candidate coordinates or source roles',formal_updates=0,smoke_updates=0,valid_prior_oracle_reused=True),dict(stage='all_source_bank_precompute',status='STOPPED_AND_CORRECTED',reason='Before first inference/optimizer, fix artificial NaN-missing backward and keep hard-J selected output in the exact joint bank;source geometry/NoOp/affine unchanged',formal_updates=0,smoke_updates=0,completed_cached_bank_rows_at_stop=41511,valid_prior_oracle_reused=True),dict(stage='pre_formal_process_import_audit',status='RESTARTED_BEFORE_OPTIMIZER',reason='Reload frozen training modules after installing mandatory generation-code guard and initialization digest receipt;valid numerical-equivalent source banks retained',formal_updates=0,smoke_updates=0,valid_prior_oracle_reused=True)],code_files=[dict(path=str(p.relative_to(ROOT)),sha256=sha(p)) for p in sorted(Path(__file__).parent.glob('*.py')) if p.name in ['a_common.py','a_data.py','a_experiment.py','a_evaluate.py','a_verify.py','a_inference.py','a_reporting.py','geometry.py','scorer.py','test_a_contract.py']]))
