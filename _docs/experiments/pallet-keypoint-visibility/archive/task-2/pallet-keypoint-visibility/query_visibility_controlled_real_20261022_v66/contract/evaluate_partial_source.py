#!/usr/bin/env python3
"""Source-only partial-label validation of the unchanged v65/v66 six-joint head.

UNKNOWN is a partial fact, never a negative. The fixed selection statistic is
the mean of four per-source informative-center partial NLLs. This program
reports that statistic; it does not choose a model or tune any threshold.
Actual execution requires complete source cache/model seals. --mock uses only
generated arrays and a random, zero-update CPU model for an API smoke check.
"""
from pathlib import Path
from collections import Counter, defaultdict
import argparse, csv, hashlib, importlib.util, json, sys, time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
SOURCES=('SYNTH','ORIGINAL_YCB','CONTROLLED_YCB','PALLOX')
PAIRS=np.asarray(((0,0),(1,0),(1,1),(2,0),(2,1),(2,2)),np.int8)
NAMES=('V','E','O','U')
MODEL_CODE_SHA='068e9ea147b093371244b02779e1df0e31e22a747063eb26b64be12419492144'
INPUT_KEYS=('x','rgb_native','sam','context','box_grid','global_rgb','global_sam','global_prompt',
            'target_index','global_index64','global_index8','global_query_valid')

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def write(p,j):Path(p).write_text(json.dumps(j,indent=2,allow_nan=False)+'\n')
def ratio(a,b):return float(a)/int(b)if b else None
def load_npz(p,keys=None):
    with np.load(p,allow_pickle=False)as z:return{k:z[k]for k in(z.files if keys is None else keys)}

def probabilities(logits):
    z=np.asarray(logits,np.float64)
    if z.ndim!=2 or z.shape[1]!=6 or not np.isfinite(z).all():raise ValueError('finite N,6 joint logits')
    z=z-z.max(1,keepdims=True);p=np.exp(z);return p/p.sum(1,keepdims=True)

def partial_nll(logits,allowed):
    z=np.asarray(logits,np.float64);a=np.asarray(allowed,np.uint8)
    if z.shape!=(len(a),6)or not np.isin(a,np.arange(1,64)).all():raise ValueError('aligned six-joint compatible masks1..63')
    permit=(a[:,None]&(1<<np.arange(6))[None])!=0
    result=np.logaddexp.reduce(z,axis=1)-np.logaddexp.reduce(np.where(permit,z,-np.inf),axis=1)
    result[a==63]=0. # Algebraic identity, including native-invalid placeholders.
    return result

def marginal(p):
    p=np.asarray(p,np.float64)
    exact=np.stack([p[:,0],p[:,1]+p[:,2],p[:,3]+p[:,4]+p[:,5]],1)
    near=np.stack([p[:,0]+p[:,1]+p[:,3],p[:,2]+p[:,4],p[:,5]],1)
    return np.stack([exact,near],1)

def confident_states(p):
    best=np.argmax(p,axis=1);confidence=np.max(p,axis=1)
    return np.where((confidence>=.5)[:,None],PAIRS[best],3).astype(np.int8)

def binary_metrics(truth,pred):
    truth=np.asarray(truth,bool);pred=np.asarray(pred,bool)
    tp=int((truth&pred).sum());fp=int((~truth&pred).sum());fn=int((truth&~pred).sum());tn=int((~truth&~pred).sum())
    precision,recall=ratio(tp,tp+fp),ratio(tp,tp+fn)
    return dict(TP=tp,FP=fp,FN=fn,TN=tn,N=len(truth),precision=precision,recall=recall,
                F1=ratio(2*tp,2*tp+fp+fn),precision_scope='Conditional on known exact/radius2 GT only; UNKNOWN excluded, never negative')

def support_metrics(gt,pred,pvisible,valid,allowed):
    gt=np.asarray(gt,np.int8);pred=np.asarray(pred,np.int8);valid=np.asarray(valid,bool)
    known=valid&(gt>=0);unknown=valid&(gt<0)
    table=np.zeros((3,4),np.int64)
    np.add.at(table,(gt[known],pred[known]),1)
    classes={}
    for c in range(3):
        tp=int(table[c,c]);den=int(table[:,c].sum());support=int(table[c].sum())
        classes[NAMES[c]]=dict(TP=tp,predicted_on_known_GT=den,GT_support=support,
                              precision=ratio(tp,den),recall=ratio(tp,support),F1=ratio(2*tp,den+support))
    return dict(total_rows=len(gt),native_valid=int(valid.sum()),native_invalid=int((~valid).sum()),
                known_GT=int(known.sum()),UNKNOWN_GT=int(unknown.sum()),
                uninformative_allowed63_native_valid=int((valid&(allowed==63)).sum()),
                informative_native_valid=int((valid&(allowed!=63)).sum()),
                confusion_3x4=table.tolist(),GT_row_order=['V','E','O'],prediction_column_order=['V','E','O','U'],
                class_metrics=classes,known_subset_precision_is_conditional=True,
                visible_binary_pV_ge_0_5=binary_metrics(gt[known]==0,pvisible[known]>=.5),
                abstention=dict(on_all_native_valid=int((valid&(pred==3)).sum()),
                                rate_all_native_valid=ratio((valid&(pred==3)).sum(),valid.sum()),
                                on_known_GT=int((known&(pred==3)).sum()),rate_known_GT=ratio((known&(pred==3)).sum(),known.sum())),
                predictions_on_UNKNOWN=dict(N=int(unknown.sum()),V=int((unknown&(pred==0)).sum()),
                                            E=int((unknown&(pred==1)).sum()),O=int((unknown&(pred==2)).sum()),
                                            U=int((unknown&(pred==3)).sum()),binary_pV_positive=int((unknown&(pvisible>=.5)).sum()),
                                            interpretation='Descriptive predictions; no FP/FN assigned to UNKNOWN GT'))

def all_source_metrics(d,p,pred=None):
    states=marginal(p);pred=confident_states(p)if pred is None else pred
    valid=d['dense_valid'][:,4,4];allowed=d['dense_allowed'][:,4,4]
    gts=np.stack([d['dense_exact_state'][:,4,4],d['dense_radius2_state'][:,4,4]],1)
    result={}
    for source in SOURCES:
        ix=d['source'].astype(str)==source
        result[source]={name:support_metrics(gts[ix,k],pred[ix,k],states[ix,k,0],valid[ix],allowed[ix])
                        for k,name in enumerate(['exact','radius2'])}
    return result

def nll_summary(d,logits):
    allowed=d['dense_allowed'][:,4,4];valid=d['dense_valid'][:,4,4]
    values=partial_nll(logits,allowed);info=valid&(allowed!=63)
    by={}
    for s in SOURCES:
        ix=(d['source'].astype(str)==s)&info
        by[s]=dict(informative_centers=int(ix.sum()),partial_NLL=float(values[ix].mean())if ix.any()else None)
    means=[by[s]['partial_NLL']for s in SOURCES]
    return dict(common4domain_mean_informative_center_partial_NLL=float(np.mean(means))if all(x is not None for x in means)else None,
                by_source=by,rule='Equal mean of four source means; all63 and native-invalid excluded. No model choice performed.'),values

def visible_baseline_metrics(d,pv,predicted_states):
    valid=d['dense_valid'][:,4,4];allowed=d['dense_allowed'][:,4,4]
    gt=np.stack([d['dense_exact_state'][:,4,4],d['dense_radius2_state'][:,4,4]],1)
    result={}
    for s in SOURCES:
        ix=d['source'].astype(str)==s
        result[s]={name:support_metrics(gt[ix,k],predicted_states[ix,k],pv[ix,k],valid[ix],allowed[ix])
                   for k,name in enumerate(['exact','radius2'])}
    return result

def baselines(d,prior=None):
    # Stored SAM is clipped raw logit/8 (exact and native radius2 maximum).
    raw=d['sam'][:,:,4,4].astype(np.float64)*8
    pv=np.empty_like(raw);pos=raw>=0;pv[pos]=1/(1+np.exp(-raw[pos]));er=np.exp(raw[~pos]);pv[~pos]=er/(1+er)
    sam_pred=np.where(pv>=.5,0,3).astype(np.int8)
    result=dict(frozen_raw_SAM_visible_only=dict(rule='sigmoid(saved_normalized_SAM*8)>=.5 => V, else U; E/O not inferred; binary visibility at.5',
        metrics=visible_baseline_metrics(d,pv,sam_pred),partial_joint_NLL=None),
        constant_visible=dict(metrics=visible_baseline_metrics(d,np.ones_like(pv),np.zeros_like(sam_pred))),
        constant_outside=dict(metrics=visible_baseline_metrics(d,np.zeros_like(pv),np.full_like(sam_pred,2))))
    if prior is not None:
        per={}
        for s in SOURCES:
            ix=d['source'].astype(str)==s;entry=prior['sources'].get(s)
            if entry is None or entry['unique_joint_centers']==0:
                per[s]=dict(available=False,reason='No sealed TRAIN unique-joint prior for source');continue
            pp=np.repeat(np.asarray(entry['probability'],np.float64)[None],int(ix.sum()),axis=0)
            local={k:v[ix]if k in ['dense_valid','dense_allowed','dense_exact_state','dense_radius2_state','source']else v for k,v in d.items()}
            per[s]=dict(available=True,TRAIN_prior=entry,metrics=all_source_metrics(local,pp)[s],
                        prior_scope='Only uniquely known TRAIN joint centers; partial ambiguous labels not imputed')
        result['TRAIN_unique_joint_prior']=per
    return result,pv

def pair_counterfactual(d,p):
    st=marginal(p);pred=confident_states(p);valid=d['dense_valid'][:,4,4]
    gt=np.stack([d['dense_exact_state'][:,4,4],d['dense_radius2_state'][:,4,4]],1)
    groups=defaultdict(dict)
    for i,s in enumerate(d['source'].astype(str)):
        if s!='PALLOX':continue
        gid=str(d['global_id'][d['target_index'][i]]);sample=str(d['sample_id'][i]);group=str(d['group_id'][i])
        view=gid.rsplit(':',1)[-1];slot=sample.rsplit(':',1)[-1]
        if view not in ['A','B']or not slot.isdecimal():raise ValueError('PALLOX target/sample ID does not encode A/B and query slot')
        key=(group,int(slot))
        if view in groups[key]:raise ValueError('Duplicate paired view/query')
        groups[key][view]=i
    rows=[];result={}
    if any(set(x)!=set(['A','B'])for x in groups.values()):raise ValueError('Incomplete PALLOX pair/query: preserve complete all-pair cohort')
    for k,name in enumerate(['exact','radius2']):
        c=Counter();dv=[];de=[]
        for (group,slot),ab in sorted(groups.items()):
            ia,ib=ab['A'],ab['B'];c['all_pair_queries']+=1
            if not(valid[ia]and valid[ib]):c['native_invalid_pair_queries']+=1;continue
            c['native_valid_pair_queries']+=1
            if gt[ia,k]<0 or gt[ib,k]<0:c['UNKNOWN_in_either_GT']+=1;continue
            c['both_GT_known']+=1
            if {int(gt[ia,k]),int(gt[ib,k])}!={0,1}:c['known_but_not_V_E_change']+=1;continue
            visible,covered=(ia,ib)if gt[ia,k]==0 else(ib,ia)
            deltaV=float(st[visible,k,0]-st[covered,k,0]);deltaE=float(st[covered,k,1]-st[visible,k,1])
            c['known_V_E_changed_queries']+=1;c['correct_pV_direction']+=int(deltaV>0);c['correct_pE_direction']+=int(deltaE>0)
            c['both_probability_directions_correct']+=int(deltaV>0 and deltaE>0)
            c['pV_direction_ties']+=int(deltaV==0);c['pE_direction_ties']+=int(deltaE==0)
            confident=pred[visible,k]!=3 and pred[covered,k]!=3;c['both_joint_confident']+=int(confident)
            c['both_target_states_correct_confident']+=int(pred[visible,k]==0 and pred[covered,k]==1)
            c['binary_visible_and_covered_correct']+=int(st[visible,k,0]>=.5 and st[covered,k,0]<.5)
            dv.append(deltaV);de.append(deltaE)
            rows.append(dict(group_id=group,slot=slot,support=name,visible_sample_id=str(d['sample_id'][visible]),
                covered_sample_id=str(d['sample_id'][covered]),pV_visible=float(st[visible,k,0]),pV_covered=float(st[covered,k,0]),
                pE_visible=float(st[visible,k,1]),pE_covered=float(st[covered,k,1]),delta_pV=deltaV,delta_pE=deltaE,
                pV_correct_direction=deltaV>0,pE_correct_direction=deltaE>0,
                visible_predicted=NAMES[pred[visible,k]],covered_predicted=NAMES[pred[covered,k]],
                both_joint_confident=bool(confident),both_states_correct=bool(pred[visible,k]==0 and pred[covered,k]==1)))
        n=c['known_V_E_changed_queries'];result[name]=dict(counts=dict(c),
            pV_direction_accuracy=ratio(c['correct_pV_direction'],n),pE_direction_accuracy=ratio(c['correct_pE_direction'],n),
            both_direction_accuracy=ratio(c['both_probability_directions_correct'],n),
            both_target_states_correct_rate=ratio(c['both_target_states_correct_confident'],n),
            delta_pV_mean=float(np.mean(dv))if dv else None,delta_pE_mean=float(np.mean(de))if de else None,
            GT_scope='Conditional paired teacher V/E; actual same fixed query in A/B, not independent human gold')
    return result,rows

def train_prior(path,deny,VAL_image_hashes):
    d=load_npz(path,['source','dense_allowed','dense_valid','image_sha256']);out={}
    if not set(d['source'].astype(str))<=set(SOURCES):raise ValueError('Unexpected TRAIN prior source')
    hashes=set(d['image_sha256'].astype(str))
    if hashes&deny or hashes&VAL_image_hashes:raise ValueError('TRAIN prior private/VAL image contamination')
    a=d['dense_allowed'][:,4,4];v=d['dense_valid'][:,4,4];unique=v&((a&(a-1))==0)
    for s in SOURCES:
        ix=(d['source'].astype(str)==s)&unique;counts=np.array([int((ix&(a==(1<<k))).sum())for k in range(6)])
        out[s]=dict(unique_joint_centers=int(counts.sum()),joint_counts=counts.tolist(),probability=(counts/counts.sum()).tolist()if counts.sum()else None)
    return dict(TRAIN_cache=ref(path),definition='Empirical six-joint frequencies on native-valid, uniquely known TRAIN centers only; no smoothing or partial-label imputation',sources=out)

def validate_cache(d,deny):
    n=len(d['x']);t=len(d['global_rgb'])
    shapes=dict(x=(n,65,9,9),rgb_native=(n,75,9,9),sam=(n,2,9,9),context=(n,64),box_grid=(n,4,9,9),
        dense_allowed=(n,9,9),dense_valid=(n,9,9),dense_exact_state=(n,9,9),dense_radius2_state=(n,9,9),
        source=(n,),sample_id=(n,),group_id=(n,),image_sha256=(n,),target_index=(n,),
        global_rgb=(t,3,64,64),global_sam=(t,1,64,64),global_prompt=(t,1,64,64),
        global_id=(t,),global_image_sha256=(t,),global_source=(t,),global_index64=(n,9,9),global_index8=(n,9,9),global_query_valid=(n,9,9))
    for k,s in shapes.items():
        if k not in d or d[k].shape!=s:raise ValueError((k,s))
    if set(d['source'].astype(str))!=set(SOURCES):raise ValueError('Same four source validation domains required')
    if set(d['image_sha256'].astype(str))&deny:raise ValueError('CURRENT223 image found in source cache')
    if len(set(d['sample_id']))!=n or len(set(d['global_id']))!=t:raise ValueError('Duplicate source identity')
    a=d['dense_allowed'];v=d['dense_valid']
    if a.dtype!=np.uint8 or not np.isin(a,np.arange(1,64)).all()or v.dtype!=bool or not(a[~v]==63).all():raise ValueError('Native partial label contract')
    for k in ['dense_exact_state','dense_radius2_state']:
        if not np.isin(d[k],[-1,0,1,2]).all()or not(d[k][~v]==-1).all():raise ValueError('Partial hard states/invalid contract')
    # Known marginal truth must agree with the compatible six-state set. This
    # rejects malformed caches which would give NLL and precision different GT.
    for key,class_bits in [('dense_exact_state',[1,6,56]),('dense_radius2_state',[11,20,32])]:
        possible=np.stack([(a&bits)!=0 for bits in class_bits],axis=1)
        expected=np.where(possible.sum(1)==1,possible.argmax(1),-1).astype(np.int8)
        expected[~v]=-1
        if not np.array_equal(d[key],expected):raise ValueError('Hard marginal incompatible with partial mask: '+key)
    ti=d['target_index']
    if ti.dtype!=np.int64 or ti.min()<0 or ti.max()>=t:raise ValueError('target mapping')
    if not np.array_equal(d['global_image_sha256'][ti],d['image_sha256'])or not np.array_equal(d['global_source'][ti],d['source']):raise ValueError('Target identity/source join')
    for k in ['x','rgb_native','sam','context','box_grid','global_rgb','global_sam']:
        if not np.isfinite(d[k]).all():raise ValueError('Nonfinite input '+k)
    if d['rgb_native'].dtype!=np.uint8 or d['global_rgb'].dtype!=np.uint8:raise ValueError('uint8 RGB expected')
    if np.abs(d['sam']).max()>4.000001 or np.abs(d['global_sam']).max()>4.000001:raise ValueError('Stored normalized SAM range')
    for key,limit in [('global_index64',4096),('global_index8',64)]:
        if d[key].dtype!=np.int64 or d[key].min()<0 or d[key].max()>=limit:raise ValueError('Global input index bounds')
    qv=d['global_query_valid'];i64=d['global_index64'];i8=d['global_index8']
    if qv.dtype!=bool or d['global_prompt'].dtype!=bool:raise ValueError('Boolean input masks required')
    if not np.array_equal(i8[qv],((i64//64)//8*8+(i64%64)//8)[qv]):raise ValueError('Coarse global input mapping')
    if(i64[~qv]!=0).any()or(i8[~qv]!=0).any():raise ValueError('Invalid input mapping placeholder')

def model_inputs(d,ix,torch):
    # This explicit whitelist is the entire model-input recipe. GT, source,
    # IDs, depth and pose are not arguments to the learned model.
    def tensor(k,index,dtype=torch.float32):return torch.as_tensor(np.ascontiguousarray(d[k][index]),dtype=dtype,device='cpu')
    targets,inv=np.unique(d['target_index'][ix],return_inverse=True)
    return dict(x=tensor('x',ix),rgb_native=tensor('rgb_native',ix)/255,sam=tensor('sam',ix),context=tensor('context',ix),box_grid=tensor('box_grid',ix),
        global_input=torch.cat((tensor('global_rgb',targets)/255,tensor('global_sam',targets),tensor('global_prompt',targets)),1),
        target_inverse=torch.as_tensor(inv,dtype=torch.long),global_index64=tensor('global_index64',ix,torch.long),
        global_index8=tensor('global_index8',ix,torch.long),global_query_valid=tensor('global_query_valid',ix,torch.bool))

def cache_sealed(cache,seal_path,expected_split):
    j=json.loads(Path(seal_path).read_text())
    if j.get('complete',j.get('completed'))is not True:raise ValueError('Complete source cache seal required')
    # Root merge seals store paths relative to the V66 project directory.
    def absolute(p):return Path(p).resolve()if Path(p).is_absolute()else(ROOT/Path(p)).resolve()
    digest=sha(cache)
    hits=[b for b in j.get('outputs',[])if absolute(b['path'])==Path(cache).resolve()and b['sha256']==digest]
    if len(hits)!=1:raise ValueError('Source cache is not bound exactly once by seal')
    if hits[0].get('split')!=expected_split:raise ValueError('Sealed source cache split is not '+expected_split)
    return j

def model_sealed(checkpoint,seal_path):
    j=json.loads(Path(seal_path).read_text())
    if j.get('completed')is not True or j.get('checkpoint_sha256')!=sha(checkpoint):raise ValueError('Complete frozen source model/checkpoint seal required')
    if j.get('eval_data_used')is not False or j.get('PnP_used')is not False:raise ValueError('Source-only no-PnP model required')
    if j.get('threshold')!=.5:raise ValueError('Source-frozen threshold must remain.5')
    return j

def write_csv(path,rows,fields=None):
    with Path(path).open('w',newline='')as f:
        fields=fields or(list(rows[0])if rows else ['no_rows'])
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def threshold_notes(p,reference=None):
    st=marginal(p);conf=p.max(1);pv=st[:,:,0]
    out=dict(thresholds_fixed=dict(binary_visible=.5,joint_MAP_confident=.5),
        engineering_near_threshold_band=1e-5,near_threshold_band_is_not_a_tolerance_or_selection_gate=True,
        joint_max_within_1e_5=int((np.abs(conf-.5)<=1e-5).sum()),
        exact_pV_within_1e_5=int((np.abs(pv[:,0]-.5)<=1e-5).sum()),radius2_pV_within_1e_5=int((np.abs(pv[:,1]-.5)<=1e-5).sum()),
        CPU_inference='Torch2.5 CPU, contiguous FP32 tensors, fixed source batch order. CUDA/CPU rounding can affect values near hard thresholds; no thresholds are changed.')
    if reference is not None:
        rp=probabilities(reference);rstate=confident_states(rp)
        out['reference_probability_comparison']=dict(max_abs=float(np.max(np.abs(p-rp))),
            confident_joint_state_flips=int(np.count_nonzero(confident_states(p)!=rstate)),
            binary_pV_flips=int(np.count_nonzero((pv>=.5)!=(marginal(rp)[:,:,0]>=.5))))
    return out

def evaluate(args):
    # Actual files are opened only in this explicitly invoked sealed-source mode.
    gate=json.loads(Path(args.evaluation_gate).read_text())
    if gate.get('source_only')is not True or gate.get('validation_sources')!=list(SOURCES):raise ValueError('Registered source-only four-domain evaluation gate required')
    if gate.get('thresholds')!=dict(binary_visible=.5,confident_joint_MAP=.5):raise ValueError('Fixed source thresholds required')
    if Path(gate['output_directory']).resolve()!=Path(args.out).resolve()or gate['name']!=args.name:raise ValueError('Different evaluation output/name from source gate')
    if gate['evaluator']['sha256']!=sha(__file__)or Path(gate['evaluator']['path']).resolve()!=Path(__file__).resolve():raise ValueError('Evaluator differs from frozen source gate')
    for key,b in gate['inputs'].items():
        if not getattr(args,key,None)or Path(getattr(args,key)).resolve()!=Path(b['path']).resolve()or sha(b['path'])!=b['sha256']:raise ValueError('Source evaluation input differs from gate: '+key)
    for b in gate['bindings']:
        if sha(b['path'])!=b['sha256']:raise ValueError('Auxiliary source provenance binding changed')
    for key in ['train_cache','train_cache_seal','reference_output']:
        if getattr(args,key)and key not in gate['inputs']:raise ValueError('Unbound optional source input: '+key)
    cache_sealed(args.cache,args.cache_seal,'VAL');model_sealed(args.checkpoint,args.model_seal)
    if sha(args.model_code)!=MODEL_CODE_SHA:raise ValueError('Unchanged frozen v65 model code required')
    if args.train_cache:
        if not args.train_cache_seal:raise ValueError('Optional TRAIN prior requires its source cache seal')
        cache_sealed(args.train_cache,args.train_cache_seal,'TRAIN')
    out=Path(args.out)
    if out.exists():raise FileExistsError('Preserve previous source evaluation output')
    deny=set(json.loads(Path(args.denylist).read_text())['image_sha256']);d=load_npz(args.cache);validate_cache(d,deny)
    import torch
    if not str(torch.__version__).startswith('2.5.'):raise RuntimeError('Registered source evaluator uses default Torch2.5 CPU')
    torch.set_num_threads(1);torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True)
    spec=importlib.util.spec_from_file_location('source_eval_frozen_global_head',args.model_code);module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    model,payload=module.load_model(args.checkpoint,'cpu')
    if module.parameter_count(model)!=95508:raise ValueError('Unchanged registered head architecture expected')
    out.mkdir(parents=True);started=time.monotonic()
    bindings={k:ref(getattr(args,k))for k in ['cache','cache_seal','checkpoint','model_seal','model_code','denylist']}
    bindings['evaluator']=ref(__file__)
    bindings['evaluation_gate']=ref(args.evaluation_gate)
    if args.train_cache:
        bindings['TRAIN_prior_cache']=ref(args.train_cache);bindings['TRAIN_prior_cache_seal']=ref(args.train_cache_seal)
    if args.reference_output:bindings['reference_output']=ref(args.reference_output)
    write(out/'RUN_STARTED.json',dict(bindings=bindings,batch_size=args.batch_size,runtime=dict(python=sys.executable,torch=str(torch.__version__),numpy=np.__version__,device='cpu'),source_only=True,current223_RGB_GT_scores_read=False))
    logits=[]
    with torch.inference_mode():
        for start in range(0,len(d['x']),args.batch_size):
            ix=np.arange(start,min(len(d['x']),start+args.batch_size));logits.append(model(**model_inputs(d,ix,torch)).cpu().numpy())
    z=np.concatenate(logits).astype(np.float32);p=probabilities(z);state=marginal(p);pred=confident_states(p)
    nll,per_nll=nll_summary(d,z);metrics=all_source_metrics(d,p)
    prior=train_prior(args.train_cache,deny,set(d['image_sha256'].astype(str)))if args.train_cache else None
    baseline,sampv=baselines(d,prior);paired,pairrows=pair_counterfactual(d,p)
    reference=None
    if args.reference_output:
        r=load_npz(args.reference_output,['joint_logits','sample_id']);lookup={str(s):i for i,s in enumerate(r['sample_id'])}
        if len(lookup)!=len(d['sample_id'])or set(lookup)!=set(d['sample_id'].astype(str)):raise ValueError('Reference source output identity set differs')
        reference=r['joint_logits'][[lookup[str(s)]for s in d['sample_id']]]
        bindings['reference_output']=ref(args.reference_output)
    gt=np.stack([d['dense_exact_state'][:,4,4],d['dense_radius2_state'][:,4,4]],1);valid=d['dense_valid'][:,4,4];allowed=d['dense_allowed'][:,4,4]
    np.savez_compressed(out/'SOURCE_OUTPUTS.npz',joint_logits=z,joint_probability=p,state_probability=state,visible_probability=state[:,:,0],
        full_probability=np.stack([p[:,:3].sum(1),p[:,:5].sum(1)],1),predicted_state=pred,raw_SAM_visible_probability=sampv,
        sample_id=d['sample_id'],source=d['source'],group_id=d['group_id'],global_id=d['global_id'][d['target_index']],image_sha256=d['image_sha256'],
        center_native_valid=valid,center_allowed=allowed,GT_state=gt,partial_NLL=per_nll)
    csvrows=[]
    for i in range(len(p)):
        r=dict(sample_id=str(d['sample_id'][i]),source=str(d['source'][i]),group_id=str(d['group_id'][i]),native_valid=bool(valid[i]),allowed=int(allowed[i]),
               informative=bool(valid[i]and allowed[i]!=63),partial_NLL=float(per_nll[i]),joint_max=float(p[i].max()),joint_MAP=int(p[i].argmax()))
        for k,name in enumerate(['exact','radius2']):
            r[name+'_GT']=int(gt[i,k]);r[name+'_predicted']=NAMES[pred[i,k]]
            for c,cn in enumerate(['V','E','O']):r[name+'_p'+cn]=float(state[i,k,c])
            r[name+'_rawSAM_pV']=float(sampv[i,k])
        csvrows.append(r)
    write_csv(out/'PER_QUERY.csv',csvrows);write_csv(out/'PALLOX_PAIR_COUNTERFACTUAL.csv',pairrows)
    summaryrows=[]
    for s in SOURCES:
        for support,m in metrics[s].items():
            for cn,c in m['class_metrics'].items():summaryrows.append(dict(model=args.name,source=s,support=support,GT_class=cn,
                native_valid=m['native_valid'],known_GT=m['known_GT'],UNKNOWN_GT=m['UNKNOWN_GT'],uninformative63=m['uninformative_allowed63_native_valid'],
                GT_support=c['GT_support'],TP=c['TP'],predicted_on_known_GT=c['predicted_on_known_GT'],conditional_precision=c['precision'],recall=c['recall'],
                abstention_rate_all=m['abstention']['rate_all_native_valid'],binary_visible_precision=m['visible_binary_pV_ge_0_5']['precision'],binary_visible_recall=m['visible_binary_pV_ge_0_5']['recall']))
    write_csv(out/'PER_SOURCE_METRICS.csv',summaryrows)
    for b in bindings.values():
        if sha(b['path'])!=b['sha256']:raise RuntimeError('Evaluation input changed during read-only evaluation')
    report=dict(schema='V66_SEALED_SOURCE_PARTIAL_VALIDATION_V1',completed=True,name=args.name,bindings=bindings,N=len(p),sources=list(SOURCES),
        fixed_selection_statistic=nll,per_source=metrics,baselines=baseline,PALLOX_pair_counterfactual=paired,
        numerical_threshold_notes=threshold_notes(p,reference),
        known_subset_precision_caveat='Every reported class/binary precision is conditional on known GT. Unknown predictions are coverage diagnostics, not negative examples or verified false positives.',
        labels_caveat='CONTROLLED new opaque-paste facts and conditional PALLOX SAM+mesh+depth teachers are different provenance tiers. No semantic corner/self GT is inferred.',
        model_input_whitelist=list(INPUT_KEYS),GT_labels_are_model_inputs=False,
        actual_model_runtime=dict(python=sys.executable,torch=str(torch.__version__),numpy=np.__version__,device='cpu',batch_size=args.batch_size),
        checkpoint_metadata={k:payload.get(k)for k in ['updates','selection_partial_NLL','threshold']},
        outputs=[ref(out/n)for n in ['SOURCE_OUTPUTS.npz','PER_QUERY.csv','PER_SOURCE_METRICS.csv','PALLOX_PAIR_COUNTERFACTUAL.csv']],
        source_only=True,current223_RGB_GT_scores_read=False,optimizer_updates=0,new_PnP_calls=0,thresholds_fitted=False,model_or_checkpoint_selected=False,seconds=time.monotonic()-started)
    if prior is not None:report['TRAIN_prior_provenance']=prior
    write(out/'SOURCE_METRICS.json',report);print(json.dumps(dict(completed=True,report=str(out/'SOURCE_METRICS.json'),selection_statistic=nll,seconds=report['seconds'])),flush=True)

def mock(mock_output):
    # Generated analytical arrays only. No source cache/checkpoint is opened.
    p=np.array([[.1,.2,.3,.1,.1,.2],[.7,.1,.05,.05,.05,.05]],np.float64)
    assert np.allclose(partial_nll(np.log(p),np.array([6,63],np.uint8)),[np.log(2),0],atol=1e-14)
    assert np.array_equal(partial_nll(np.log(p),np.full(2,63,np.uint8)),np.zeros(2))
    m=support_metrics(np.array([-1,0,1,2]),np.array([1,0,1,2]),np.array([.9,.9,.1,.1]),np.ones(4,bool),np.array([63,1,4,32],np.uint8))
    assert m['known_GT']==3 and m['UNKNOWN_GT']==1 and m['predictions_on_UNKNOWN']['E']==1
    assert m['visible_binary_pV_ge_0_5']['precision']==1 and m['class_metrics']['E']['precision']==1
    n=8;d=dict(source=np.repeat(SOURCES,2),dense_valid=np.ones((n,9,9),bool),dense_allowed=np.ones((n,9,9),np.uint8),
        dense_exact_state=np.zeros((n,9,9),np.int8),dense_radius2_state=np.zeros((n,9,9),np.int8))
    pp=np.repeat(np.array([[.8,.04,.04,.04,.04,.04]]),n,axis=0);logs=np.log(pp)
    score,_=nll_summary(d,logs);assert abs(score['common4domain_mean_informative_center_partial_NLL']+np.log(.8))<1e-14
    # One photographic conditional V/E probe at the identical query slot.
    pair=dict(source=np.array(['PALLOX','PALLOX']),sample_id=np.array(['PALLOXPAIR:mock:A:0','PALLOXPAIR:mock:B:0']),group_id=np.array(['PALLOXPAIR:mock']*2),
        global_id=np.array(['PALLOXPAIR:mock:A','PALLOXPAIR:mock:B']),target_index=np.array([0,1],np.int64),dense_valid=np.ones((2,9,9),bool),
        dense_exact_state=np.stack([np.zeros((9,9),np.int8),np.ones((9,9),np.int8)]),dense_radius2_state=np.stack([np.zeros((9,9),np.int8),np.ones((9,9),np.int8)]))
    prob=np.array([[.8,.04,.04,.04,.04,.04],[.04,.04,.8,.04,.04,.04]])
    cf,rr=pair_counterfactual(pair,prob)
    assert all(v['both_direction_accuracy']==1 and v['both_target_states_correct_rate']==1 for v in cf.values())and len(rr)==2
    # Default Torch2.5 CPU API/whitelist smoke, generated inputs, no optimizer.
    import torch
    torch.set_num_threads(1);torch.manual_seed(20261022)
    code=ROOT/'model/query_global_amodal_head.py';assert sha(code)==MODEL_CODE_SHA
    spec=importlib.util.spec_from_file_location('source_eval_mock_head',code);module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    model=module.QueryGlobalAmodalHead().eval();assert module.parameter_count(model)==95508
    generated=dict(x=np.zeros((2,65,9,9),np.float16),rgb_native=np.zeros((2,75,9,9),np.uint8),sam=np.zeros((2,2,9,9),np.float16),context=np.zeros((2,64),np.float16),box_grid=np.zeros((2,4,9,9),np.float32),
        global_rgb=np.zeros((1,3,64,64),np.uint8),global_sam=np.zeros((1,1,64,64),np.float16),global_prompt=np.zeros((1,1,64,64),bool),target_index=np.zeros(2,np.int64),
        global_index64=np.zeros((2,9,9),np.int64),global_index8=np.zeros((2,9,9),np.int64),global_query_valid=np.zeros((2,9,9),bool))
    with torch.inference_mode():a=model(**model_inputs(generated,np.arange(2),torch)).numpy();b=model(**model_inputs(generated,np.arange(2),torch)).numpy()
    assert a.shape==(2,6)and np.array_equal(a,b)and np.isfinite(a).all()
    result=dict(PASS=True,schema='PARTIAL_SOURCE_EVALUATOR_GENERATED_MOCK_V1',evaluator=ref(__file__),model_code=ref(code),
        partial_likelihood_compatible_mass_PASS=True,all63_zero_loss_PASS=True,known_only_precision_excludes_UNKNOWN_PASS=True,
        four_domain_equal_macro_NLL_PASS=True,photographic_pair_direction_and_both_correct_PASS=True,
        unchanged_head_CPU_API_and_repeat_PASS=True,model_input_truth_keys=0,actual_source_cache_checkpoint_reads=0,
        private_RGB_GT_scores_read=False,GPU_used=False,optimizer_updates=0,PnP_calls=0,
        runtime=dict(python=sys.executable,torch=str(torch.__version__),numpy=np.__version__,device='cpu'))
    path=Path(mock_output);assert not path.exists();write(path,result);print(json.dumps(result),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--mock',action='store_true')
    for key in ['cache','cache-seal','checkpoint','model-seal','model-code','out','denylist','train-cache','train-cache-seal','reference-output','evaluation-gate']:
        ap.add_argument('--'+key)
    ap.add_argument('--name',default='frozen_head');ap.add_argument('--batch-size',type=int,default=256);a=ap.parse_args()
    if a.mock:mock(ROOT/'contract/EVALUATE_PARTIAL_SOURCE_MOCK_V2.json')
    else:
        for key in ['cache','cache_seal','checkpoint','model_seal','model_code','out','denylist','evaluation_gate']:
            if not getattr(a,key):ap.error('--'+key.replace('_','-')+' is required for sealed-source evaluation')
        if a.batch_size<1:ap.error('Positive CPU batch size required')
        evaluate(a)
