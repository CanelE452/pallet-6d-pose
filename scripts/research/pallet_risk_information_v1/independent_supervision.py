"""Independent synthetic-only supervision, loss and gradient audit.

No production relation/error/loss implementation is used to calculate the expected
answer. Production functions are invoked only on the comparison side. No model,
optimizer, real reference, candidate generation or pose solve is used.
"""
from __future__ import annotations
import argparse,gzip,hashlib,json,math,resource,time,traceback
from collections import Counter
from pathlib import Path
import numpy as np


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb')as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def rows(path):
    with gzip.open(path,'rt')as f:
        for line in f:
            if line.strip():yield json.loads(line)
def json_scalar(v):
    if isinstance(v,np.generic):return v.item()
    raise TypeError('Unsupported JSON type: '+type(v).__name__)
def write(path,value):
    with Path(path).open('x')as f:json.dump(value,f,sort_keys=True,indent=2,allow_nan=False,default=json_scalar);f.write('\n')


def independent_errors(pose,target,stable=False):
    if not isinstance(pose,dict)or pose.get('available')is not True:return[math.nan,math.nan]
    rotation=np.asarray(pose['R_physical'],np.float64);center=np.asarray(pose['centroid'],np.float64)
    truth=np.asarray(target['R_fixed'],np.float64);origin=np.asarray(target['t'],np.float64)
    if rotation.shape!=(3,3)or truth.shape!=(3,3)or center.shape!=(3,)or origin.shape!=(3,):return[math.nan,math.nan]
    if not all(np.isfinite(v).all()for v in(rotation,center,truth,origin)):return[math.nan,math.nan]
    order=int(target['symmetry_order'])
    if order not in(1,2):raise ValueError('Original C1/C2 only')
    delta=center-origin
    distance=(math.sqrt(math.fsum(float(v)*float(v)for v in delta))if stable else float(np.linalg.norm(delta)))*100.
    angles=[]
    for phase in range(order):
        if stable:
            symmetry=np.eye(3)if phase==0 else np.diag([-1.,1.,-1.])
        else:
            theta=2.*np.pi*phase/order
            symmetry=np.array([[math.cos(theta),0.,math.sin(theta)],[0.,1.,0.],[-math.sin(theta),0.,math.cos(theta)]])
        mismatch=np.matmul(np.matmul(truth,symmetry).T,rotation)
        cosine=float(np.clip((np.trace(mismatch)-1.)*.5,-1.,1.))
        if stable:
            skew=(float(mismatch[2,1]-mismatch[1,2]),float(mismatch[0,2]-mismatch[2,0]),float(mismatch[1,0]-mismatch[0,1]))
            angle=math.atan2(.5*math.sqrt(math.fsum(v*v for v in skew)),cosine)
        else:angle=float(np.arccos(cosine))
        angles.append(math.degrees(angle)if stable else float(np.degrees(angle)))
    return[distance,min(angles)]


def preference(a,b,success_a,success_b):
    if success_a!=success_b:return -1 if success_a else 1
    if success_a:return 0
    better=all(x<=y for x,y in zip(a,b))and any(x<y for x,y in zip(a,b))
    worse=all(y<=x for x,y in zip(a,b))and any(y<x for x,y in zip(a,b))
    return -1 if better else 1 if worse else 0


def independent_supervision(raw,stable,success):
    raw=np.asarray(raw,float).reshape(-1,2);stable=np.asarray(stable,float).reshape(-1,2);success=list(map(bool,success));n=len(raw)
    valid=[[math.isfinite(float(v))and v>=0 for v in pair]for pair in raw]
    altvalid=[[math.isfinite(float(v))and v>=0 for v in pair]for pair in stable]
    altgood=[all(v)and max(pair)<=5 for pair,v in zip(stable,altvalid)]
    targets=[[math.log1p(float(v)/5.)if valid[i][axis]else 0. for axis,v in enumerate(pair)]for i,pair in enumerate(raw)]
    groups=[[],[]];ambiguous=[];counts=Counter()
    for i in range(n):
        for j in range(i+1,n):
            counts['total_pairs']+=1
            if success[i]and success[j]:counts['both_success_no_order']+=1;continue
            if not all(valid[i]+valid[j]+altvalid[i]+altvalid[j]):counts['unavailable_or_nonfinite_pair']+=1;continue
            first=preference(raw[i],raw[j],success[i],success[j]);second=preference(stable[i],stable[j],altgood[i],altgood[j])
            if first!=second:counts['NUMERICAL_ORDER_AMBIGUOUS']+=1;ambiguous.append((i,j,first,second));continue
            if first==0:counts['incomparable_or_equal_failure_pair']+=1;continue
            groups[0 if success[i]!=success[j]else 1].append((i,j)if first==-1 else(j,i))
    counts.update(dict(success_failure_pairs=len(groups[0]),failure_Pareto_pairs=len(groups[1]),usable_pairs=sum(map(len,groups)),actions=n,
        positive_action_labels=sum(success),negative_action_labels=n-sum(success),all_zero_frame=int(n>0 and not any(success)),zero_action_frame=int(n==0),
        no_order_supervision_frame=int(not any(groups)),all_zero_with_Pareto_supervision=int(n>0 and not any(success)and bool(groups[1])),
        all_zero_without_order_supervision=int(n>0 and not any(success)and not any(groups)),
        finite_T_axes=sum(v[0]for v in valid),finite_R_axes=sum(v[1]for v in valid),
        missing_or_nonfinite_T_axes=n-sum(v[0]for v in valid),missing_or_nonfinite_R_axes=n-sum(v[1]for v in valid)))
    return dict(continuous_target=np.asarray(targets,float).reshape(-1,2),continuous_validity=np.asarray(valid,bool).reshape(-1,2),
        pair_groups=tuple(np.asarray(g,np.int64).reshape(-1,2)for g in groups),numerical_ambiguous_pairs=ambiguous,counts=dict(counts))


def analytic_continuous(output,target,validity):
    output=np.asarray(output,float);target=np.asarray(target,float);valid=np.asarray(validity,bool);gradient=np.zeros_like(output)
    included=[i for i in range(len(output))if any(valid[i])]
    if not included:return None,gradient
    means=[]
    for i in included:
        axes=[j for j in range(2)if valid[i,j]];terms=[]
        for j in axes:
            d=float(output[i,j]-target[i,j]);terms.append(.5*d*d if abs(d)<1 else abs(d)-.5)
            gradient[i,j]=(d if abs(d)<1 else math.copysign(1.,d))/(len(included)*len(axes))
        means.append(math.fsum(terms)/len(axes))
    return math.fsum(means)/len(means),gradient


def analytic_order(scores,groups):
    scores=np.asarray(scores,float);gradient=np.zeros_like(scores);present=[g for g in groups if len(g)]
    if not present:return None,gradient
    losses=[]
    for group in present:
        terms=[]
        for better,worse in group:
            d=float(scores[worse]-scores[better]);terms.append(max(d,0.)+math.log1p(math.exp(-abs(d))))
            probability=1./(1.+math.exp(-d))if d>=0 else math.exp(d)/(1.+math.exp(d))
            derivative=probability/(len(present)*len(group));gradient[better]-=derivative;gradient[worse]+=derivative
        losses.append(math.fsum(terms)/len(group))
    return math.fsum(losses)/len(losses),gradient


def numerical_bound(value):
    """Predeclared float64 primitive/reduction comparison, never an accuracy gate."""
    return 64.*np.finfo(np.float64).eps*max(1.,abs(float(value)))


def run(archive,upstream,output):
    import torch
    from . import supervision as production
    from . import risk_models as losses
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    archive=Path(archive).resolve();upstream=Path(upstream).resolve();output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();cpu=time.process_time();checks=Counter();failures=[];diagnostics={k:Counter()for k in('train','validation')};totals=Counter();maxima=Counter();digests={k:hashlib.sha256()for k in('pairs','stable','targets')}
    def check(name,condition,detail=None):
        checks[name]+=1
        if not bool(condition):failures.append(dict(check=name,detail=detail))
    try:
        paths=dict(features=archive/'d1_synthetic_candidates_v1/HEAD_FEATURES.jsonl.gz',feature_seal=archive/'d1_synthetic_candidates_v1/FEATURE_SEAL.json',
            labels=archive/'d1_action_labels_v1/ACTION_TARGETS.jsonl.gz',label_seal=archive/'d1_action_labels_v1/LABEL_SEAL.json',
            targets=upstream/'synthetic_v1/SYNTH_GT_LABELS.jsonl.gz',target_seal=upstream/'synthetic_v1/PROVISION_SEAL.json',split=upstream/'synthetic_v1/DATA_SPLIT_LOCK.json')
        binding={k:dict(path=str(p),sha256=sha(p))for k,p in paths.items()};ls=read(paths['label_seal'])
        for name,key in(('features','features_sha256'),('labels','labels_sha256'),('targets','targets_sha256'),('target_seal','target_seal_sha256'),('split','split_lock_sha256')):check('input_binding/'+name,binding[name]['sha256']==ls[key])
        check('feature_seal',binding['features']['sha256']==read(paths['feature_seal'])['head_features_sha256'])
        labels={r['id']:r for r in rows(paths['labels'])};targets={r['id']:r for r in rows(paths['targets'])};split={r['id']:r['split']for r in read(paths['split'])['all_rows']}
        order_counts=Counter();frame_rows=[]
        for frame in rows(paths['features']):
            fid=frame['id'];sp=frame['split'];label=labels[fid];target=targets[fid];metrics=label['action_metrics'];totals['frames']+=1;totals[sp+'/frames']+=1
            check('exact_original_join',label['split']==target['split']==sp==split[fid])
            # P0 already independently verified canonical H1 bank/order. Join
            # these supervised records to original actual poses, preserving that
            # exact order; labels do not become prediction features.
            by={c['candidate_id']:c for c in frame['candidates']};entries=[]
            for metric in metrics:
                key=metric['candidate_id'];candidate=by.get(key)
                if candidate is None:
                    p=frame['fallback']if key=='N0'else frame['e0_regenerated']if key=='E0_REGENERATED'else None
                    if p is None:raise ValueError('Supervision pose absent from original bank')
                    candidate=dict(candidate_id=key,actual_pose=p)
                entries.append(candidate)
            raw=np.asarray([[m['translation_cm']if m['translation_cm']is not None else math.nan,m['rotation_deg']if m['rotation_deg']is not None else math.nan]for m in metrics],float).reshape(-1,2)
            success=np.asarray([m['success_5cm5deg']for m in metrics],bool)
            independent_raw=np.asarray([independent_errors(c['actual_pose'],target)for c in entries],float).reshape(-1,2)
            stable=np.asarray([independent_errors(c['actual_pose'],target,True)for c in entries],float).reshape(-1,2)
            order_counts[str(target['symmetry_order'])]+=1
            check('original_raw_error_bitwise',np.array_equal(raw,independent_raw,equal_nan=True),fid)
            check('original_success_definition',np.array_equal(success,np.isfinite(independent_raw).all(-1)&(independent_raw<=5).all(-1)),fid)
            expected=independent_supervision(raw,stable,success);actual=production.prepare(entries,label,target)
            check('stable_formula_bitwise',np.array_equal(stable,actual['stable_errors'],equal_nan=True),fid)
            for name in('continuous_validity','continuous_target'):
                if name=='continuous_target':
                    difference=float(np.max(np.abs(expected[name]-actual[name])))if len(raw)else 0.;maxima['log_target_absolute_difference']=max(maxima['log_target_absolute_difference'],difference)
                    check('log_target_primitive',np.all(np.abs(expected[name]-actual[name])<=64*np.finfo(float).eps*np.maximum(1.,np.abs(expected[name]))),fid)
                else:check('axis_validity_exact',np.array_equal(expected[name],actual[name]),fid)
            check('pair_categories_and_directions_exact',all(np.array_equal(a,b)for a,b in zip(expected['pair_groups'],actual['pair_groups'])),fid)
            check('numerical_ambiguity_exact',expected['numerical_ambiguous_pairs']==actual['numerical_ambiguous_pairs'],fid)
            check('all_diagnostic_counts_exact',expected['counts']==actual['counts'],fid);diagnostics[sp].update(expected['counts'])
            # Fixed input-only test scores, not learned outputs or fitted seeds.
            n=len(entries);scores=np.asarray([math.sin(i)+math.cos(i*.125)for i in range(n)],np.float64)
            outputs=np.asarray([[(i%7+1)*.2,(i%5+1)*.3]for i in range(n)],np.float64).reshape(-1,2)
            for kind,value in(('order',scores),('continuous',outputs)):
                tensor=torch.tensor(value,dtype=torch.float64,requires_grad=True)
                if kind=='order':
                    expected_loss,expected_grad=analytic_order(value,expected['pair_groups']);actual_loss=losses.order_frame_loss(tensor,actual['pair_groups'])
                else:
                    expected_loss,expected_grad=analytic_continuous(value,expected['continuous_target'],expected['continuous_validity']);actual_loss=losses.continuous_frame_loss(tensor,torch.tensor(actual['continuous_target'],dtype=torch.float64),torch.tensor(actual['continuous_validity'],dtype=torch.bool))
                check(kind+'_loss_contribution',(expected_loss is None)==(actual_loss is None),fid)
                if actual_loss is not None:
                    actual_loss.backward();d=abs(float(actual_loss)-expected_loss);g=float(np.max(np.abs(tensor.grad.numpy()-expected_grad)))if n else 0.
                    maxima[kind+'_loss_absolute_difference']=max(maxima[kind+'_loss_absolute_difference'],d);maxima[kind+'_gradient_absolute_difference']=max(maxima[kind+'_gradient_absolute_difference'],g)
                    check(kind+'_analytic_loss',d<=numerical_bound(expected_loss),fid)
                    check(kind+'_analytic_gradient',g<=numerical_bound(float(np.max(np.abs(expected_grad)))),fid)
                    totals[sp+'/'+kind+'_loss_frames']+=1
            train_number=totals[sp+'/frames']-1
            for seed in(20261018,20261019,20261020):
                for epoch in(0,1,49):
                    wanted=[]
                    for category,pairs in enumerate(expected['pair_groups']):
                        if sp=='train'and len(pairs)>256:
                            generator=np.random.default_rng(np.random.SeedSequence([seed,epoch,train_number,category]));index=np.sort(generator.choice(len(pairs),256,replace=False));wanted.append(pairs[index])
                        else:wanted.append(pairs)
                    sampled=production.sampled_pairs(actual,seed,epoch,train_number,training=sp=='train')
                    check('fixed_uniform_sampling_contract',all(np.array_equal(a,b)for a,b in zip(wanted,sampled)),dict(id=fid,seed=seed,epoch=epoch))
            for g in expected['pair_groups']:digests['pairs'].update(g.astype('<i8').tobytes())
            digests['stable'].update(stable.astype('<f8').tobytes());digests['targets'].update(expected['continuous_target'].astype('<f8').tobytes())
            frame_rows.append(dict(id=fid,split=sp,counts=expected['counts'],rank_loss_contributes=bool(expected['counts']['usable_pairs']),continuous_loss_contributes=bool(expected['continuous_validity'].any())))
        check('all1003_frames',totals['frames']==1003 and totals['train/frames']==768 and totals['validation/frames']==235)
        for name,p in paths.items():check('source_bytes_preserved',sha(p)==binding[name]['sha256'],name)
        write(output/'SUPERVISION_DIAGNOSTICS.json',dict(schema='INDEPENDENT_SYNTHETIC_SUPERVISION_DIAGNOSTICS_V1',counts={k:dict(v)for k,v in diagnostics.items()},loss_contribution_counts=dict(totals),symmetry_orders=dict(order_counts)))
        with gzip.open(output/'FRAME_DIAGNOSTICS.jsonl.gz','xt')as f:
            for row in frame_rows:f.write(json.dumps(row,sort_keys=True,allow_nan=False,default=json_scalar)+'\n')
        receipt=dict(schema='RISK_INDEPENDENT_SUPERVISION_CHECK_V1',status='PASS'if not failures else'FAIL',checks=sum(checks.values()),checks_by_kind=dict(checks),failure_count=len(failures),failures=failures,
            input_bindings=binding,diagnostics_sha256=sha(output/'SUPERVISION_DIAGNOSTICS.json'),frame_diagnostics_sha256=sha(output/'FRAME_DIAGNOSTICS.jsonl.gz'),max_numerical_differences=dict(maxima),
            stream_sha256={k:v.hexdigest()for k,v in digests.items()},source_sha256=sha(__file__),production_source_sha256={m.__name__:sha(m.__file__)for m in(production,losses)},
            formula_contract='Original raw norm/acos/sin(pi) success retained; separate fsum/atan2 exact C1/C2 is used only to exclude ambiguous order pairs. Numerical comparison bound fixed before execution:64*float64eps*max(1,abs(value)).',
            independent_scope='Independent formula, pair relation, masks, counts, analytic SmoothL1/RankNet and gradients. Production functions are comparison targets only. Synthetic references decoded; real references never opened.',
            new_optimizer_fits=0,new_model_forward=0,new_PnP_solves=0,real_reference_opens=0,
            cost=dict(wall_seconds=time.monotonic()-started,CPU_seconds=time.process_time()-cpu,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,threads=1,device='CPU'))
        write(output/'VERIFICATION.json',receipt);return receipt
    except BaseException as exc:
        write(output/'FAILED_ATTEMPT.json',dict(error=repr(exc),traceback=traceback.format_exc(),checks=dict(checks),failures=failures,wall_seconds=time.monotonic()-started));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True);p.add_argument('--upstream',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=run(a.archive,a.upstream,a.output);print(json.dumps({k:r[k]for k in('status','checks','failure_count','max_numerical_differences','cost')}));raise SystemExit(0 if r['status']=='PASS'else 1)
