"""Posthoc saved TREAT same-weight reference9-zero intervention; no training."""
import argparse,csv,hashlib,json,os,stat
from pathlib import Path
SCOPE_SHA='b8cfee05a3d3faf78747f475b250bf66850b2faa30ad0c9151b43b28dd12dd4a'
SCOPE_SIZE=4216
KEYS={'features','frozen_core_checkpoint','TREAT_checkpoint','BASE_rows','TREAT_rows','reference_inputs','TREAT_trainlog','fit_complete'}
META=('sample_index','row_id','family','role','target','available')

def guard(p,existing=True):
    p=Path(p);text=str(p).casefold()
    if not p.is_absolute()or '..'in p.parts or any(t in text for t in ('v112','cache','.pyc','current223','publicheld','oldval','oldhold','source_holdout','v177','v178','v182','private_evaluation')):raise ValueError('Forbidden before FS')
    for q in reversed((p,*p.parents)):
        try:s=os.lstat(q)
        except FileNotFoundError:
            if existing:raise
            continue
        if stat.S_ISLNK(s.st_mode):raise ValueError('Symlink ancestor/artifact')
    return p

def raw(p):
    with os.fdopen(os.open(guard(p),os.O_RDONLY|os.O_NOFOLLOW),'rb')as f:return f.read()

def binding(p):
    p=guard(p);h=hashlib.sha256();n=0
    with os.fdopen(os.open(p,os.O_RDONLY|os.O_NOFOLLOW),'rb')as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b);n+=len(b)
    return {'path':str(p),'sha256':h.hexdigest(),'size_bytes':n}

def checked(b):
    if binding(b['path'])!=b:raise ValueError('Exact SHA/size/path mismatch')
    return guard(b['path'])

def write(p,d):
    p=guard(p,False)
    with p.open('x',encoding='utf-8')as f:json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    return binding(p)

def csvrows(p):
    with guard(p).open(newline='',encoding='utf-8')as f:return list(csv.DictReader(f))

def run(entry_path):
    e=json.loads(raw(entry_path))
    if e.get('approved_posthoc_reference_zero')is not True or e.get('automatic_retry')is not False:raise ValueError('Root true entry required before payload')
    if checked(e['code'])!=guard(__file__):raise ValueError('Self CODE pin')
    checked(e['scope'])
    if e['scope']['sha256']!=SCOPE_SHA or e['scope']['size_bytes']!=SCOPE_SIZE:raise ValueError('Prospective posthoc scope pin')
    scope=json.loads(raw(e['scope']['path']))
    if e['inputs']!=scope['inputs']or set(e['inputs'])!=KEYS:raise ValueError('Exact closed eight inputs')
    paths={k:checked(b)for k,b in e['inputs'].items()};out=guard(e['output_directory'],False)
    if out.exists():raise ValueError('Fresh output only, no retry')
    out.mkdir(parents=True)
    try:
        import numpy as np
        import torch
        from torch.nn import functional as F
        torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
        complete=json.loads(raw(paths['fit_complete']))
        if complete.get('completed')is not True or complete.get('PASS')is not True or complete['optimizer_updates']!={'CONTROL_NO_REFERENCE':1000,'TREAT_REFERENCE':1000}or complete['new_DEV_core_reference_or_head_outputs_before_bothfinalfreeze']!=0:raise ValueError('Both final frozen SOURCE fit required')
        for k,b in [('TREAT_checkpoint',complete['checkpoints']['TREAT_REFERENCE']),('frozen_core_checkpoint',complete['frozen_core_checkpoint']),('BASE_rows',complete['BASE_FROZEN_V239_CSV']),('TREAT_rows',complete['outputs']['TREAT_REFERENCE']),('reference_inputs',complete['reference_inputs']),('TREAT_trainlog',complete['logs']['TREAT_REFERENCE'])]:
            if any(e['inputs'][k][t]!=b[t]for t in ('sha256','size_bytes')):raise ValueError('Fit artifact binding')
        core=torch.load(paths['frozen_core_checkpoint'],weights_only=True,map_location='cpu');treat=torch.load(paths['TREAT_checkpoint'],weights_only=True,map_location='cpu')
        if core['arm']!='TREAT_CANONICAL_CONTEXT'or core['updates']!=1000 or treat['arm']!='TREAT_REFERENCE'or treat['updates']!=1000 or treat['core_optimizer_updates']!=0:raise ValueError('Frozen checkpoint identity')
        s=core['head_state_dict'];w=s['base.head.0.weight'];b=s['base.head.0.bias'];r=treat['residual_state_dict'];del core,treat,s
        if w.shape!=(64,1186)or b.shape!=(64,)or set(r)!={'net.0.weight','net.0.bias','net.2.weight','net.2.bias'}:raise ValueError('Exact hidden/residual schema')
        if any(t.dtype!=torch.float32 or not torch.isfinite(t).all()for t in [w,b,*r.values()]):raise ValueError('Finite FP32 tensors')
        x=np.load(paths['features'],mmap_mode='r',allow_pickle=False);ref=np.load(paths['reference_inputs'],mmap_mode='r',allow_pickle=False)
        if x.dtype!=np.float32 or x.shape!=(7617,1186)or ref.dtype!=np.float32 or ref.shape!=(7617,9):raise ValueError('Original fixed feature/reference shape')
        base=csvrows(paths['BASE_rows']);saved=csvrows(paths['TREAT_rows'])
        if len(base)!=7617 or len(saved)!=7617 or any(any(a[k]!=z[k]for k in META)or int(a['sample_index'])!=i for i,(a,z)in enumerate(zip(base,saved))):raise ValueError('All7617 exact same row/family/role/target/availability')
        family=np.asarray([int(a['family'])for a in base]);target=np.asarray([int(a['target'])for a in base]);available=np.asarray([a['available']=='True'for a in base]);role=np.asarray([a['role']for a in base])
        if not np.isin(family,[0,1]).all()or not np.isin(target,[0,1]).all()or not np.isin(role,['FIT','DEV']).all()or any(a['available']not in ('True','False')for a in base):raise ValueError('Original SOURCE metadata')
        baseline=np.asarray([[float(a['logit0']),float(a['logit1'])]for a in base],np.float32);expected=np.asarray([[float(a['logit0']),float(a['logit1'])]for a in saved],np.float32)
        actual=np.zeros_like(baseline);zero=np.zeros_like(baseline)
        def residual(h,q):return F.linear(F.relu(F.linear(torch.cat((h,q),1),r['net.0.weight'],r['net.0.bias'])),r['net.2.weight'],r['net.2.bias'])
        with torch.no_grad():
            for start in range(0,7617,64):
                ids=np.arange(start,min(start+64,7617));h=F.relu(F.linear(torch.as_tensor(np.array(x[ids],copy=True)),w,b));q=torch.as_tensor(np.array(ref[ids],copy=True));z=torch.as_tensor(baseline[ids])
                actual[ids]=(z+residual(h,q)).numpy();zero[ids]=(z+residual(h,torch.zeros_like(q))).numpy()
        if not np.isfinite(actual).all()or not np.isfinite(zero).all()or not np.allclose(actual,expected,atol=1e-5,rtol=1e-5):raise ValueError('Primary TREAT reconstruction fixed tol1e-5')
        idx=np.arange(7617);ae=actual[idx,family];ze=zero[idx,family];be=baseline[idx,family]
        pred=lambda z:np.where(available,np.where(z>=0,'E','D'),'U')
        ap,zp,bp=pred(ae),pred(ze),pred(be)
        if ap.tolist()!=[a['prediction']for a in saved]:raise ValueError('Primary prediction reconstruction exact')
        delta=ae-ze;aligned=(2*target-1)*delta
        def stats(a):
            a=np.asarray(a,np.float64)
            return {'N':len(a),'mean':float(a.mean()),'std':float(a.std()),'min':float(a.min()),'median':float(np.median(a)),'p95':float(np.quantile(a,.95)),'max':float(a.max())}if len(a)else {'N':0,'mean':None,'std':None,'min':None,'median':None,'p95':None,'max':None}
        def cm(mask,p):return [[int(np.count_nonzero(mask&(target==t)&(p==s)))for s in ('D','E','U')]for t in (0,1)]
        groups=[]
        for f in (0,1):
            for role_name in ('FIT','DEV'):
                m=(family==f)&(role==role_name);ac=ap==np.where(target,'E','D');zc=zp==np.where(target,'E','D')
                groups.append({'family':f,'role':role_name,'N':int(m.sum()),'BASE_CM_truthDE_predDEU':cm(m,bp),'TREAT_actual_CM_truthDE_predDEU':cm(m,ap),'same_weights_refzero_CM_truthDE_predDEU':cm(m,zp),'decision_flips':int(np.count_nonzero(m&(ap!=zp))),'actual_correct_zero_wrong':int(np.count_nonzero(m&ac&~zc)),'actual_wrong_zero_correct':int(np.count_nonzero(m&~ac&zc)),'reference_contribution_abs':stats(np.abs(delta[m])),'reference_contribution_signed':stats(delta[m]),'target_aligned_contribution':stats(aligned[m]),'by_target':{str(t):{'N':int((m&(target==t)).sum()),'contribution_abs':stats(np.abs(delta[m&(target==t)])),'target_aligned':stats(aligned[m&(target==t)])}for t in (0,1)}})
        logs=[json.loads(line)for line in raw(paths['TREAT_trainlog']).decode().splitlines()if line]
        if len(logs)!=1000 or [a['update']for a in logs]!=list(range(1,1001))or not all(a['all_fit_available_known']is True and a['core_optimizer_updates']==0 for a in logs):raise ValueError('Original1000 update saved log identity')
        losses={}
        for key in ('family0_BCE','family1_BCE','total_loss'):
            a=np.asarray([z[key]for z in logs],np.float64)
            if not np.isfinite(a).all():raise ValueError('Saved loss finite')
            losses[key]={'first':float(a[0]),'last':float(a[-1]),'first100_mean':float(a[:100].mean()),'last100_mean':float(a[-100:].mean()),'all':stats(a)}
        rows=[]
        for i,a in enumerate(base):
            rows.append({**{k:a[k]for k in META},'BASE_E_logit':float(be[i]),'TREAT_actual_E_logit':float(ae[i]),'TREAT_referencezero_E_logit':float(ze[i]),'actual_residual_E_logit':float(ae[i]-be[i]),'referencezero_residual_E_logit':float(ze[i]-be[i]),'reference_contribution_actual_minus_zero':float(delta[i]),'contribution_abs':float(abs(delta[i])),'target_aligned_contribution':float(aligned[i]),'BASE_prediction':bp[i],'TREAT_actual_prediction':ap[i],'TREAT_referencezero_prediction':zp[i],'decision_flip':bool(ap[i]!=zp[i])})
        with guard(out/'ROWS.csv',False).open('x',newline='',encoding='utf-8')as dest:
            writer=csv.DictWriter(dest,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        summary={'completed':True,'PASS':True,'posthoc':True,'rows':7617,'TREAT_reconstruction_max_abs':float(np.max(np.abs(actual-expected))),'atol':1e-5,'rtol':1e-5,'primary_predictions_exact':7617,'groups':groups,'saved_TREAT_trainlog':losses,'intervention':'Same frozen weights, same hidden/base, only reference9 set to0; family-selected E-logit threshold stays0. Target-aligned delta=(2*target-1)*(actual_Elogit-refzero_Elogit).','authority':'SYN original intended-native witness vs YCB associated-r AUX remain distinct; SOURCE DEV exposed/reused.','limits':'Posthoc direct fixed-model input intervention; not proof of all causal failure mechanisms/transfer. Refzero is not a new labelled E sample. No full core forward, producer import, optimizer, bank fit, new threshold/model/label/current evaluation.'}
        sb=write(out/'SUMMARY.json',summary);rb=binding(out/'ROWS.csv')
        result={'completed':True,'PASS':True,'code':binding(__file__),'entry':binding(entry_path),'scope':e['scope'],'inputs':e['inputs'],'summary':sb,'rows':rb,'fit_complete':e['inputs']['fit_complete'],'new_optimizer_updates':0,'full_core_forward':0,'encoder_RGB_GT_current_held_or_PnP':0,'posthoc_model_choice':False}
        write(out/'COMPLETE.json',result);print(json.dumps(result,ensure_ascii=False))
    except BaseException as exc:
        write(out/'FAILURE.json',{'completed':False,'PASS':False,'type':type(exc).__name__,'error':str(exc),'automatic_retry':False});raise

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--entry',required=True);a=p.parse_args();run(a.entry)
