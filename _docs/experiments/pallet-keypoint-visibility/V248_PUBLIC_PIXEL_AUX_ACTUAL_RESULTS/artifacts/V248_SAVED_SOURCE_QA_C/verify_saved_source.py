"""Saved V248 arithmetic; --saved-dir uses stdlib only and never follows JSON pointers.
--complete additionally verifies explicitly admitted logs/nonpixel AUX samples; no PT/features/RGB.
"""
import argparse,csv,hashlib,io,json,math,os,stat,traceback
from pathlib import Path
from collections import Counter

C='CONTROL_NO_PUBLIC_AUX';T='TREAT_PUBLIC_PIXEL_AUX';ARMS=(C,T)
DRAW_SHA='20a1fc74ea15a04a16fe29f8e5b9a114b4629677710ed43c1628b2a1ac00b734'
AUX_KEYS={'target','fit','available','group_index','row_id','query_xy','L','pair_id','view','source_role','original_source_split'}

def need(ok,msg):
    if not ok:raise ValueError(msg)

def raw(path):
    p=Path(path);need(p.is_absolute() and '..' not in p.parts,'absolute leaf')
    need(not any(x in str(p).casefold() for x in ('v112','__pycache__','.pyc','current223','publicheld','source_holdout','private_evaluation')),'lexical before FS')
    for q in reversed(p.parents):need(stat.S_ISDIR(os.lstat(q).st_mode),'ancestor symlink')
    need(stat.S_ISREG(os.lstat(p).st_mode),'leaf symlink/nonregular')
    with os.fdopen(os.open(p,os.O_RDONLY|os.O_NOFOLLOW),'rb') as f:return f.read()

def close(a,b):
    if isinstance(a,dict):need(set(a)==set(b),'summary keys');[close(v,b[k]) for k,v in a.items()]
    elif isinstance(a,list):need(len(a)==len(b),'summary length');[close(x,y) for x,y in zip(a,b)]
    elif isinstance(a,float):need(isinstance(b,(int,float)) and math.isfinite(b) and math.isclose(a,b,rel_tol=0,abs_tol=1e-11),'arithmetic float mismatch')
    else:need(type(a) is type(b) and a==b,'arithmetic scalar mismatch')

def rows(data,aux=False):
    rr=list(csv.DictReader(io.StringIO(data.decode(),newline='')));out=[]
    for i,r in enumerate(rr):
        idx='aux_index' if aux else 'sample_index';need(int(r[idx])==i and r['available'] in ('True','False'),'ordered index/bool')
        z=dict(r);z[idx]=i;z['available']=r['available']=='True';z['target']=int(r['target']);need(z['target'] in (0,1),'binary target')
        if aux:
            need(r['role'] in ('AUX_FIT','AUX_DEV') and r['view'] in ('A','B') and r['original_source_split']=='TRAIN_LAB','AUX namespace')
            keys=('logit',);value_key='logit';negative='V'
        else:
            z['family']=int(r['family']);need(z['family'] in (0,1) and r['role'] in ('FIT','DEV'),'main family/role')
            keys=('logit0','logit1','selected_family_logit');value_key='selected_family_logit';negative='D'
        for k in keys:z[k]=None if r[k]=='' else float(r[k])
        if z['available']:
            need(all(z[k] is not None and math.isfinite(z[k]) for k in keys),'finite available logits')
            if not aux:need(z[value_key]==z['logit'+str(z['family'])],'selected family scalar')
            need(r['prediction']==('E' if z[value_key]>=0 else negative),'fixed zero threshold')
        else:need(all(z[k] is None for k in keys) and r['prediction']=='U','unavailable retained U')
        out.append(z)
    need(len({x['row_id'] for x in out})==len(out),'unique rows');return out

def cell(rr,aux=False):
    labels=('V','E','U') if aux else ('D','E','U');cm=[[sum(r['target']==y and r['prediction']==p for r in rr) for p in labels] for y in (0,1)]
    if aux:item={'rows':len(rr),'CM_V_E_by_V_E_U':cm,'unsupported_U':sum(not r['available'] for r in rr),'native_gold':False}
    else:item={'rows':len(rr),'known_DE':len(rr),'CM_D_E_by_D_E_U':cm,'target_UNKNOWN':0,'original_input_U':sum(not r['available'] for r in rr)}
    for j,name in enumerate(labels[:2]):
        n=sum(cm[j]);p=cm[0][j]+cm[1][j];tp=cm[j][j]
        item[name]={'truth':n,'predicted':p,'precision':tp/p if p else None,'recall':tp/n if n else None}
    return item

def reduce(rr,aux=False):
    if aux:return {role:cell([x for x in rr if x['role']==role],True) for role in ('AUX_FIT','AUX_DEV')}
    return {f'family{f}_{role}':cell([x for x in rr if (x['family'],x['role'])==(f,role)]) for f in (0,1) for role in ('FIT','DEV')}

def paired(a,b,aux=False):
    fields=('row_id','role','available','target')+(('pair_id','view','original_source_split') if aux else ('family',))
    need(len(a)==len(b) and all(all(x[k]==y[k] for k in fields) for x,y in zip(a,b)),'paired row identity')
    groups=Counter();result={}
    for x,y in zip(a,b):
        group=x['role'] if aux else 'family'+str(x['family'])+'_'+x['role'];truth=('V' if aux else 'D') if x['target']==0 else 'E'
        groups[group]+=1;d=result.setdefault(group,{'rows':0,'recovered':0,'damaged':0,'prediction_flips':0})
        d['rows']+=1;d['recovered']+=int(x['prediction']!=truth and y['prediction']==truth);d['damaged']+=int(x['prediction']==truth and y['prediction']!=truth);d['prediction_flips']+=int(x['prediction']!=y['prediction'])
    for d in result.values():d['net']=d['recovered']-d['damaged']
    return result

def gates(base,control,treat,aux):
    old=[base[f'family{f}_DEV'][k]['recall'] for f in (0,1) for k in ('D','E')];new=[treat[f'family{f}_DEV'][k]['recall'] for f in (0,1) for k in ('D','E')]
    need(all(v is not None for v in old+new),'fixed main denominator')
    gain=100*sum((sum(new[2*f:2*f+2])-sum(old[2*f:2*f+2]))/2 for f in (0,1))/2;drop=100*max(x-y for x,y in zip(old,new))
    legacy={'PASS':gain>=3 and drop<=2 and new[3]>=.5 and min(new[:2])>=.9,'mean_family_DEV_balanced_recall_gain_pp':gain,'max_family_class_recall_drop_pp':drop,'YCB_DEV_E_recall':new[3],'SYN_DEV_D_recall':new[0],'SYN_DEV_E_recall':new[1],'fixed_min_gain_pp':3,'fixed_max_class_drop_pp':2,'fixed_YCB_DEV_E_floor':.5,'fixed_SYN_DEV_D_E_floor':.9,'current_admission':False,'automatic_retry':False,'new_current_inference':0}
    m=100*(sum(new)/4-sum(control[f'family{f}_DEV'][k]['recall'] for f in (0,1) for k in ('D','E'))/4);v=aux['AUX_DEV']['V']['recall'];e=aux['AUX_DEV']['E']['recall'];matched=m>=1.;public=v is not None and e is not None and min(v,e)>=.8;passed=bool(legacy['PASS'] and matched and public)
    return {'PASS':passed,'legacy_SOURCE_vs_V239':legacy,'matched_treatment_vs_control_gain_pp':m,'matched_min_gain_pp':1.,'matched_PASS':matched,'AUX_DEV_V_recall':v,'AUX_DEV_E_recall':e,'AUX_DEV_min_each':.8,'AUX_DEV_PASS':public,'empty_class_fill':False,'current_admission':False,'separate_current_evaluation_eligible':passed,'automatic_current_execution':False}

def arithmetic(main,aux,summary):
    need(all(len(main[a])==7617 and len(aux[a])==1568 for a in ARMS),'separate7617/1568 census')
    mm={a:reduce(main[a]) for a in ARMS};aa={a:reduce(aux[a],True) for a in ARMS}
    close(mm,summary['arms']);close(aa,summary['public_AUX'])
    base=summary['baseline_V239']
    for d in base.values():
        cm=d['CM_D_E_by_D_E_U'];need(d['rows']==sum(map(sum,cm)),'saved baseline CM totals')
        for j,name in enumerate(('D','E')):
            n=sum(cm[j]);p=cm[0][j]+cm[1][j];close({'truth':n,'predicted':p,'precision':cm[j][j]/p if p else None,'recall':cm[j][j]/n if n else None},d[name])
    gate=gates(base,mm[C],mm[T],aa[T]);close(gate,summary['SOURCE_gate'])
    need(summary['native_and_conditional_pixel_referents_separate'] is True and summary['SOURCE_DEV_and_internal_PUBLIC_TRAIN_AUX_DEV_exposed_not_blind'] is True and summary['new_current_payload_or_inference']==0 and summary['current_admission'] is False,'authority')
    return {'main':mm,'public_AUX':aa,'paired_main':paired(main[C],main[T]),'paired_AUX':paired(aux[C],aux[T],True),'SOURCE_gate':gate,'baseline_limit':'Legacy V239 baseline CM is from the admitted saved summary; old baseline CSV and main sample arrays were not reopened.'}

def actual(complete_path,sha):
    import numpy as np
    b=raw(complete_path);need(hashlib.sha256(b).hexdigest()==sha,'complete pin');c=json.loads(b);need(c['completed'] is True and c['PASS'] is True,'fit complete')
    admitted=[*c['outputs'].values(),*c['aux_outputs'].values(),*c['logs'].values(),c['summary'],c['freeze'],c['global_feature_QA'],c['gradient_QA'],c['draw_ledger'],c['aux_feature_complete'],c['aux_samples_npz'],c['pair_usage']]
    white={(x['path'],x['sha256']) for x in admitted};accessed=[]
    def load(x):
        need((x['path'],x['sha256']) in white,'not admitted saved artifact');bb=raw(x['path']);need(hashlib.sha256(bb).hexdigest()==x['sha256'] and len(bb)==x['size_bytes'],'saved SHA/size');accessed.append(x);return bb
    def js(x):return json.loads(load(x))
    main={a:rows(load(c['outputs'][a])) for a in ARMS};aux={a:rows(load(c['aux_outputs'][a]),True) for a in ARMS};summary=js(c['summary']);result=arithmetic(main,aux,summary)
    freeze=js(c['freeze']);g=js(c['global_feature_QA']);gradient=js(c['gradient_QA']);packet=js(c['aux_feature_complete']);role=js(c['pair_usage'])
    for d in (freeze,g,gradient,packet):need(d['completed'] is True and d['PASS'] is True,'saved QA complete')
    need(c['freeze']['sha256']=='7f8f26131a01f534404defae4e44bce2e769738ad1187f704e8431a9b21c182d','frozen barrier')
    for k,ck in [('checkpoints','checkpoints'),('logs','logs'),('main_feature_QA','global_feature_QA'),('gradient_QA','gradient_QA'),('draw_ledger','draw_ledger')]:need(freeze[k]==c[ck],'freeze binding '+k)
    need(freeze['new_original_DEV_or_AUX_DEV_logits_before_bothfinalfreeze']==freeze['baseline_CSV_interpretation_before_freeze']==c['new_DEV_logits_before_bothfinalfreeze']==0,'DEV/baseline barrier')
    need(freeze['matched_initial_model_sha256']==c['matched_initial_model_sha256']==gradient['state_before_after_sha256'],'matched/probe state identity')
    need(gradient['AUX_only'] is True and gradient['optimizer_updates']==gradient['shared_condition32_gradient_nonzeros']==gradient['old_output_or_relational_gradient_nonzeros']==0 and all(gradient[k] is True for k in ('shared_visual_weight_nonzero','shared_bias_nonzero','public_scalar_gradients_nonzero')),'recorded gradient isolation')
    need((g['main_all_rows'],g['known'],g['masked_S_U'],g['FIT'],g['DEV'])==(15040,7617,7423,5876,1741) and g['before_optimizer'] is True and g['new_DEV_logits']==0,'global population barrier')
    need(g['main']['active_bytes_sha256']==g['main']['node_bytes_sha256'] and g['main']['tolerance_used']==g['main']['features_overwritten']==g['main']['source_group_FIT_DEV_overlap']==0,'recorded byte-sharing QA')
    need(packet['samples']==c['aux_samples_npz'] and packet['base_state_sha256_before']==packet['base_state_sha256_after']=='bb4c7a80aa873bf3d34c3c1d44a28cee331503d7ccf52aba85f2d23edb65c3c3' and packet['encoder_state_unchanged'] is True and packet['teacher_fields_as_forward_features'] is False,'recorded frozen producer metadata')
    with np.load(io.BytesIO(load(c['aux_samples_npz'])),allow_pickle=False) as z:
        need(set(z.files)==AUX_KEYS,'nonpixel AUX11 keys');ad={k:z[k] for k in AUX_KEYS}
    n=len(ad['target']);need(n==1568 and ad['target'].dtype==np.int8 and ad['fit'].dtype==ad['available'].dtype==bool and ad['query_xy'].dtype==ad['L'].dtype==np.float64 and ad['group_index'].dtype==np.int64,'nonpixel AUX dtypes/count')
    need(ad['query_xy'].shape==(n,2) and np.isfinite(ad['query_xy']).all() and (ad['L']==64).all() and np.isin(ad['target'],[0,1]).all() and ad['available'].all(),'immutable prospective AUX geometry/support')
    ids=[r['pair_id'] for r in role['roles']];need(len(set(ids))==22,'22 pair roles');dev=set(sorted(ids,key=lambda p:hashlib.sha256(('V248:AUX_PAIR_USAGE:17:'+p).encode()).hexdigest())[:5]);groups={}
    for i,r in enumerate(aux[C]):
        need(r['row_id']==str(ad['row_id'][i]) and r['pair_id']==str(ad['pair_id'][i]) and r['view']==str(ad['view'][i]) and r['target']==int(ad['target'][i]) and r['available']==bool(ad['available'][i]) and r['role']==str(ad['source_role'][i])==('AUX_FIT' if ad['fit'][i] else 'AUX_DEV') and str(ad['original_source_split'][i])=='TRAIN_LAB','all AUX scalar/NPZ identities')
        need(r['pair_id'] in ids and bool(ad['fit'][i])==(r['pair_id'] not in dev),'pair usage siblings')
        group=int(ad['group_index'][i]);need(0<=group<44 and (group not in groups or groups[group]==(r['pair_id'],r['view'])),'pair/view group');groups[group]=(r['pair_id'],r['view'])
    need(Counter('AUX_FIT' if x else 'AUX_DEV' for x in ad['fit'])=={'AUX_FIT':1120,'AUX_DEV':448},'AUX role counts')
    rng=np.random.Generator(np.random.PCG64(17));pools=[[r['sample_index'] for r in main[C] if r['role']=='FIT' and r['available'] and (r['family'],r['target'])==(f,y)] for f in (0,1) for y in (0,1)]
    draws=np.stack([np.concatenate([rng.choice(p,8,replace=True) for p in pools]) for _ in range(1000)]).astype('<i8')
    arng=np.random.Generator(np.random.PCG64(17));apools=[np.flatnonzero(ad['fit']&ad['available']&(ad['target']==y)) for y in (0,1)];adraws=np.stack([np.concatenate([arng.choice(p,16,replace=True) for p in apools]) for _ in range(1000)]).astype('<i8')
    need(hashlib.sha256(draws.tobytes()).hexdigest()==c['main_draws_sha256']==DRAW_SHA and hashlib.sha256(adraws.tobytes()).hexdigest()==c['aux_draws_sha256'],'independent PCG17 ledger SHA')
    ledger=[json.loads(x) for x in load(c['draw_ledger']).splitlines()];logs={a:[json.loads(x) for x in load(c['logs'][a]).splitlines()] for a in ARMS};need(len(ledger)==len(logs[C])==len(logs[T])==1000,'all1000 records')
    for i,d in enumerate(ledger):
        need(d['update']==i+1 and d['main_indices']==draws[i].tolist() and d['aux_indices']==adraws[i].tolist() and d['AUX_used_by_control'] is False and d['aux_source']=='SUPPORTED_AUX_FIT_ONLY','independent ledger indices')
        for arm in ARMS:
            l=logs[arm][i];need(l['update']==i+1 and l['main_indices']==d['main_indices'] and l['aux_indices']==d['aux_indices'] and l['main_draws_sha256']==DRAW_SHA and l['aux_draws_sha256']==c['aux_draws_sha256'] and l['all_main_FIT_available_known'] is True and l['all_aux_FIT_supported'] is True,'matched FIT-only logs')
            need(l['main_family_counts']==l['aux_class_counts']==[16,16] and l['AUX_forward_and_backward']==(arm==T),'logged branch/class census')
            v=[l[k] for k in ('family0_BCE','family1_BCE','main_loss','total_loss')];need(all(math.isfinite(x) and x>=0 for x in v) and abs(l['main_loss']-.5*(v[0]+v[1]))<1e-6,'finite FP32 loss bookkeeping')
            if arm==C:need(l['aux_BCE'] is None and l['aux_weight']==0 and l['total_loss']==l['main_loss'],'control no AUX loss')
            else:need(math.isfinite(l['aux_BCE']) and l['aux_weight']==.25 and abs(l['total_loss']-l['main_loss']-.25*l['aux_BCE'])<1e-6,'fixed treatment AUX loss')
    need(c['new_optimizer_updates']==freeze['new_updates']=={a:1000 for a in ARMS} and c['total_new_optimizer_updates']==2000 and c['public_optimizer_updates']=={C:0,T:1000} and c['current_admission'] is False,'recorded update/admission metadata')
    code_bytes=raw(Path(__file__).absolute())
    result.update(completed=True,PASS=True,fit_complete={'path':str(complete_path),'sha256':sha,'size_bytes':len(b)},checker_code={'path':str(Path(__file__).absolute()),'sha256':hashlib.sha256(code_bytes).hexdigest(),'size_bytes':len(code_bytes)},admitted_inputs=accessed,logs_and_independent_PCG17_draws_all1000_PASS=True,nonpixel_AUX_rows_joined=1568,gradient_and_global_QA_metadata_consistent=True,raw_main_arrays_PT_weights_RGB_teacher_current_held_access=0,model_selection=False,current_admission=False,limits='No CP/optimizer tensors, encoder/features, original main sample arrays or teacher pixels were replayed. Gradient/bytefreeze claims are bound saved QA metadata plus prior CODE QA. Baseline counts provenance is saved summary only. No native or blind-real success claim.')
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True);g.add_argument('--saved-dir');g.add_argument('--complete');p.add_argument('--sha256');p.add_argument('--out');a=p.parse_args()
    out=None
    try:
        if a.out:out=Path(a.out).absolute();out.mkdir()
        if a.saved_dir:
            d=Path(a.saved_dir).absolute();main={arm:rows(raw(d/(arm+'_ALL_SOURCE_ROWS.csv'))) for arm in ARMS};aux={arm:rows(raw(d/(arm+'_ALL_PUBLIC_AUX_ROWS.csv')),True) for arm in ARMS};r=arithmetic(main,aux,json.loads(raw(d/'SOURCE_RESULTS.json')));r.update(completed=True,PASS=True,mode='stdlib_saved_CSV_arithmetic_only_no_pointer_follow')
        else:need(a.sha256 is not None,'exact complete SHA required');r=actual(Path(a.complete).absolute(),a.sha256)
        if out:
            f=out/'REVIEW.json'
            with f.open('x',encoding='utf-8') as w:json.dump(r,w,ensure_ascii=False,indent=2,allow_nan=False);w.write('\n')
            bb=raw(f);print(json.dumps({'completed':True,'PASS':True,'REVIEW':{'path':str(f),'sha256':hashlib.sha256(bb).hexdigest(),'size_bytes':len(bb)}}))
        else:print(json.dumps(r,ensure_ascii=False,allow_nan=False))
    except BaseException:
        if out:
            with (out/'FAILED_EXECUTION.json').open('x') as w:json.dump({'completed':False,'PASS':False,'automatic_retry':False,'traceback':traceback.format_exc()},w,indent=2)
        raise
