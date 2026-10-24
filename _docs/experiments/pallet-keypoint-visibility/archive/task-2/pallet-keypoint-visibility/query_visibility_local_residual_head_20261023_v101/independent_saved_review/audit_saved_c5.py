"""Independent saved-only C5 audit. NumPy metrics; torch only for CPU RNG replay.

Never imports the model/trainer, opens feature/pixel arrays or deserializes weights.
"""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import hashlib,json,math,traceback
import numpy as np
H=Path(__file__).resolve().parent;V=H.parent;RUN=V/'runs/C5_FROZEN_C1_CENTRAL_RGB_SOURCE_V1'
NOLD=31680;NNEW=2304;N=NOLD+NNEW
NAMES=('old_LOCO_BEFORE_D','old_PALLOX_BEFORE_D','old_SHAM_D','old_BEFORE_E','old_COVER_E','new_BEFORE_D','new_SHAM_D','new_COVER_E','new_ALL_S')
STAGES=('BEFORE','COVER','SHAM');checks=0
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def check(b):assert sha(b['path'])==b['sha256'],b['path'];return Path(b['path'])
def read(p):return json.loads(Path(p).read_text())
def rows(p):return [json.loads(x)for x in Path(p).read_text().splitlines()if x.strip()]
def write(p,x):
    with Path(p).open('x')as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def same(actual,expected,path='root'):
    global checks
    if isinstance(expected,dict):
        assert set(actual)==set(expected),(path,set(actual)^set(expected))
        for k in expected:same(actual[k],expected[k],path+'.'+str(k))
    elif isinstance(expected,list):
        assert len(actual)==len(expected),path
        for i,(a,b)in enumerate(zip(actual,expected)):same(a,b,path+f'[{i}]')
    elif isinstance(expected,float):
        assert isinstance(actual,(float,int))and math.isclose(actual,expected,rel_tol=2e-14,abs_tol=2e-14),(path,actual,expected)
        checks+=1
    else:assert actual==expected,(path,actual,expected);checks+=1
def arreq(actual,expected,name):assert np.array_equal(actual,np.asarray(expected),equal_nan=True)if np.asarray(expected).dtype.kind in 'fc'else np.array_equal(actual,np.asarray(expected)),name

def independent_metadata(b):
    om=read(check(b['old_manifest']));nm=read(check(b['new_manifest']))
    ol=rows(check(om['input_lineage']));nl=rows(check(nm['input_lineage']));og=rows(check(b['old_GT']));ng=rows(check(b['new_GT']));q=rows(check(b['new_queries']));cfg=rows(check(b['new_config']))
    assert len(ol)==len(og)==NOLD and len(nl)==NNEW and len(ng)==len(q)==len(cfg)==96
    labels=[];typed=[];source=[];split=[];original=[];variant=[];pairs_old=[];pairs_new=[];oldlabels=[];appearance=[];aspect=[];az=[];el=[]
    canon={('LOCO','TRAIN'):'TRAIN',('LOCO','VALIDATION'):'VALIDATION',('PALLOX','TRAIN_LAB'):'TRAIN',('PALLOX','VAL_HALL'):'VALIDATION'}
    for i,(r,g)in enumerate(zip(ol,og)):
        base,j=divmod(i,33);rank=None if j==0 else(j-1)//2;v='BEFORE'if j==0 else'OPAQUE_COVER'if j%2 else'CUE_PRESERVING_SHAM'
        assert r['occurrence_index']==g['occurrence_index']==i and r['base_row_index']==g['base_row_index']==r['slot_index']==g['slot_index']==base
        assert r['donor_rank']==g['donor_rank']==rank and r['variant']==g['variant']==v and r['occurrence_id']==g['occurrence_id']
        assert r['source']==ol[base*33]['source']and r['split']==ol[base*33]['split']
        assert r['query_xy']==ol[base*33]['query_xy']and r['L']==ol[base*33]['L']
        assert g['native_expected_part_state']=='UNKNOWN'and g['human_gold']is False
        y=g['candidate_aux_hidden_label_GT_only'];assert y in(-1,0,1)and g['known_candidate_GT_only']==(y>=0)
        if j==0:assert y==g['query_anchored_before_label']and g['pair_ready']is False
        elif j%2:
            following=og[i+1];assert following['pair_ready']==g['pair_ready']
            assert [y,following['candidate_aux_hidden_label_GT_only']]==([1,0]if g['pair_ready']else[-1,-1])
            if g['pair_ready']:assert labels[base*33]==0;pairs_old.append([base*33,i,i+1])
        labels.append(y);typed.append('U'if y<0 else'D'if y==0 else'E');source.append(r['source']);split.append(canon[(r['source'],r['split'])]);original.append(r['split']);variant.append(v);oldlabels.append(g['original_C1_V95_label'])
        appearance.append('NOT_APPLICABLE');aspect.append('NOT_APPLICABLE');az.append(-1);el.append(-1)
    for base in range(96):
        g=ng[base];c=cfg[base];qq=q[base]
        assert g['base_index']==c['base_index']==qq['base_index']==base and g['base_id']==c['base_id']==qq['base_id']==f'V99_BASE_{base:03d}'
        assert g['split']==c['split']==qq['split']and g['after_force_U']is False
        for si,v in enumerate(STAGES):
            for k,x in enumerate(g['stages'][v]):
                i=24*base+8*si+k;r=nl[i]
                assert r['occurrence_index']==r['input_unit_index']==i and r['base_index']==base and r['base_id']==g['base_id']and r['aux_corner_index']==k
                assert r['variant']==v and r['split']==g['split']and r['query_xy']==qq['query8_xy'][k]and r['L']==qq['L'][k]
                assert r['native8_mapping']==x['native8_mapping']=='UNKNOWN'and x['human_gold']is False and r['source_emulated_not_initializer']is True
                t=x['state'];labels.append({'D':0,'E':1,'S':1,'U':-1}[t]);typed.append(t);source.append('SYNTH_EXPECTED_CUE');split.append(g['split']);original.append(g['split']);variant.append(v)
                appearance.append(c['appearance_family']);aspect.append(c['aspect_family']);az.append(c['azimuth_index']);el.append(c['elevation_index'])
        selected=g['selected_aux_index'];pp=[NOLD+24*base+selected,NOLD+24*base+8+selected,NOLD+24*base+16+selected]
        assert [typed[j]for j in pp]==['D','E','D'];pairs_new.append(pp)
    line=ol+nl
    meta=dict(labels=np.array(labels,np.int8),typed=np.array(typed),source=np.array(source),split=np.array(split),original_split=np.array(original),variant=np.array(variant),domain=np.array(['V96_REAL']*NOLD+['V99_SYNTH']*NNEW),appearance=np.array(appearance),aspect=np.array(aspect),azimuth=np.array(az,np.int8),elevation=np.array(el,np.int8),old_pairs=np.array(pairs_old,np.int64),new_pairs=np.array(pairs_new,np.int64),lineage=line,old_original_label=np.array(oldlabels,np.int8))
    meta['known']=meta['labels']>=0
    meta['available']=np.array([r['input_facts']['operator_query_available']for r in line],bool)
    meta['units']=np.r_[[r['input_unit_index']for r in ol],np.arange(NNEW)+1348]
    meta['q']=np.array([r['query_xy']if r['query_xy']is not None else[np.nan,np.nan]for r in line],np.float64)
    meta['L']=np.array([r['L']if r['L']is not None else np.nan for r in line],np.float64)
    meta['ids']=np.array([r['occurrence_id']if 'occurrence_id'in r else r['image_id']+'::AUX'+str(r['aux_corner_index'])for r in line])
    assert len(labels)==N and len(pairs_old)==194 and len(pairs_new)==96
    assert np.bincount(meta['labels'][:NOLD:33]+1,minlength=3).tolist()==[895,61,4]
    return meta

def masks(m,s):
    old=(m['domain']=='V96_REAL')&(m['split']==s);new=(m['domain']=='V99_SYNTH')&(m['split']==s);y=m['labels'];v=m['variant'];t=m['typed']
    return dict(old_LOCO_BEFORE_D=old&(m['source']=='LOCO')&(v=='BEFORE')&(y==0),old_PALLOX_BEFORE_D=old&(m['source']=='PALLOX')&(v=='BEFORE')&(y==0),old_SHAM_D=old&(v=='CUE_PRESERVING_SHAM')&(y==0),old_BEFORE_E=old&(v=='BEFORE')&(y==1),old_COVER_E=old&(v=='OPAQUE_COVER')&(y==1),new_BEFORE_D=new&(v=='BEFORE')&(t=='D'),new_SHAM_D=new&(v=='SHAM')&(t=='D'),new_COVER_E=new&(v=='COVER')&(t=='E'),new_ALL_S=new&(t=='S'))

def counts(m):
    out={}
    for domain in('V96_REAL','V99_SYNTH'):
        out[domain]={}
        for s in('TRAIN','VALIDATION'):
            a=(m['domain']==domain)&(m['split']==s)
            out[domain][s]=dict(rows=int(a.sum()),typed_states={t:int((a&(m['typed']==t)).sum())for t in('D','E','S','U')},by_variant={v:{t:int((a&(m['variant']==v)&(m['typed']==t)).sum())for t in('D','E','S','U')}for v in np.unique(m['variant'][a])})
    out['VAL_cohorts']={k:int(v.sum())for k,v in masks(m,'VALIDATION').items()}
    for key,pkey in(('old_accepted_pairs','old_pairs'),('new_selected_pairs','new_pairs')):out[key]={s:int((m['split'][m[pkey][:,0]]==s).sum())for s in('TRAIN','VALIDATION')}
    return out

def confusion(m,p,mask,new=False):
    chosen=mask&m['known'];cm=np.zeros((2,3),np.int64)
    col=np.where(p[chosen]<0,2,p[chosen]);np.add.at(cm,(m['labels'][chosen],col),1)
    ratio=lambda a,b:float(a/b)if b else None
    r=dict(matrix=cm.tolist(),rows=['conditional_AUX_D','conditional_AUX_hidden_E_or_S'if new else'conditional_AUX_hidden_E'],columns=['D','hidden'if new else'E','U'],D_recall=ratio(cm[0,0],cm[0].sum()),D_precision=ratio(cm[0,0],cm[:,0].sum()),known_U_rate=ratio(cm[:,2].sum(),cm.sum()),all_N=int(mask.sum()),known_N=int(cm.sum()),unknown_GT_N=int((mask&~m['known']).sum()),unknown_GT_predictions={str(k):int((mask&~m['known']&(p==k)).sum())for k in(0,1,-1)})
    r['binary_hidden_recall'if new else'E_recall']=ratio(cm[1,1],cm[1].sum());r['binary_hidden_precision'if new else'E_precision']=ratio(cm[1,1],cm[:,1].sum())
    if new:r['cause_classification']=False
    return r

def pair_result(m,p,key,chosen,new=False):
    pp=m[key];pred=p[pp];hits=np.all(pred==np.array([0,1,0]),axis=1);n=int(chosen.sum())
    if new:return dict(selected_triplets=n,all3_correct=int((chosen&hits).sum()),all3_correct_rate=float(hits[chosen].mean())if n else None,before_D_correct=int((chosen&(pred[:,0]==0)).sum()),cover_E_correct=int((chosen&(pred[:,1]==1)).sum()),sham_D_correct=int((chosen&(pred[:,2]==0)).sum()),any_U=int((chosen&(pred<0).any(1)).sum()))
    ranks=(pp[:,1]%33-1)//2
    return dict(accepted_pairs=n,all3_correct=int((chosen&hits).sum()),all3_correct_rate=float((chosen&hits).sum()/n)if n else None,rank_counts={str(k):dict(N=int((chosen&(ranks==k)).sum()),all3_correct=int((chosen&(ranks==k)&hits).sum()))for k in range(16)},controlled_tested=n>0)

def recompute_metrics(m,z,a,base,delta,design):
    p=np.where(a,(z>=0).astype(np.int8),-1);bp=np.where(a,(base>=0).astype(np.int8),-1)
    result=dict(old_exact_V96={},new={},preservation={})
    for s in('TRAIN','VALIDATION'):
        om=(m['domain']=='V96_REAL')&(m['split']==s);nm=(m['domain']=='V99_SYNTH')&(m['split']==s);mm=masks(m,s)
        op=(m['split'][m['old_pairs'][:,0]]==s);npair=m['split'][m['new_pairs'][:,0]]==s
        result['old_exact_V96'][s]=dict(all_known=confusion(m,p,om),by_variant={v:confusion(m,p,om&(m['variant']==v))for v in('BEFORE','OPAQUE_COVER','CUE_PRESERVING_SHAM')},five_cohorts={k[4:]:confusion(m,p,mask)for k,mask in mm.items()if k.startswith('old_')},paired=pair_result(m,p,'old_pairs',op),paired_by_source={src:pair_result(m,p,'old_pairs',op&(m['source'][m['old_pairs'][:,0]]==src))for src in('LOCO','PALLOX')})
        nn=dict(all_known=confusion(m,p,nm,True),by_state={t:confusion(m,p,nm&(m['typed']==t),True)for t in('D','E','S','U')},by_variant={v:confusion(m,p,nm&(m['variant']==v),True)for v in STAGES},point_metrics_by_axes={},selected_pairs=pair_result(m,p,'new_pairs',npair,True),pairs_by_axes={})
        for axis in('appearance','aspect','azimuth','elevation'):
            nn['point_metrics_by_axes'][axis]={str(v):confusion(m,p,nm&(m[axis]==v),True)for v in np.unique(m[axis][nm])}
            vals=m[axis][m['new_pairs'][:,0]];nn['pairs_by_axes'][axis]={str(v):pair_result(m,p,'new_pairs',npair&(vals==v),True)for v in np.unique(vals[npair])}
        result['new'][s]=nn
    for dom in('V96_REAL','V99_SYNTH'):
        result['preservation'][dom]={}
        for s in('TRAIN','VALIDATION'):
            mask=(m['domain']==dom)&(m['split']==s)&m['known'];bc=mask&(bp==m['labels']);fc=mask&(p==m['labels'])
            result['preservation'][dom][s]=dict(known_N=int(mask.sum()),base_correct=int(bc.sum()),base_correct_still_correct=int((bc&fc).sum()),base_correct_became_wrong_or_U=int((bc&~fc).sum()),base_wrong_becomes_correct=int((mask&~bc&fc).sum()),known_mean_abs_delta=float(abs(delta[mask]).mean())if mask.any()else None,max_abs_delta=float(abs(delta[mask]).max())if mask.any()else None)
    vm=masks(m,'VALIDATION');gates={}
    specs=[('old_LOCO_before_D_recall_min','old_LOCO_BEFORE_D',0),('old_PALLOX_before_D_recall_min','old_PALLOX_BEFORE_D',0),('old_natural_before_E_recall_min','old_BEFORE_E',1),('old_sham_D_recall_min','old_SHAM_D',0),('old_cover_E_recall_min','old_COVER_E',1),('new_BEFORE_D_recall_min','new_BEFORE_D',0),('new_SHAM_D_recall_min','new_SHAM_D',0),('new_COVER_E_recall_min','new_COVER_E',1),('new_ALL_S_hidden_recall_min','new_ALL_S',1)]
    for key,cohort,target in specs:gates[key]=dict(value=float((p[vm[cohort]]==target).mean()),limit=design['source_gates'][key])
    gates['old_accepted_VAL_pair_all3_min']=dict(value=result['old_exact_V96']['VALIDATION']['paired']['all3_correct_rate'],limit=design['source_gates']['old_accepted_VAL_pair_all3_min'])
    gates['new_selected_VAL_pair_all3_min']=dict(value=result['new']['VALIDATION']['selected_pairs']['all3_correct_rate'],limit=design['source_gates']['new_selected_VAL_pair_all3_min'])
    for key,dom in(('old_known_VAL_U_rate_max','V96_REAL'),('new_known_VAL_U_rate_max','V99_SYNTH')):
        mask=(m['domain']==dom)&(m['split']=='VALIDATION')&m['known'];gates[key]=dict(value=float((~a[mask]).mean()),limit=design['source_gates'][key])
    for k,g in gates.items():g['PASS']=g['value']is not None and(g['value']<=g['limit']if k.endswith('_max')else g['value']>=g['limit'])
    assert len(gates)==13
    return result,gates

def audit_sampler(meta,trace,base,available):
    # Torch is a CPU random-number generator only: no model/weights/feature arrays.
    import torch
    assert torch.__version__=='2.1.1+cu118'
    g=torch.Generator(device='cpu').manual_seed(205);tm=masks(meta,'TRAIN');pools={k:np.flatnonzero(v)for k,v in tm.items()}
    old=meta['old_pairs'][meta['split'][meta['old_pairs'][:,0]]=='TRAIN'];new=meta['new_pairs'][meta['split'][meta['new_pairs'][:,0]]=='TRAIN']
    quotas=(('old_LOCO_BEFORE_D',4),('old_PALLOX_BEFORE_D',4),('old_BEFORE_E',8),('new_ALL_S',8),('new_BEFORE_D',0),('new_ALL_E',8))
    # ALL_D/E pools include every variant, matching the fixed design, not onlyBEFORE.
    mm=(meta['domain']=='V99_SYNTH')&(meta['split']=='TRAIN')
    for t in('S','D','E'):pools['new_ALL_'+t]=np.flatnonzero(mm&(meta['typed']==t))
    quotas=(('old_LOCO_BEFORE_D',4),('old_PALLOX_BEFORE_D',4),('old_BEFORE_E',8),('new_ALL_S',8),('new_ALL_D',8),('new_ALL_E',8))
    assert len(trace)==2000
    for step,r in enumerate(trace,1):
        assert r['update']==step
        os=torch.randint(len(old),(4,),generator=g).numpy();ns=torch.randint(len(new),(4,),generator=g).numpy();triples=np.r_[old[os],new[ns]]
        parts=[triples.reshape(-1)]
        for name,n in quotas:parts.append(pools[name][torch.randint(len(pools[name]),(n,),generator=g).numpy()])
        pre=np.concatenate(parts);perm=torch.randperm(64,generator=g).numpy();ix=pre[perm];positions=np.argsort(perm)[np.arange(24).reshape(8,3)]
        arreq(r['occurrence_indices'],ix,'traceoccurrence');arreq(r['input_unit_indices'],meta['units'][ix],'traceunits');arreq(r['matched_triplet_batch_positions'],positions,'tripletpositions')
        audit=r['draw'];arreq(audit['old_pair_pool_indices'],os,'oldpool');arreq(audit['new_pair_pool_indices'],ns,'newpool');arreq(audit['unshuffled_occurrence_indices'],pre,'preshuffle');arreq(audit['shuffle_permutation'],perm,'shuffle')
        assert audit['cohort_draw_quotas']==dict(old_triplets=4,new_triplets=4,**dict(quotas))
        assert (meta['split'][ix]=='TRAIN').all()and meta['known'][ix].all()
        assert (meta['labels'][ix]==0).sum()==32 and(meta['labels'][ix]==1).sum()==32
        arreq(ix[positions],triples,'matchedtriples');arreq(meta['labels'][triples],np.tile([0,1,0],(8,1)),'triplelabels')
        assert r['available_known_count']==int(available[ix].sum())and r['known_unavailable_count']==int((~available[ix]).sum())and r['unknown_count']==0
        correct=available[ix]&((base[ix]>=0)==(meta['labels'][ix]==1))
        assert r['correct_base_preservation_N']==int(correct.sum())and r['available_margin_pairs']==int((available[triples[:,1]]&available[triples[:,2]]).sum())
        vals=[r[k]for k in('BCE','delta_L2','matched_hinge','loss','gradient_norm_before_clip')];assert np.isfinite(vals).all()and min(vals)>=0
        expected=np.float32(np.float32(r['BCE'])+np.float32(.05)*np.float32(r['delta_L2']))+np.float32(.25)*np.float32(r['matched_hinge'])
        assert float(expected)==r['loss'],('lossarithmetic',step,float(expected),r['loss'])
    return dict(updates=2000,rows_per_update=64,D_per_update=32,hidden_per_update=32,matched_triplets_per_update=8,allTRAIN_only=True,all_known=True,all_occurrence_and_unit_indices_exact=True,all_seed205_CPU_RNG_draws_and_permutations_exact=True,loss_arithmetic_FP32_exact=True,model_or_optimizer_reexecutions=0)

def run():
    scope=read(H/'SCOPE_BEFORE_SAVED_OUTPUTS.json');fit=read(check(scope['FIT_COMPLETE']));assert fit['completed']and fit['PASS_execution']and fit['optimizer_updates']==2000
    for k in('MODEL_FREEZE','source_results','source_gates','scores','history','train_updates','update0_parity'):check(fit[k])
    protocol=read(check(scope['protocol']));design=read(check(scope['design']));bindings=protocol['bindings']
    for key in('old_manifest','new_manifest','old_GT','new_GT','new_queries','new_config','old_C1_scores'):check(bindings[key])
    meta=independent_metadata(bindings)
    with np.load(check(fit['scores']),allow_pickle=False)as f:s={k:f[k]for k in f.files}
    expected={'scalar_target_D0_EorS1_Uminus1':meta['labels'],'typed_state':meta['typed'],'candidate_known':meta['known'],'source':meta['source'],'split':meta['split'],'original_split':meta['original_split'],'variant':meta['variant'],'domain':meta['domain'],'global_occurrence_index':np.arange(N),'domain_occurrence_index':np.r_[np.arange(NOLD),np.arange(NNEW)],'occurrence_id':meta['ids'],'input_unit_index':meta['units'],'query_xy_raw':meta['q'],'L_raw':meta['L'],'query_available':meta['available'],'old_accepted_triplets':meta['old_pairs'],'new_selected_triplets':meta['new_pairs'],'old_original_C1_V95_label':meta['old_original_label'],'appearance':meta['appearance'],'aspect':meta['aspect'],'azimuth':meta['azimuth'],'elevation':meta['elevation']}
    for k,v in expected.items():arreq(s[k],v,k)
    z=s['query_logit'];a=s['query_available'];base=s['base_logit'];delta=s['delta_logit']
    for x in(z,base,delta):assert x.shape==(N,)and x.dtype==np.float32 and np.isfinite(x).all()
    assert a.dtype==bool and np.array_equal(z,base+delta)and np.count_nonzero(z[~a])==np.count_nonzero(base[~a])==np.count_nonzero(delta[~a])==0
    pred=np.where(a,(z>=0).astype(np.int8),-1);arreq(s['prediction'],pred,'rawsigndecoder')
    safe=np.where(a,z,0).astype(np.float64);nll=np.full(N,np.nan);k=meta['known'];nll[k]=np.logaddexp(0,safe[k])-meta['labels'][k]*safe[k];arreq(s['NLL_known_FP64'],nll,'knownNLL')
    with np.load(check(bindings['old_C1_scores']),allow_pickle=False)as f:
        assert np.array_equal(base[:NOLD],f['query_logit'])and np.array_equal(a[:NOLD],f['query_available'])
    results=read(check(fit['source_results']));same(results['counts'],counts(meta),'counts');same(protocol['counts'],counts(meta),'protocolcounts')
    history=rows(check(fit['history']));assert [r['update']for r in history]==list(range(0,2001,100));vm=masks(meta,'VALIDATION')
    for r in history:
        assert tuple(r['VAL_cohorts'])==NAMES
        for name,mask in vm.items():assert r['VAL_cohorts'][name]['count']==int(mask.sum())and r['VAL_cohorts'][name]['unavailable']==int((mask&~a).sum())and np.isfinite(r['VAL_cohorts'][name]['mean_BCE'])
        assert r['criterion_nine_VAL_cohorts_BCE']==float(np.mean([r['VAL_cohorts'][name]['mean_BCE']for name in NAMES]))
    best=history[0]
    for r in history[1:]:
        if r['criterion_nine_VAL_cohorts_BCE']<best['criterion_nine_VAL_cohorts_BCE']:best=r
    freeze=read(check(fit['MODEL_FREEZE']));assert best['update']==fit['selected_update']==results['selected_update']==freeze['selected_update']==1900
    parts={name:dict(count=int(mask.sum()),mean_BCE=float(nll[mask].mean()),unavailable=int((mask&~a).sum()))for name,mask in vm.items()}
    same(parts,best['VAL_cohorts'],'selectedhistoryparts');same(parts,results['VAL_cohorts'],'resultparts')
    criterion=float(np.mean([v['mean_BCE']for v in parts.values()]));same(criterion,results['criterion_nine_VAL_cohorts_BCE'],'selectedcriterion')
    metrics,gates=recompute_metrics(meta,z,a,base,delta,design);same(metrics,results['metrics'],'allmetrics')
    savedg=read(check(fit['source_gates']));assert set(savedg['gates'])==set(gates)
    for name,g in gates.items():
        for key in('value','limit','PASS'):same(savedg['gates'][name][key],g[key],'gate.'+name+'.'+key)
    passed=all(x['PASS']for x in gates.values());assert savedg['all_gates_pass']==savedg['PASS']==passed and not savedg['current_inference_authorized']
    sampler=audit_sampler(meta,rows(check(fit['train_updates'])),base,a)
    parity=read(check(fit['update0_parity']));assert parity['PASS']and parity['base_state_sha256']==freeze['base_state_sha256']==parity['load_receipt']['loaded_base_state_sha256']and parity['all_delta_zero']and parity['old_combined_base_logit_and_availability_bitexact']and parity['new2304_combined_equals_exact_frozenbase']
    assert freeze['base_frozen']and freeze['parameter_count']==29794 and freeze['trainable_parameter_count']==3221 and freeze['current_used_for_fit_selection']is False and freeze['current_inference_authorized']is False
    write(H/'INDEPENDENT_RECOMPUTED_SOURCE_METRICS.json',dict(completed=True,metrics=metrics,gates=gates,selected_cohorts=parts,criterion=criterion,source_gate_PASS=passed))
    review=dict(completed=True,PASS=True,blockers=[],scope=bind(H/'SCOPE_BEFORE_SAVED_OUTPUTS.json'),auditor=bind(__file__),FIT_COMPLETE=scope['FIT_COMPLETE'],trainer=scope['trainer'],score_binding=fit['scores'],occurrences=N,old_rows=NOLD,new_rows=NNEW,identity_label_qL_split_unit_availability_joins='ALL33984 exact',combined_FP32_base_plus_delta_exact=True,rawsign0_decoder_and_U_safezero_exact=True,known_FP64_NLL_exact=True,old_frozen_C1_savedbase_logit_and_availability_exact=True,metadata_original_before_C1_V95_E20_retained_separately=True,canonical_old_before_D_E_U=[61,4,895],geometric_S_is_hidden_binary_cause_not_predicted=True,history_rows=21,earliest_strict_min_selected_update=best['update'],selected_criterion=criterion,sampler=sampler,numeric_and_structure_leaf_comparisons=checks,recomputed_metrics=bind(H/'INDEPENDENT_RECOMPUTED_SOURCE_METRICS.json'),all13_source_gates_recomputed=True,source_gate_pass_count=sum(g['PASS']for g in gates.values()),source_gate_fail_count=sum(not g['PASS']for g in gates.values()),source_gates_all_PASS=passed,current_inference_authorized=False,base_state_hash_receipts_consistent=True,checkpoint_parameters_deserialized=0,new_model_features_forward_fit_private_current_PnP=0,limits=['Audit PASS establishes saved arithmetic/lineage/recipe consistency, not successful generalization or human semantic gold.','All21 history criteria are replayed from saved cohort means; only selected33984 saved scores are available for independent per-row NLL, no intermediate checkpoint forward.','Base value immutability is sourceparity/freeze receipt consistency plus exact old savedbase scores; checkpoint tensors were not loaded.','CPU torch is used solely to reproduce the fixed seed205 sampler RNG; all joins, metric/loss arithmetic and gates are independently NumPy.'])
    write(H/'REVIEW.json',review);write(H/'SEAL.json',dict(completed=True,PASS=True,review=bind(H/'REVIEW.json'),files=[bind(p)for p in sorted(H.iterdir())if p.is_file()]))
    print(json.dumps(dict(review=bind(H/'REVIEW.json'),seal=bind(H/'SEAL.json'),gate_pass=review['source_gate_pass_count'],gate_fail=review['source_gate_fail_count'],selected_update=best['update'],comparisons=checks)))

if __name__=='__main__':run()
