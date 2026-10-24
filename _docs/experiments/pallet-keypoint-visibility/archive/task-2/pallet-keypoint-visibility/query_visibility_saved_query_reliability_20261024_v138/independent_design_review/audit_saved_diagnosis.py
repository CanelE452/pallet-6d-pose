"""Independent NumPy-only V138 saved AUC/target/bin audit; no models/private/RGB."""
from pathlib import Path
import hashlib,json,time,sys,collections
import numpy as np
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;V=H.parent;P=V.parent;D=V/'saved_observability_diagnosis_v1';T=time.monotonic()
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def bind(p):return dict(path=str(p),sha256=sha(p))
def read(p):return json.loads(Path(p).read_text())
def checked(b):assert sha(b['path'])==b['sha256'];return Path(b['path'])
def packet(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def lines(p):return [json.loads(s) for s in Path(p).read_text().splitlines() if s]
def eq(a,b):assert np.array_equal(a,b,equal_nan=True)
def same_dict(expected,actual):
    assert set(expected)==set(actual)
    for k,a in expected.items():
        b=actual[k]
        if isinstance(a,dict):same_dict(a,b)
        elif isinstance(a,float):assert isinstance(b,(float,int)) and abs(a-b)<1e-14,(k,a,b)
        else:assert a==b,(k,a,b)
def auc_independent(s,y,mask):
    # Mann-Whitney pair counting, independently of the producer's pooled tied ranks.
    defined=np.isfinite(s);known=mask&((y==0)|(y==1));used=known&defined;pos=s[used&(y==1)];neg=np.sort(s[used&(y==0)]);np_,nn=len(pos),len(neg)
    auc=None
    if np_ and nn:
        lo=np.searchsorted(neg,pos,'left');hi=np.searchsorted(neg,pos,'right');auc=float((lo.sum()+.5*(hi-lo).sum())/(np_*nn))
    return dict(AUC_fixed_higher_score_positive=auc,original_rows=int(mask.sum()),target_known=int(known.sum()),target_unknown=int((mask&~((y==0)|(y==1))).sum()),input_defined_all=int((mask&defined).sum()),input_undefined_all=int((mask&~defined).sum()),ranked_positive=np_,ranked_negative=nn,unranked_positive=int((known&~defined&(y==1)).sum()),unranked_negative=int((known&~defined&(y==0)).sum()),one_or_zero_ranked_classes=not(np_ and nn))
def bincounts(b,y,mask):
    out={}
    for code,name in ((-2,'NO_TRAIN_REFERENCE'),(-1,'INPUT_UNDEFINED'),(0,'Q1'),(1,'Q2'),(2,'Q3'),(3,'Q4'),(4,'Q5')):
        cohort=mask&(b==code);known=cohort&np.isin(y,[0,1]);pos=int((known&(y==1)).sum());n=int(known.sum())
        out[name]=dict(original_rows=int(cohort.sum()),target_known=n,target_unknown=int((cohort&~np.isin(y,[0,1])).sum()),positive=pos,negative=n-pos,positive_rate=pos/n if n else None)
    return out

inputreview=read(H/'SAVED_INPUT_TABLE_REVIEW.json');assert inputreview['PASS'] and inputreview['GT_reads']==inputreview['C29_score_reads']==0
sp=D/'DIAGNOSIS_SEAL.json';seal=read(sp);assert seal['completed'] and seal['PASS'] and seal['all26624_preserved']
for key in ('input_seal','binning_seal','results','full_table','code'):checked(seal[key])
results=read(seal['results']['path']);entry=read(checked(results['entry']));inputseal=read(checked(seal['input_seal']));x=packet(checked(inputseal['table']))
bs=read(seal['binning_seal']['path']);assert bs['completed'] and bs['PASS'] and bs['GT_read'] is False and bs['C29score_read'] is False and bs['input_seal']==seal['input_seal']
bb=packet(checked(bs['bins']));proof=read(checked(bs['provenance']));assert proof['GT_or_C29_score_body_reads']==0 and proof['all_TRAIN_input_distribution_not_teacher_or_VAL']
names=x['risk_names'].tolist()+x['descriptor_names'].tolist();values=np.c_[x['risks'],np.where(x['descriptor_valid'],x['descriptors'],np.nan)]
assert len(names)==68 and names==bb['signal_names'].tolist();eq(bb['global_index'],x['global_index']);assert np.array_equal(bb['corpus'],x['corpus']);eq(bb['source_occurrence_index'],x['source_occurrence_index']);eq(bb['score_defined'],np.isfinite(values))
expectedbins=np.full(values.shape,-1,np.int8);data={};inputbinproofs=0
for corpus in ('OLD_NATIVE','CONTINUOUS_NATIVE'):
    ids=np.flatnonzero(x['corpus']==corpus);train=x['split'][ids]=='TRAIN'
    for j,name in enumerate(names):
        s=values[ids,j];finite=s[train&np.isfinite(s)];binj=np.full(len(ids),-1,np.int8)
        if len(finite):
            edges=np.quantile(finite,[.2,.4,.6,.8]);good=np.isfinite(s);binj[good]=np.searchsorted(edges,s[good],side='right');digest=hashlib.sha256(np.asarray(edges,dtype='<f8').tobytes()).hexdigest();distinct=len(np.unique(edges))
        else:binj[np.isfinite(s)]=-2;digest=hashlib.sha256(b'').hexdigest();distinct=0
        expectedbins[ids,j]=binj
        expected=dict(TRAIN_finite_input_N=len(finite),TRAIN_all_input_N=int(train.sum()),edge_sha256=digest,distinct_quantile_edges=distinct,train_bin_counts={str(i):int((train&(binj==i)).sum()) for i in (-2,-1,0,1,2,3,4)},numeric_boundaries_exported=False,side='right',GT_or_VAL_used=False,no_finite_TRAIN_reference=not len(finite),bin_minus2_means_defined_input_without_TRAIN_reference=True)
        same_dict(expected,proof['proof'][corpus][name]);inputbinproofs+=1
    gs=read(checked(entry['GT_seals'][corpus]));assert gs['completed'] and gs['PASS'] and gs['RGB_read'] is False
    led=next(a for a in gs['outputs'] if Path(a['path']).name==f'ALL{len(ids)}_SLOTS_TWO_CONVENTIONS_GT_ONLY.jsonl');rr=lines(checked(led));assert len(rr)==2*len(ids);gt=rr[::2];score=packet(checked(entry['C29_scores'][corpus]))
    assert np.array_equal(score['occurrence_id'],x['occurrence_id'][ids]);eq(score['query_xy'],x['query_xy'][ids]);eq(score['L'],x['L'][ids]);eq(score['query_available'],x['query_available'][ids])
    pred=score['prediction'];av=score['query_available'];eq(pred,np.where(av,score['raw_head_logit_before_U']>=0,-1));assert np.isfinite(score['raw_head_logit_before_U'][score['head_forwarded']]).all() and np.isnan(score['raw_head_logit_before_U'][~score['head_forwarded']]).all()
    state=[];error=[]
    for i,r in enumerate(gt):
        assert r['convention']=='CF_PRIMARY_DECLARED' and r['query_id']==x['occurrence_id'][ids[i]] and r['scene_index']==i//8 and r['base_index']==i//32 and r['native_slot']==i%8 and r['split']==x['split'][ids[i]]
        q=np.asarray([np.nan]*2 if r['query_xy'] is None else r['query_xy']);ref=np.asarray(r['reference_xy']);eq(q,x['query_xy'][ids[i]])
        error_value=np.nan if not np.isfinite(q).all() else float(np.linalg.norm(q-ref));stored=np.nan if r['error_px'] is None else r['error_px'];assert (np.isnan(stored) and np.isnan(error_value)) or abs(stored-error_value)<1e-11
        assert r['roi_side']==(None if not np.isfinite(x['L'][ids[i]]) else float(x['L'][ids[i]]));state.append(r['teacher_state']);error.append(stored)
    state=np.asarray(state);error=np.asarray(error);assert np.array_equal(state,score['raw_teacher_state_CF']);assert np.isin(state,list('DESU')).all();eq(pred==-1,~av)
    truth=np.where(state=='D',0,np.where(state=='E',1,-1)).astype(np.int8);known=truth>=0;loc=np.where(np.isfinite(error),(error>3).astype(np.int8),-1).astype(np.int8);mistake=np.where(known&av,pred!=truth,-1).astype(np.int8);failure=np.where(known,pred!=truth,-1).astype(np.int8)
    targets={'GT_ERROR_GT3':loc,'C29_DE_MISTAKE_WHEN_INPUT_AVAILABLE':mistake,'C29_KNOWN_FAILURE_INCLUDING_INPUT_U':failure,'EXTERNAL_TEACHER_E1_D0':truth}
    data[corpus]=dict(ids=ids,teacher=state,error=error,pred=pred,available=av,targets=targets,fold=x['split'][ids],GT=led,score=entry['C29_scores'][corpus]);account={}
    for fold in ('ALL','TRAIN','VALIDATION'):
        fm=np.ones(len(ids),bool) if fold=='ALL' else x['split'][ids]==fold
        account[fold]=dict(original_rows=int(fm.sum()),teacher_states=dict(collections.Counter(state[fm].tolist())),C29_input_U_all_states=int((fm&~av).sum()),known_DE=int((fm&known).sum()),known_DE_input_U=int((fm&known&~av).sum()),finite_GT_qerror=int((fm&np.isfinite(error)).sum()),NO_QUERY_qerror_unknown=int((fm&~np.isfinite(error)).sum()))
    same_dict(account,results['accounting'][corpus])
eq(expectedbins,bb['bins'])
rows=lines(seal['full_table']['path']);assert len(rows)==15168==results['records'];seen=set();summary_index={};signalindex={n:j for j,n in enumerate(names)};auc_count=0
for r in rows:
    corpus=r['corpus'];d=data[corpus];ids=d['ids'];teacher=d['teacher'];error=d['error'];av=d['available'];fold=r['fold'];scope=r['availability_scope'];target=r['target'];name=r['signal'];j=signalindex[name]
    fm=np.ones(len(ids),bool) if fold=='ALL' else d['fold']==fold
    if target=='EXTERNAL_TEACHER_E1_D0':
        assert j<10 and r['teacher']=='ALL_WITH_SU_OUTSIDE_TARGET';group=r['GT_error_diagnostic_stratum'];em={'ALL_GT_ERROR':np.ones(len(ids),bool),'GT_ERROR_LE3':np.isfinite(error)&(error<=3),'GT_ERROR_GT3':np.isfinite(error)&(error>3),'NO_QUERY':~np.isfinite(error)}[group];mask=fm&em
    else:
        tm=np.ones(len(ids),bool) if r['teacher']=='ALL' else np.isin(teacher,list('DE')) if r['teacher']=='DE' else teacher==r['teacher'];mask=fm&tm;group=''
    if scope=='C29_INPUT_AVAILABLE_ONLY_SUPPLEMENT':mask &= av
    else:assert scope=='ALL_ORIGINAL'
    y=d['targets'][target];s=values[ids,j];expectedauc=auc_independent(s,y,mask);same_dict(expectedauc,r['auc']);same_dict(bincounts(expectedbins[ids,j],y,mask),r['TRAIN_input_quantile_bins']);auc_count+=1
    assert r['kind']==('PRIMARY_FIXED_HIGHER_RISK' if j<10 else 'RAW_VALUE_SIGNED_DESCRIPTIVE_NO_RISK_POLARITY') and r['fixed_sign_not_selected'] is True and r['definitional_inputU_overlap']==(j in (5,6))
    key=(corpus,fold,r['teacher'],scope,target,group,name);assert key not in seen;seen.add(key);summary_index[key]=r
expectedkeys=set()
for corpus in data:
    for fold in ('ALL','TRAIN','VALIDATION'):
        for teacher in ('ALL','DE','D','E','S','U'):
            for scope in ('ALL_ORIGINAL','C29_INPUT_AVAILABLE_ONLY_SUPPLEMENT'):
                for target in ('GT_ERROR_GT3','C29_DE_MISTAKE_WHEN_INPUT_AVAILABLE','C29_KNOWN_FAILURE_INCLUDING_INPUT_U'):
                    for name in names:expectedkeys.add((corpus,fold,teacher,scope,target,'',name))
        for group in ('ALL_GT_ERROR','GT_ERROR_LE3','GT_ERROR_GT3','NO_QUERY'):
            for scope in ('ALL_ORIGINAL','C29_INPUT_AVAILABLE_ONLY_SUPPLEMENT'):
                for name in names[:10]:expectedkeys.add((corpus,fold,'ALL_WITH_SU_OUTSIDE_TARGET',scope,'EXTERNAL_TEACHER_E1_D0',group,name))
assert seen==expectedkeys
for corpus,folds in results['summary'].items():
    for fold,teachers in folds.items():
        for teacher,targets_ in teachers.items():
            if teacher=='EXTERNAL_E1_D0_BY_GTERROR':
                for group,signals in targets_.items():
                    for name,auc in signals.items():same_dict(summary_index[(corpus,fold,'ALL_WITH_SU_OUTSIDE_TARGET','ALL_ORIGINAL','EXTERNAL_TEACHER_E1_D0',group,name)]['auc'],auc)
            else:
                for target,signals in targets_.items():
                    for name,auc in signals.items():same_dict(summary_index[(corpus,fold,teacher,'ALL_ORIGINAL',target,'',name)]['auc'],auc)
selected={}
for corpus in data:
    selected[corpus]={}
    for teacher,target,group in [('ALL','GT_ERROR_GT3',''),('D','GT_ERROR_GT3',''),('E','GT_ERROR_GT3',''),('ALL','C29_DE_MISTAKE_WHEN_INPUT_AVAILABLE',''),('D','C29_DE_MISTAKE_WHEN_INPUT_AVAILABLE',''),('E','C29_DE_MISTAKE_WHEN_INPUT_AVAILABLE',''),('ALL_WITH_SU_OUTSIDE_TARGET','EXTERNAL_TEACHER_E1_D0','ALL_GT_ERROR'),('ALL_WITH_SU_OUTSIDE_TARGET','EXTERNAL_TEACHER_E1_D0','GT_ERROR_LE3')]:
        keyname=teacher+'::'+target+(('::'+group) if group else '');selected[corpus][keyname]={}
        for name in names[:10]:selected[corpus][keyname][name]=summary_index[(corpus,'VALIDATION',teacher,'ALL_ORIGINAL',target,group,name)]['auc']
receipt=dict(completed=True,PASS=True,diagnosis_seal=bind(sp),input_only_review=bind(H/'SAVED_INPUT_TABLE_REVIEW.json'),code=bind(Path(__file__)),all15168_fixed_AUC_and_sevenbin_records_recomputed_independently=True,auc_method='Independent positive-versus-sorted-negative less-than/tie pair counts, not producer pooled ranks.',targets_all26624_primary_CF_recomputed=True,original_q_reference_error_recomputed=True,all136_TRAIN_input_bin_references_hashes_and_26624x68_bins_exact=True,all_full_table_grid_cells_present_once=True,all_source_accounting_and_summary_fulltable_matches=True,accounting=results['accounting'],selected_VALIDATION_fixed10_tables=selected,stage='saved GT/C29 numeric audit after root diagnosis COMPLETE',source_only=True,RGB_visualfeatures_checkpoint_model_private_V112_fit_PnP_calls_or_reads=0,seconds=time.monotonic()-T)
out=H/'SAVED_DIAGNOSIS_REVIEW.json';assert not out.exists();out.write_text(json.dumps(receipt,indent=2,ensure_ascii=False,allow_nan=False)+'\n');print(json.dumps(dict(completed=True,PASS=True,review=bind(out),records=auc_count,seconds=receipt['seconds'])),flush=True)
