"""V150 saved source score/teacher arithmetic only; no learned-model imports."""
from pathlib import Path
from collections import Counter
import csv,hashlib,json,math,time
import numpy as np
H=Path(__file__).resolve().parent;V=H.parent;P=V.parent
POLICY=V/'ROOT_SOURCE_MECHANISM_POLICY_BEFORE_SAVED_ANALYSIS.json'
POLICY_SHA='777c44db64d04ee439f2fc210898ccbca18deb10dee60df0a051493c4dc460ce'
MODELS=('INITIAL_C32','C33_FINAL2000','C34_FINAL2000')
COMPARISONS=((0,1),(1,2),(0,2));INPUTS=[]
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def check(b):
    p=Path(b['path']);s=str(p).lower()
    assert '_v112'not in s and '/private_evaluation/'not in s and '_v149'not in s
    assert p.suffix.lower()not in('.pt','.pth','.png','.jpg','.jpeg')
    assert not(p.suffix=='.npz'and'feature'in p.name.lower())
    assert sha(p)==b['sha256'],p
    if b not in INPUTS:INPUTS.append(b)
    return p
def clean(x):
    if isinstance(x,dict):return{k:clean(v)for k,v in x.items()}
    if isinstance(x,(list,tuple,np.ndarray)):return[clean(v)for v in x]
    if isinstance(x,(np.integer,)):return int(x)
    if isinstance(x,(np.bool_,)):return bool(x)
    if isinstance(x,(float,np.floating)):return float(x)if math.isfinite(x)else None
    return x
def write(p,x):
    with Path(p).open('x')as f:json.dump(clean(x),f,indent=2,allow_nan=False);f.write('\n')
def lines(p):return[json.loads(s)for s in Path(p).read_text().splitlines()if s]
def exact(a,b):return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
def stats(x):
    x=np.asarray(x,dtype=np.float64);x=x[np.isfinite(x)]
    if not len(x):return dict(N=0)
    return dict(N=len(x),mean=x.mean(),std=x.std(),minimum=x.min(),q05=np.quantile(x,.05),q25=np.quantile(x,.25),median=np.median(x),q75=np.quantile(x,.75),q95=np.quantile(x,.95),maximum=x.max(),positive_fraction=np.mean(x>0),negative_fraction=np.mean(x<0),zero_fraction=np.mean(x==0))
def auc(y,z):
    y=np.asarray(y);z=np.asarray(z);n1=int((y==1).sum());n0=int((y==0).sum())
    if not n1 or not n0:return None
    order=np.argsort(z,kind='mergesort');zz=z[order];yy=y[order];total=0.;start=0;below0=0
    while start<len(zz):
        end=start+1
        while end<len(zz)and zz[end]==zz[start]:end+=1
        positives=int((yy[start:end]==1).sum());negatives=end-start-positives
        total+=positives*(below0+.5*negatives);below0+=negatives;start=end
    return total/(n1*n0)
def sigmoid(z):
    z=np.asarray(z,dtype=np.float64);return np.exp(-np.logaddexp(0.,-z))
def cm(y,p,mask):return[[int((mask&(y==a)&(p==b)).sum())for b in(0,1,-1)]for a in(0,1)]
def ratio(n,d):return n/d if d else None
def metrics(y,p,mask):
    c=cm(y,p,mask);a,b=c
    return dict(CM_D_E_rows_D_E_U_columns=c,D_recall=ratio(a[0],sum(a)),E_recall=ratio(b[1],sum(b)),D_precision=ratio(a[0],a[0]+b[0]),E_precision=ratio(b[1],b[1]+a[1]),known_DE=sum(a)+sum(b),prediction_U_known=a[2]+b[2],U_rate_known=ratio(a[2]+b[2],sum(a)+sum(b)))
def load_score(binding,n):
    names=('raw_head_logit_before_U','query_available','head_forwarded','global_index','occurrence_id','query_id','query_xy','L','scene_index','base_index','native_slot','frame_index','split')
    with np.load(check(binding),allow_pickle=False)as f:a={k:f[k].copy()for k in names if k in f.files}
    assert a['raw_head_logit_before_U'].shape==(n,)and a['raw_head_logit_before_U'].dtype==np.float32
    assert a['global_index'].dtype.kind in'iu'and np.array_equal(a['global_index'],np.arange(n))
    assert a['query_available'].dtype==bool and a['head_forwarded'].dtype==bool
    assert np.array_equal(np.isfinite(a['raw_head_logit_before_U']),a['head_forwarded'])
    assert not(a['query_available']&~a['head_forwarded']).any()
    return a
def join_score(a,b):
    assert np.array_equal(a['global_index'],b['global_index'])
    for key in('query_xy','L','query_available','head_forwarded'):
        assert exact(a[key],b[key]),key
    k='occurrence_id'if'occurrence_id'in a else'query_id';assert exact(a[k],b[k])
def make_native(name,n,bindings,GT):
    packs=[load_score(b,n)for b in bindings]
    for p in packs[1:]:join_score(packs[0],p)
    rows=lines(check(GT));assert len(rows)==2*n
    rows=rows[::2];assert all(r['convention']=='CF_PRIMARY_DECLARED'for r in rows)
    a=packs[0]
    for i,r in enumerate(rows):
        assert r['native_slot']==i%8 and r['scene_index']==i//8 and r['base_index']==i//32
        assert r['query_id']==str(a['occurrence_id'][i])
        assert np.array_equal(a['query_xy'][i],r['query_xy']if r['query_xy']is not None else[np.nan,np.nan],equal_nan=True)
        assert np.array_equal(a['L'][i],r['roi_side']if r['roi_side']is not None else np.nan,equal_nan=True)
    meta=dict(corpus=name,rows=rows,state=np.array([r['teacher_state']for r in rows]),fold=np.array([r['split']for r in rows]),native=np.arange(n)%8,
        variant=np.array([r['variant']for r in rows]),band=np.array([r['position_band']for r in rows]),boundary=np.array([r['correspondence_boundary']for r in rows],bool),q=a['query_xy'],L=a['L'],available=a['query_available'],constructed=a['head_forwarded'])
    for k in('foreign_family','material','aspect','geometry_family'):meta[k]=np.array([r['factors'][k]for r in rows])
    return finish(meta,packs)
def finish(m,packs):
    m['z']=np.stack([p['raw_head_logit_before_U']for p in packs]);m['target']=np.where(m['state']=='D',0,np.where(m['state']=='E',1,-1))
    m['pred']=np.where(m['available'][None,:],(m['z']>=0).astype(np.int8),-1)
    m['N']=len(m['state']);assert m['pred'].shape==(3,m['N'])
    return m
def make_public(bindings,label_binding):
    packs=[load_score(b,512)for b in bindings]
    for p in packs[1:]:join_score(packs[0],p)
    rows=lines(check(label_binding));a=packs[0];assert len(rows)==512
    errors=[];bands=[]
    for i,r in enumerate(rows):
        assert r['global_index']==i and r['frame_index']==i//8 and r['native_slot']==i%8
        assert str(a['query_id'][i])==r['id']+':native'+str(i%8)and np.array_equal(a['query_xy'][i],r['query_xy'])and a['L'][i]==r['L']
        assert bool(a['query_available'][i])==r['original_input_valid']
        validref=r['semantic_state']in'DE'and r['source_reference_xy']is not None
        e=float(np.linalg.norm(a['query_xy'][i]-np.asarray(r['source_reference_xy'])))if validref else None;errors.append(e)
        bands.append('UNCERTIFIED_REFERENCE'if e is None else'LE3_PX'if e<=3 else'GT3_LE8_PX'if e<=8 else'GT8_LE20_PX'if e<=20 else'GT20_PX')
    m=dict(corpus='PUBLIC_NATIVE_WEAK',rows=rows,state=np.array([r['semantic_state']for r in rows]),fold=np.array(['TRAIN'if r['split']=='TRAIN_LAB'else'VALIDATION'for r in rows]),native=np.arange(512)%8,
        variant=np.array(['ORIGINAL_REAL']*512),band=np.array(bands),boundary=np.zeros(512,bool),q=a['query_xy'],L=a['L'],available=a['query_available'],constructed=a['head_forwarded'],public_reference_error=errors)
    for k in('foreign_family','material','aspect','geometry_family'):m[k]=np.array(['NOT_CERTIFIED_SOURCE_FACTOR']*512)
    assert Counter(m['state'])==Counter(D=70,E=14,U=428)and m['available'].sum()==463
    return finish(m,packs)
def group_masks(m):
    for fold in('ALL','TRAIN','VALIDATION'):
        f=np.ones(m['N'],bool)if fold=='ALL'else m['fold']==fold
        yield fold,f
        for key in('native','variant','band','foreign_family','material','aspect','geometry_family'):
            for value in np.unique(m[key]):yield fold+'::'+key+'::'+str(value),f&(m[key]==value)
        for k in range(8):
            for variant in np.unique(m['variant']):yield fold+'::native_variant::'+str(k)+'::'+variant,f&(m['native']==k)&(m['variant']==variant)
def comparison(m,mask,i,j):
    use=mask&m['available']&np.isfinite(m['z'][i])&np.isfinite(m['z'][j]);d=m['z'][j,use].astype(np.float64)-m['z'][i,use]
    y=m['target'];known=mask&(y>=0);p,q=m['pred'][i],m['pred'][j]
    out=dict(delta_raw_logit=stats(d),delta_D_E_correct_signed_margin=stats(d*np.where(y[use]==1,1.,-1.))if np.all(y[use]>=0)else None,
        class_flips={a+'->'+b:int((mask&(p==x)&(q==z)).sum())for a,x in(('D',0),('E',1),('U',-1))for b,z in(('D',0),('E',1),('U',-1))},
        recovered_known=int((known&(p!=y)&(q==y)).sum()),damaged_known=int((known&(p==y)&(q!=y)).sum()),
        finite_available_N=int(use.sum()),inputU_retained_N=int((mask&~m['available']).sum()),
        uniform_offset_residual_RMSE=float(np.std(d))if len(d)else None,exact_constant_shift=bool(len(d)>0 and np.all(d==d[0])))
    return out
def group_report(m,name,mask):
    out=dict(corpus=m['corpus'],group=name,N=int(mask.sum()),teacher_states=dict(Counter(m['state'][mask])),available=int((mask&m['available']).sum()),inputU=int((mask&~m['available']).sum()),states={})
    for state in('ALL','D','E','S','U'):
        f=mask if state=='ALL'else mask&(m['state']==state);known=f&m['available']&(m['target']>=0)
        models={}
        for i,key in enumerate(MODELS):
            models[key]=dict(binary_metrics=metrics(m['target'],m['pred'][i],f),finite_available_logit=stats(m['z'][i,f&m['available']]),
                fixed_signed_AUC_E1_vs_D0=auc(m['target'][known],m['z'][i,known]),AUC_available_known_N=int(known.sum()),full_known_N=int((f&(m['target']>=0)).sum()))
        out['states'][state]=dict(N=int(f.sum()),models=models,comparisons={MODELS[i]+'->'+MODELS[j]:comparison(m,f,i,j)for i,j in COMPARISONS})
    return out
def public_gradient(m):
    train=(m['fold']=='TRAIN')&m['available']&(m['target']>=0);assert train.sum()==43
    out={}
    for i,name in enumerate(MODELS):
        record={};push={};total=0.
        for state,y in(('D',0),('E',1)):
            ix=np.flatnonzero(train&(m['target']==y));z=m['z'][i,ix].astype(np.float64);s=sigmoid(z);loss=np.logaddexp(0,z)-y*z;deriv=s-y;weighted=.5*deriv/len(ix)
            record[state]=dict(N=len(ix),raw_logit=stats(z),sigmoid_E=stats(s),mean_BCE=loss.mean(),unweighted_derivative_sum=deriv.sum(),unweighted_derivative_mean=deriv.mean(),classbalanced_bias_derivative_sum=weighted.sum(),classbalanced_derivative_per_row=stats(weighted),indices=ix.tolist())
            push[state]=abs(float(weighted.sum()));total+=float(weighted.sum())
        out[name]=dict(classes=record,balanced_BCE=.5*(record['D']['mean_BCE']+record['E']['mean_BCE']),common_additive_bias_dloss_dz_sum=total,
            descent_direction_for_common_bias='NEGATIVE_FAVORS_D'if total>0 else'POSITIVE_FAVORS_E'if total<0 else'ZERO',D_to_E_absolute_class_bias_push_ratio=ratio(push['D'],push['E']),D_share_absolute_class_bias_push=ratio(push['D'],sum(push.values())))
    return dict(models=out,interpretation='Derivative of scalar logits/common additive bias only, not a full parameter gradient or proof of parameter-level cause. Native gradients, feature Jacobians, clipping, AdamW and trajectories also matter. Public43 was class-balanced, not count-weighted35:8.')
def pairs_native(m,stream):
    totals={};nbase=m['N']//32
    for base in range(nbase):
        for slot in range(8):
            ii=(base*4+np.arange(4))*8+slot;rr=[m['rows'][int(k)]for k in ii];assert all(r['base_index']==base and r['native_slot']==slot for r in rr)
            assert len({r['canonical_index']for r in rr})==1
            e,d=int(ii[1]),int(ii[2]);known=m['state'][e]=='E'and m['state'][d]=='D';available=bool(m['available'][e]and m['available'][d]);fold=str(m['fold'][e])
            row=dict(corpus=m['corpus'],base_index=base,native_slot=slot,canonical_index=rr[0]['canonical_index'],fold=fold,indices=ii,teacher_states=m['state'][ii],known_COVER_E_NEAR_D=known,both_available=available,
                q_movement_COVER_NEAR_px=float(np.linalg.norm(m['q'][e]-m['q'][d]))if np.isfinite(m['q'][[e,d]]).all()else None,L_abs_change_px=abs(float(m['L'][e]-m['L'][d]))if np.isfinite(m['L'][[e,d]]).all()else None,models={})
            for j,name in enumerate(MODELS):row['models'][name]=dict(raw_four=m['z'][j,ii],pred_four=m['pred'][j,ii],E_minus_D_logit_gap=float(m['z'][j,e]-m['z'][j,d])if np.isfinite(m['z'][j,[e,d]]).all()else None,both_correct=bool(known and m['pred'][j,e]==1 and m['pred'][j,d]==0))
            stream.write(json.dumps(clean(row),allow_nan=False)+'\n')
            if known:
                t=totals.setdefault(fold,dict(known_pairs=0,both_available=0,inputU_pairs=0,correct=[0,0,0],gaps=[[],[],[]],q_move=[]));t['known_pairs']+=1;t['both_available']+=available;t['inputU_pairs']+=not available
                if row['q_movement_COVER_NEAR_px']is not None:t['q_move'].append(row['q_movement_COVER_NEAR_px'])
                for j,name in enumerate(MODELS):
                    t['correct'][j]+=row['models'][name]['both_correct']
                    if available:t['gaps'][j].append(row['models'][name]['E_minus_D_logit_gap'])
    return {fold:dict(known_pairs=t['known_pairs'],both_available=t['both_available'],inputU_pairs=t['inputU_pairs'],both_correct_full_known_pairs=dict(zip(MODELS,t['correct'])),E_minus_D_gap_available={name:stats(t['gaps'][j])for j,name in enumerate(MODELS)},actual_q_movement=stats(t['q_move']),isolated_same_query_causal_contrast=False)for fold,t in totals.items()}
def gate_mask(m,group):
    parts=group.split('::');f=m['fold']==parts[0];i=1
    while i<len(parts):
        k=parts[i]
        if k=='NONBOUNDARY':f&=~m['boundary'];i+=1
        elif k=='BOUNDARY':f&=m['boundary'];i+=1
        elif k=='SAME_TARGET_MATERIAL':f&=m['foreign_family']=='SAME_TARGET_MATERIAL';i+=1
        elif k=='material_aspect':f&=(m['material']==parts[i+1])&(m['aspect']==parts[i+2]);i+=3
        else:assert k in('foreign_family','geometry_family');f&=m[k]==parts[i+1];i+=2
    return f
def gates_replay(native,limits,reports):
    out={}
    for j,name in enumerate(MODELS):
        gg={}
        for key,limit in limits.items():
            new=key.startswith('NEW_NATIVE::');prefix='NEW_NATIVE::'if new else'NATIVE::';group,state=key[len(prefix):].rsplit('::',1);m=native[int(new)];f=gate_mask(m,group)&(m['state']==state);n=int(f.sum());value=float((m['pred'][j,f]==int(state=='E')).mean())if n else None
            gg[key]=dict(N=n,value=value,limit=limit['limit'],PASS=bool(n and value is not None and value>=limit['limit']))
            expected=reports[j][key];assert gg[key]['N']==expected['N']and gg[key]['value']==expected['value']and gg[key]['limit']==expected['limit']and gg[key]['PASS']==expected['PASS'],(name,key)
        out[name]=dict(passed=sum(v['PASS']for v in gg.values()),total=len(gg),gates=gg)
    return out
def allrow(m,stream):
    for i,r in enumerate(m['rows']):
        y=int(m['target'][i]);row=dict(corpus=m['corpus'],index=i,id=r.get('query_id',r.get('id')),fold=m['fold'][i],original_split=r['split'],native_slot=int(m['native'][i]),variant=m['variant'][i],teacher_state=m['state'][i],target_D0_E1_Uminus1=y,
            query_xy=m['q'][i],L=m['L'][i],query_available=m['available'][i],head_forwarded=m['constructed'][i],position_band=m['band'][i],historical_area_disagreement=m['boundary'][i],
            error_px=r['error_px']if'error_px'in r else m['public_reference_error'][i],factors={k:m[k][i]for k in('foreign_family','material','aspect','geometry_family')},models={},comparisons={})
        for j,name in enumerate(MODELS):
            z=float(m['z'][j,i]);row['models'][name]=dict(raw_logit=z,prediction_D0_E1_Uminus1=int(m['pred'][j,i]),correct_signed_margin=z*(1 if y==1 else-1)if y>=0 and math.isfinite(z)else None)
            if m['corpus']=='PUBLIC_NATIVE_WEAK'and m['fold'][i]=='TRAIN'and m['available'][i]and y>=0:row['models'][name]['public_balanced_logit_derivative']=.5*(float(sigmoid(z))-y)/(35 if y==0 else 8)
        for a,b in COMPARISONS:row['comparisons'][MODELS[a]+'->'+MODELS[b]]=dict(delta_raw_logit=float(m['z'][b,i]-m['z'][a,i]),old_prediction=int(m['pred'][a,i]),new_prediction=int(m['pred'][b,i]))
        stream.write(json.dumps(clean(row),allow_nan=False)+'\n')
def main():
    started=time.monotonic();assert sha(POLICY)==POLICY_SHA;policy=read(POLICY);control=read(check(policy['source_control']));assert control['completed']and control['PASS']and control['selected_model']is None
    protocol=read(check(control['protocol']));base=read(check(protocol['C32_source_results']));fixed=read(check(protocol['fixed_public_pools_and_native35']));limits=fixed['native35_limits'];assert len(limits)==35
    arms=list(control['arms']);assert arms==['C33_PUBLIC_NATIVE_WEIGHT0_FIXED2000','C34_PUBLIC_NATIVE_WEIGHT1_FIXED2000']
    fits=[read(check(control['arms'][name]['fit']))for name in arms];results=[read(check(f['source_results']))for f in fits];assert all(r['selected_model']is None for r in results)
    out=[f['outputs']for f in fits];native=[]
    for name,n,suffix,source in zip(('OLD_NATIVE','NEW_NATIVE'),(6144,20480),('OLD6144','NEW20480'),protocol['native']):
        bindings=[source['baseline']]+[o['FINAL_'+suffix+'_SCORES.npz']for o in out]
        m=make_native(name,n,bindings,source['GT']);native.append(m)
        for o in out:
            initial=load_score(o['INITIAL_'+suffix+'_SCORES.npz'],n)
            assert exact(initial['raw_head_logit_before_U'],m['z'][0]);join_score(initial,load_score(source['baseline'],n))
    pub=make_public([out[0]['INITIAL_PUBLIC512_SCORES.npz'],out[0]['FINAL_PUBLIC512_SCORES.npz'],out[1]['FINAL_PUBLIC512_SCORES.npz']],protocol['public_labels'])
    otherinitial=load_score(out[1]['INITIAL_PUBLIC512_SCORES.npz'],512);assert exact(otherinitial['raw_head_logit_before_U'],pub['z'][0])
    reports=[{k:v for k,v in base['gates'].items()if k in limits}]+[r['native']['gates']for r in results]
    gates=gates_replay(native,limits,reports)
    ledger=H/'ALL27136_SOURCE_ROWS_LOGIT_DELTA.jsonl'
    with ledger.open('x')as stream:
        for m in(*native,pub):allrow(m,stream)
    pairfile=H/'ALL6656_NATIVE_BASE_SLOT_FOUR_VARIANTS.jsonl'
    with pairfile.open('x')as stream:pairs={m['corpus']:pairs_native(m,stream)for m in native}
    groups=H/'FIXED_COHORT_CLASS_LOGIT_DIAGNOSIS.jsonl';group_count=0;headline={}
    with groups.open('x')as stream:
        for m in(*native,pub):
            for name,mask in group_masks(m):
                report=group_report(m,name,mask);stream.write(json.dumps(clean(report),allow_nan=False)+'\n');group_count+=1
                if name in('TRAIN','VALIDATION'):headline[m['corpus']+'::'+name]=report
    gradient=public_gradient(pub)
    logs=[]
    for i,f in enumerate(fits):
        rr=lines(check(f['outputs']['ALL2000_DRAWS_AND_LOSSES.jsonl']));assert len(rr)==2000
        for j,r in enumerate(rr):assert r['update']==j+1 and len(r['public_TRAIN_indices'])==43 and r['public_weight']==i
        logs.append(dict(arm=arms[i],first=rr[0],last=rr[-1],public_D_BCE_all2000=stats([r['public_D_BCE']for r in rr]),public_E_BCE_all2000=stats([r['public_E_BCE']for r in rr]),public_weight=i))
    # Reproduce initial public scalar losses against both first logged FP32 losses.
    for r in logs:
        initial=gradient['models']['INITIAL_C32'];assert abs(r['first']['public_D_BCE']-initial['classes']['D']['mean_BCE'])<2e-5 and abs(r['first']['public_E_BCE']-initial['classes']['E']['mean_BCE'])<2e-5
    summary=dict(completed=True,PASS=True,policy=bind(POLICY),source_control=policy['source_control'],models=list(MODELS),all_rows=27136,native6144_new20480_public512_retained=True,
        native35_exact_replay=gates,public_initial_and_final_scalar_gradient=gradient,fold_headlines=headline,native_same_physical_corner_pairs=pairs,public_loss_trace=logs,fixed_group_records=group_count,
        inputU={m['corpus']:dict(all_rows=m['N'],U=int((~m['available']).sum()),raw_nonfinite=int((~m['constructed']).sum()),finite_but_inputU=int((~m['available']&m['constructed']).sum()),teacher_states=dict(Counter(m['state'])))for m in(*native,pub)},
        limits=['Scalar bias derivative does not identify the full parameter gradient or unique cause.','C32->C33 is common extra native fine-tuning; C33->C34 is net public-gradient intervention including optimizer/clipping trajectory.','Same-base COVER/NEAR images retain different original q/L; pair gaps are not isolated same-query causal effects.','Public labels are source-specific weak lower2/3 only; limited projected references do not redefine virtual native semantics.','Source AUC and offset summaries are diagnostic; no threshold/normalizer/model selection, no gate relaxation.'],selected_model=None,full_goal_solved=False,current223_V149_V112_model_feature_checkpoint_RGB_forward_fit_access=0)
    target=H/'RESULTS.json';write(target,summary)
    for binding in list(INPUTS):check(binding)
    write(H/'DIAGNOSIS_SEAL.json',dict(completed=True,PASS=True,code=bind(__file__),policy=bind(POLICY),inputs=INPUTS,results=bind(target),allrow_ledger=bind(ledger),all_native_pair_ledger=bind(pairfile),fixed_cohort_reports=bind(groups),all27136_retained=True,
        source35_criteria_unchanged=True,selected_model=None,model_feature_checkpoint_RGB_current_V149_V112_forward_fit_reads_or_calls=0,seconds=time.monotonic()-started))
    print(json.dumps(dict(completed=True,PASS=True,seal=bind(H/'DIAGNOSIS_SEAL.json'),results=bind(target),public_initial=clean(gradient['models']['INITIAL_C32']))),flush=True)
if __name__=='__main__':main()
