"""Independent V150 saved-source arithmetic. No producer imports/calls.

Read only completed source scores, source GT/weak labels, code and JSON receipts.
No feature, checkpoint, image, model, optimizer, private benchmark or pose access.
"""
from pathlib import Path
from collections import Counter
import difflib
import hashlib
import json
import math
import numpy as np

H = Path(__file__).resolve().parent
V = H.parent
S = V / 'source_saved_diagnosis'
SEAL = S / 'DIAGNOSIS_SEAL.json'
EXPECTED = '3dce2f1ff20277c02c434bdeab10ee29e2e0b786591942d223e1cbee78c74c87'
MODELS = ('INITIAL_C32', 'C33_FINAL2000', 'C34_FINAL2000')
PAIRS = ((0,1),(1,2),(0,2))
READS = {}

def sha(path):
    p = Path(path)
    assert '_v112' not in str(p).lower() and '_v149' not in str(p).lower()
    assert p.suffix.lower() not in ('.pt','.pth','.png','.jpg','.jpeg')
    assert not (p.suffix == '.npz' and 'feature' in p.name.lower())
    return hashlib.sha256(p.read_bytes()).hexdigest()

def bind(path):
    p=Path(path).resolve(); b=dict(path=str(p),sha256=sha(p)); READS[str(p)]=b
    return b

def checked(b):
    assert bind(b['path']) == b, b
    return Path(b['path'])

def read(path):
    bind(path)
    return json.loads(Path(path).read_text())

def lines(path):
    bind(path)
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s]

def packet(binding):
    path=checked(binding)
    with np.load(path,allow_pickle=False) as f:
        return {k:f[k].copy() for k in f.files}

def byteeq(a,b):
    return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()

def same(a,b,where=''):
    if isinstance(a,dict):
        assert isinstance(b,dict) and set(a)==set(b),(where,set(a),set(b))
        for k in a:same(a[k],b[k],where+'/'+str(k))
    elif isinstance(a,(list,tuple)):
        assert isinstance(b,(list,tuple)) and len(a)==len(b),where
        for i,(x,y) in enumerate(zip(a,b)):same(x,y,where+'/'+str(i))
    elif isinstance(a,(float,np.floating)):
        assert b is not None and math.isclose(float(a),float(b),rel_tol=1e-12,abs_tol=1e-12),(where,a,b)
    else:
        assert a==b,(where,a,b)

def scalar(x):
    return None if not np.isfinite(x) else float(x)

def stats(x):
    x=np.asarray(x,np.float64).ravel();x=x[np.isfinite(x)]
    if not len(x):return dict(N=0)
    qs=np.quantile(x,[.05,.25,.5,.75,.95])
    return dict(N=len(x),mean=float(x.mean()),std=float(x.std()),minimum=float(x.min()),
        q05=float(qs[0]),q25=float(qs[1]),median=float(qs[2]),q75=float(qs[3]),q95=float(qs[4]),maximum=float(x.max()),
        positive_fraction=float((x>0).mean()),negative_fraction=float((x<0).mean()),zero_fraction=float((x==0).mean()))

def auc(y,z):
    # Independent pair-count implementation; producer uses a merged rank loop.
    d=np.sort(z[y==0].astype(np.float64));e=z[y==1].astype(np.float64)
    if not len(d) or not len(e):return None
    less=np.searchsorted(d,e,side='left');leq=np.searchsorted(d,e,side='right')
    return float(np.sum(less+.5*(leq-less))/(len(d)*len(e)))

def divide(n,d):return float(n/d) if d else None

def metrics(m,p,mask):
    cm=[[int(np.count_nonzero(mask&(m['y']==t)&(p==v))) for v in (0,1,-1)] for t in (0,1)]
    d,e=cm;n=sum(d)+sum(e);u=d[2]+e[2]
    return dict(CM_D_E_rows_D_E_U_columns=cm,D_recall=divide(d[0],sum(d)),E_recall=divide(e[1],sum(e)),
        D_precision=divide(d[0],d[0]+e[0]),E_precision=divide(e[1],e[1]+d[1]),known_DE=n,prediction_U_known=u,U_rate_known=divide(u,n))

def mask_group(m,name):
    p=name.split('::');fold=p.pop(0)
    mask=np.ones(m['N'],bool) if fold=='ALL' else m['fold']==fold
    while p:
        k=p.pop(0)
        if k in ('NONBOUNDARY','BOUNDARY'):mask &= (m['boundary'] if k=='BOUNDARY' else ~m['boundary'])
        elif k=='SAME_TARGET_MATERIAL':mask &= m['foreign_family']=='SAME_TARGET_MATERIAL'
        elif k=='material_aspect':mask &= (m['material']==p.pop(0))&(m['aspect']==p.pop(0))
        elif k=='native_variant':mask &= (m['native']==int(p.pop(0)))&(m['variant']==p.pop(0))
        elif k=='native':mask &= m['native']==int(p.pop(0))
        else:
            assert k in ('variant','band','foreign_family','material','aspect','geometry_family'),name
            mask &= m[k]==p.pop(0)
    return mask

def contrast(m,mask,a,b):
    use=mask&m['available'];d=m['z'][b,use].astype(np.float64)-m['z'][a,use].astype(np.float64)
    known=mask&(m['y']>=0);p=m['pred'][a];q=m['pred'][b]
    return dict(delta_raw_logit=stats(d),delta_D_E_correct_signed_margin=stats(d*np.where(m['y'][use]==1,1.,-1.)) if np.all(m['y'][use]>=0) else None,
        class_flips={an+'->'+bn:int(np.count_nonzero(mask&(p==av)&(q==bv))) for an,av in [('D',0),('E',1),('U',-1)] for bn,bv in [('D',0),('E',1),('U',-1)]},
        recovered_known=int(np.count_nonzero(known&(p!=m['y'])&(q==m['y']))),damaged_known=int(np.count_nonzero(known&(p==m['y'])&(q!=m['y']))),
        finite_available_N=int(use.sum()),inputU_retained_N=int(np.count_nonzero(mask&~m['available'])),
        uniform_offset_residual_RMSE=float(d.std()) if len(d) else None,exact_constant_shift=bool(len(d) and np.all(d==d[0])))

def cohort(m,name):
    mask=mask_group(m,name);out=dict(corpus=m['corpus'],group=name,N=int(mask.sum()),teacher_states=dict(Counter(m['state'][mask])),
        available=int((mask&m['available']).sum()),inputU=int((mask&~m['available']).sum()),states={})
    for state in ('ALL','D','E','S','U'):
        f=mask if state=='ALL' else mask&(m['state']==state);known=f&m['available']&(m['y']>=0)
        out['states'][state]=dict(N=int(f.sum()),models={name:dict(binary_metrics=metrics(m,m['pred'][i],f),
            finite_available_logit=stats(m['z'][i,f&m['available']]),fixed_signed_AUC_E1_vs_D0=auc(m['y'][known],m['z'][i,known]),
            AUC_available_known_N=int(known.sum()),full_known_N=int(np.count_nonzero(f&(m['y']>=0)))) for i,name in enumerate(MODELS)},
            comparisons={MODELS[i]+'->'+MODELS[j]:contrast(m,f,i,j) for i,j in PAIRS})
    return out

def make_domain(name,packs,rows,public=False):
    n=len(rows);a=packs[0]
    for pack in packs:
        assert pack['raw_head_logit_before_U'].shape==(n,) and pack['raw_head_logit_before_U'].dtype==np.float32
        assert np.array_equal(pack['global_index'],np.arange(n)) and pack['global_index'].dtype.kind in 'iu'
        assert np.array_equal(np.isfinite(pack['raw_head_logit_before_U']),pack['head_forwarded'])
        for key in a:
            if key not in ('raw_head_logit_before_U','query_logit','prediction'):
                assert key in pack and byteeq(a[key],pack[key]),(name,key)
        pred=np.where(pack['query_available'],(pack['raw_head_logit_before_U']>=0).astype(np.int8),-1)
        assert byteeq(pred.astype(np.int8),pack['prediction'])
        if 'query_logit' in pack:
            expected=pack['raw_head_logit_before_U'].copy();expected[~pack['query_available']]=np.nan
            assert np.array_equal(expected,pack['query_logit'],equal_nan=True)
    states=[];bands=[];errors=[]
    for i,r in enumerate(rows):
        assert r['native_slot']==i%8
        q=np.asarray(r['query_xy'] if r['query_xy'] is not None else [np.nan,np.nan],np.float64)
        assert np.array_equal(a['query_xy'][i],q,equal_nan=True)
        L=r['L'] if public else r['roi_side'];assert (L is None and np.isnan(a['L'][i])) or L==a['L'][i]
        if public:
            assert r['global_index']==i and r['frame_index']==i//8 and r['id']==str(a['frame_id'][i]) and r['source_id']==str(a['source_id'][i])
            assert r['id']+':native'+str(i%8)==str(a['query_id'][i])
            assert r['original_input_valid']==bool(a['query_available'][i])==bool(a['nearest_valid'][i])
            state=r['semantic_state'];ref=r['source_reference_xy'];err=float(np.linalg.norm(q-ref)) if state in 'DE' and ref is not None else None
            bands.append('UNCERTIFIED_REFERENCE' if err is None else 'LE3_PX' if err<=3 else 'GT3_LE8_PX' if err<=8 else 'GT8_LE20_PX' if err<=20 else 'GT20_PX')
        else:
            assert r['scene_index']==i//8 and r['base_index']==i//32 and r['convention']=='CF_PRIMARY_DECLARED'
            assert r['query_id']==str(a['occurrence_id'][i])
            state=r['teacher_state'];err=r['error_px'];bands.append(r['position_band'])
            assert state==str(a['raw_teacher_state_CF'][i]) and r['correspondence_boundary']==bool(a['convention_boundary'][i])
        assert r['split']==str(a['split'][i]);states.append(state);errors.append(err)
    state=np.asarray(states);y=np.where(state=='D',0,np.where(state=='E',1,-1))
    if not public:assert np.array_equal(y,a['external_target_CF'])
    m=dict(corpus=name,N=n,rows=rows,q=a['query_xy'],L=a['L'],available=a['query_available'],forwarded=a['head_forwarded'],
        state=state,y=y,native=np.arange(n)%8,band=np.asarray(bands),errors=errors,packs=packs,
        fold=np.array([('TRAIN' if r['split']=='TRAIN_LAB' else 'VALIDATION') if public else r['split'] for r in rows]),
        variant=np.asarray(['ORIGINAL_REAL' if public else r['variant'] for r in rows]),
        boundary=np.zeros(n,bool) if public else a['convention_boundary'])
    for k in ('foreign_family','material','aspect','geometry_family'):
        m[k]=np.asarray(['NOT_CERTIFIED_SOURCE_FACTOR' if public else r['factors'][k] for r in rows])
    m['z']=np.stack([p['raw_head_logit_before_U'] for p in packs]);m['pred']=np.stack([p['prediction'] for p in packs])
    return m

def analytic(m):
    train=(m['fold']=='TRAIN')&m['available']&(m['y']>=0);assert train.sum()==43
    out={}
    for j,name in enumerate(MODELS):
        rec={};grad=[]
        for s,y in [('D',0),('E',1)]:
            ix=np.flatnonzero(train&(m['y']==y));z=m['z'][j,ix].astype(np.float64)
            sig=1/(1+np.exp(-z));g=sig-y;weighted=.5*g/len(z);loss=np.logaddexp(0,z)-y*z
            rec[s]=dict(N=len(z),raw_logit=stats(z),sigmoid_E=stats(sig),mean_BCE=float(loss.mean()),unweighted_derivative_sum=float(g.sum()),
                unweighted_derivative_mean=float(g.mean()),classbalanced_bias_derivative_sum=float(weighted.sum()),classbalanced_derivative_per_row=stats(weighted),indices=ix.tolist())
            grad.append(float(weighted.sum()))
        total=sum(grad);d,e=map(abs,grad)
        out[name]=dict(classes=rec,balanced_BCE=.5*(rec['D']['mean_BCE']+rec['E']['mean_BCE']),common_additive_bias_dloss_dz_sum=total,
            descent_direction_for_common_bias='NEGATIVE_FAVORS_D' if total>0 else 'POSITIVE_FAVORS_E' if total<0 else 'ZERO',
            D_to_E_absolute_class_bias_push_ratio=divide(d,e),D_share_absolute_class_bias_push=divide(d,d+e))
    return out

def write(path,a):
    assert not path.exists(),path
    path.write_text(json.dumps(a,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def main():
    assert sha(SEAL)==EXPECTED;seal=read(SEAL);assert seal['completed'] and seal['PASS']
    for b in seal['inputs']:checked(b)
    result=read(checked(seal['results']));assert result['selected_model'] is None and result['all_rows']==27136
    control=read(result['source_control']['path']);checked(result['source_control'])
    protocol=read(checked(control['protocol']));limits=read(checked(protocol['fixed_public_pools_and_native35']))['native35_limits']
    assert len(limits)==35
    armnames=list(control['arms']);fits=[read(checked(control['arms'][a]['fit'])) for a in armnames]
    reports=[read(checked(f['source_results'])) for f in fits]
    domains={};input_parity={}
    for name,n,suffix,source in zip(('OLD_NATIVE','NEW_NATIVE'),(6144,20480),('OLD6144','NEW20480'),protocol['native']):
        allGT=lines(checked(source['GT']));assert len(allGT)==n*2
        gt=[r for r in allGT if r['convention']=='CF_PRIMARY_DECLARED'];assert len(gt)==n
        packs=[packet(source['baseline'])]+[packet(f['outputs']['FINAL_'+suffix+'_SCORES.npz']) for f in fits]
        domains[name]=make_domain(name,packs,gt)
        for f in fits:
            initial=packet(f['outputs']['INITIAL_'+suffix+'_SCORES.npz'])
            assert set(initial)==set(packs[0]) and all(byteeq(initial[k],packs[0][k]) for k in initial)
        input_parity[name]=dict(all_original_rows=n,all_initial_packet_keys_and_bytes_exact_C32=True,availability=int(packs[0]['query_available'].sum()),
            inputU=int((~packs[0]['query_available']).sum()),raw_nonfinite=int((~packs[0]['head_forwarded']).sum()))
    prows=lines(checked(protocol['public_labels']))
    pubpacks=[packet(fits[0]['outputs']['INITIAL_PUBLIC512_SCORES.npz'])]+[packet(f['outputs']['FINAL_PUBLIC512_SCORES.npz']) for f in fits]
    pother=packet(fits[1]['outputs']['INITIAL_PUBLIC512_SCORES.npz'])
    assert set(pother)==set(pubpacks[0]) and all(byteeq(pother[k],pubpacks[0][k]) for k in pother)
    pub=make_domain('PUBLIC_NATIVE_WEAK',pubpacks,prows,public=True);domains[pub['corpus']]=pub
    assert pub['forwarded'].all() and pub['available'].sum()==463 and set(np.flatnonzero((pub['y']>=0)&~pub['available']))=={402,434}
    input_parity[pub['corpus']]=dict(all_original_rows=512,all_initial_packet_keys_and_bytes_exact_across_arms=True,availability=463,inputU=49,raw_nonfinite=0)
    # Full source ledger: IDs/q/L/fold/state/S-U/availability/logits/prediction/deltas.
    ledger=lines(checked(seal['allrow_ledger']));assert len(ledger)==27136
    cursor=0
    for m in domains.values():
        for i,r in enumerate(m['rows']):
            row=ledger[cursor];cursor+=1
            same(row['corpus'],m['corpus']);same(row['index'],i);same(row['id'],r.get('query_id',r.get('id')))
            same(row['fold'],str(m['fold'][i]));same(row['original_split'],r['split']);same(row['native_slot'],i%8)
            same(row['variant'],str(m['variant'][i]));same(row['teacher_state'],str(m['state'][i]));same(row['target_D0_E1_Uminus1'],int(m['y'][i]))
            same(row['query_xy'],[scalar(v) for v in m['q'][i]]);same(row['L'],scalar(m['L'][i]))
            same(row['query_available'],bool(m['available'][i]));same(row['head_forwarded'],bool(m['forwarded'][i]))
            same(row['position_band'],str(m['band'][i]));same(row['historical_area_disagreement'],bool(m['boundary'][i]));same(row['error_px'],m['errors'][i])
            same(row['factors'],{k:str(m[k][i]) for k in ('foreign_family','material','aspect','geometry_family')})
            for j,name in enumerate(MODELS):
                z=float(m['z'][j,i]);expected=dict(raw_logit=scalar(z),prediction_D0_E1_Uminus1=int(m['pred'][j,i]),
                    correct_signed_margin=scalar(z*(1 if m['y'][i]==1 else -1)) if m['y'][i]>=0 else None)
                if m is pub and m['fold'][i]=='TRAIN' and m['available'][i] and m['y'][i]>=0:
                    expected['public_balanced_logit_derivative']=.5*((1/(1+math.exp(-z)))-int(m['y'][i]))/(35 if m['y'][i]==0 else 8)
                same(row['models'][name],expected,'allrow/'+str(cursor)+'/'+name)
            for a,b in PAIRS:
                expected=dict(delta_raw_logit=scalar(m['z'][b,i]-m['z'][a,i]),old_prediction=int(m['pred'][a,i]),new_prediction=int(m['pred'][b,i]))
                same(row['comparisons'][MODELS[a]+'->'+MODELS[b]],expected)
    # All 441 predeclared cohort reports, including S/U rows and every comparison.
    groups=lines(checked(seal['fixed_cohort_reports']));assert len(groups)==441==result['fixed_group_records']
    for record in groups:same(cohort(domains[record['corpus']],record['group']),record,record['corpus']+'/'+record['group'])
    headlines={k:cohort(domains[k.split('::')[0]],k.split('::')[1]) for k in result['fold_headlines']}
    same(headlines,result['fold_headlines'])
    # Gate values are independently rebuilt from primary CF truth, with full U denominator.
    gate_results={};base=read(checked(protocol['C32_source_results']));missingN=[]
    for j,name in enumerate(MODELS):
        gg={}
        for key,fixed in limits.items():
            prefix='NEW_NATIVE::' if key.startswith('NEW_NATIVE::') else 'NATIVE::'
            m=domains['NEW_NATIVE' if prefix=='NEW_NATIVE::' else 'OLD_NATIVE']
            group,state=key[len(prefix):].rsplit('::',1);mask=mask_group(m,group)&(m['state']==state);n=int(mask.sum())
            value=float(np.count_nonzero(m['pred'][j,mask]==int(state=='E'))/n)
            got=dict(N=n,value=value,limit=fixed['limit'],PASS=value>=fixed['limit'])
            same(got,result['native35_exact_replay'][name]['gates'][key],name+'/'+key)
            historical=base['gates'][key] if j==0 else reports[j-1]['native']['gates'][key]
            for k in ('value','limit','PASS'):same(got[k],historical[k])
            if 'N' in historical:same(n,historical['N'])
            else:
                assert j==0 and prefix=='NATIVE::';same(n,sum(base['native_CF'][group]['CM_D_E_rows_D_E_U_columns'][int(state=='E')]))
                missingN.append(key)
            gg[key]=got
        gate_results[name]=dict(passed=sum(x['PASS'] for x in gg.values()),total=35,gates=gg)
        same(gate_results[name],result['native35_exact_replay'][name])
    analytic_result=analytic(pub);same(analytic_result,result['public_initial_and_final_scalar_gradient']['models'])
    # All same-base/canonical native pairs: four variants are retained, never same-q claimed.
    pairrows=lines(checked(seal['all_native_pair_ledger']));assert len(pairrows)==6656
    paircount=0
    for m in (domains['OLD_NATIVE'],domains['NEW_NATIVE']):
        for baseindex in range(m['N']//32):
            for slot in range(8):
                row=pairrows[paircount];paircount+=1;ix=np.array([baseindex*32+slot+8*v for v in range(4)])
                same(row['corpus'],m['corpus']);same(row['base_index'],baseindex);same(row['native_slot'],slot);same(row['indices'],ix.tolist())
                same(row['teacher_states'],m['state'][ix].tolist());same(row['canonical_index'],m['rows'][int(ix[0])]['canonical_index'])
                same(row['fold'],str(m['fold'][ix[0]]));known=bool(m['state'][ix[1]]=='E' and m['state'][ix[2]]=='D')
                same(row['known_COVER_E_NEAR_D'],known);same(row['both_available'],bool(m['available'][ix[[1,2]]].all()))
                same(row['q_movement_COVER_NEAR_px'],scalar(np.linalg.norm(m['q'][ix[1]]-m['q'][ix[2]])))
                same(row['L_abs_change_px'],scalar(abs(m['L'][ix[1]]-m['L'][ix[2]])))
                for j,name in enumerate(MODELS):
                    same(row['models'][name],dict(raw_four=[scalar(z) for z in m['z'][j,ix]],pred_four=m['pred'][j,ix].tolist(),
                        E_minus_D_logit_gap=scalar(m['z'][j,ix[1]]-m['z'][j,ix[2]]),both_correct=bool(known and m['pred'][j,ix[1]]==1 and m['pred'][j,ix[2]]==0)))
    # Independent root scalar receipt: all43 rows, analytic sums and public state shifts.
    root=read(V/'ROOT_PUBLIC_SCALAR_DIAGNOSIS.json')
    initial=analytic_result['INITIAL_C32'];same(root['initial_public_TRAIN43_scalar_bias_derivative']['D'],initial['classes']['D']['classbalanced_bias_derivative_sum'])
    same(root['initial_public_TRAIN43_scalar_bias_derivative']['E'],initial['classes']['E']['classbalanced_bias_derivative_sum'])
    same(root['initial_public_TRAIN43_scalar_bias_derivative']['total'],initial['common_additive_bias_dloss_dz_sum'])
    same(root['initial_public_TRAIN43_BCE']['total'],initial['balanced_BCE'])
    for rr in root['initial_TRAIN43_all_rows']:
        i=rr['global_index'];z=float(pub['z'][0,i]);y=int(pub['y'][i]);weight=.5/(35 if y==0 else 8)
        same(rr['initial_z'],z);same(rr['weighted_BCE'],weight*float(np.logaddexp(0,z)-y*z));same(rr['weighted_dL_dz'],weight*(1/(1+math.exp(-z))-y))
    for group,rr in root['C34_minus_C33_public_valid_shifts'].items():
        fold,state=group.split('::');mask=(pub['fold']==('TRAIN' if fold=='TRAIN_LAB' else 'VALIDATION'))&(pub['state']==state)&pub['available']
        d=pub['z'][2,mask].astype(np.float64)-pub['z'][1,mask].astype(np.float64)
        same(rr,dict(N=len(d),mean=float(d.mean()),min=float(d.min()),max=float(d.max()),median=float(np.median(d))))
    # Separately preserve and inspect the two schema-only first failures/repairs.
    qprov=S/'provenance/FIRST_C32_OPTIONAL_N_SCHEMA_FAILURE'
    rprov=V/'provenance/ROOT_FIRST_SCALAR_GUARD_FAILURE'
    query_failure=read(qprov/'FAILED_EXECUTION.json');root_failure=read(rprov/'FAILURE_AND_SCHEMA_DIAGNOSIS.json')
    assert not query_failure['output_ledger_summary_seal_created'] and root_failure['result_files_written']==0
    qold=(qprov/'analyze_saved_source.py').read_text();qnew=(S/'analyze_saved_source.py').read_text()
    rold=(rprov/'root_public_scalar_diagnosis.py').read_text();rnew=(V/'root_public_scalar_diagnosis.py').read_text()
    for path in (qprov/'analyze_saved_source.py',S/'analyze_saved_source.py',rprov/'root_public_scalar_diagnosis.py',V/'root_public_scalar_diagnosis.py'):bind(path)
    qdiff=''.join(difflib.unified_diff(qold.splitlines(True),qnew.splitlines(True)))
    rdiff=''.join(difflib.unified_diff(rold.splitlines(True),rnew.splitlines(True)))
    assert "gate['N']=sum(base['native_CF'][group]" in qdiff and 'C32_original_native_gate_optional_N_adapter' in qdiff
    assert 'nearest_valid' in rdiff and "a['head_forwarded'].all()" in rdiff
    repairs=dict(query_optional_N=dict(first_failure=bind(qprov/'FAILED_EXECUTION.json'),original_code=bind(qprov/'analyze_saved_source.py'),
        final_code=bind(S/'analyze_saved_source.py'),missing_C32_N_count=len(missingN),missing_C32_N_keys=missingN,
        independently_checked_N_from_primary_GT_and_original_CM=True,value_limit_PASS_unchanged=True),
        root_execution_vs_validity=dict(first_failure=bind(rprov/'FAILURE_AND_SCHEMA_DIAGNOSIS.json'),original_code=bind(rprov/'root_public_scalar_diagnosis.py'),
        final_code=bind(V/'root_public_scalar_diagnosis.py'),all512_forwarded_but_only463_valid=True,original49U_preserved=True))
    nativeval=[]
    for m in (domains['OLD_NATIVE'],domains['NEW_NATIVE']):
        f=(m['fold']=='VALIDATION')&m['available']&(m['y']>=0);nativeval.append((m['y'][f],m['z'][:,f]))
    combined_auc={name:auc(np.r_[nativeval[0][0],nativeval[1][0]],np.r_[nativeval[0][1][j],nativeval[1][1][j]]) for j,name in enumerate(MODELS)}
    summary=dict(completed=True,PASS=True,source27136_all_ID_q_L_fold_state_inputU_and_scores_rechecked=True,
        input_parity=input_parity,all_441_fixed_cohort_records_recomputed=True,all_6656_four_variant_pair_records_recomputed=True,
        native35=gate_results,analytic_public43=analytic_result,fold_headlines=headlines,combined_native_VAL_AUC_available_known=combined_auc,
        repairs_separate=repairs,selected_model=None,full_goal_solved=False,
        interpretation=['Public C34-C33 TRAIN D mean is negative while E mean is positive; observed effect is not a uniform negative bias shift.',
            'Old-native VAL AUC changes slightly upward but new-native VAL AUC falls versus C33; no consistent broad-native discrimination improvement.',
            'Initial scalar bias derivative is descriptive, not attribution of full shared-parameter/AdamW/clipping dynamics.',
            'No new ranking fit is recommended by this audit; weak partial roles, limited cardboard E coverage and D67 floor-query concentration remain source evidence.'],
        calls=dict(producer_helper_import_or_call=0,model_fit_forward_backward=0,private_current_or_forbidden_phase_access=0,feature_checkpoint_or_RGB_body=0,labels_modified_or_selected=0))
    write(H/'RECALCULATED_SOURCE_DIAGNOSIS.json',summary)
    # Verify that every explicitly read source artifact remains unchanged.
    original=list(READS.values())
    for b in original:assert sha(b['path'])==b['sha256']
    write(H/'REVIEW.json',dict(completed=True,PASS=True,code=bind(__file__),diagnosis_seal=bind(SEAL),
        source_control=result['source_control'],result=bind(H/'RECALCULATED_SOURCE_DIAGNOSIS.json'),
        rows=27136,native_pairs=6656,cohort_records=441,native_gates=35,all_original_source_bindings_unchanged=True,
        source_inputs=original,selected_model=None,calls=summary['calls']))
    print(json.dumps(dict(completed=True,PASS=True,receipt=bind(H/'REVIEW.json'),input_parity=input_parity,
        native35_passed={k:v['passed'] for k,v in gate_results.items()},combined_native_VAL_AUC=combined_auc),indent=2))

if __name__=='__main__':main()
