"""Independently reconcile two frozen exact score sets; numpy/scalars only.

Does not import the evaluator, torch, any model, or any feature/RGB reader.
"""
from pathlib import Path
from collections import Counter
import csv,hashlib,json,math
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
V66=ROOT.parent/'query_visibility_controlled_real_20261022_v66'
OUT=ROOT/'source_evaluation/OWNERSHIP_FROZEN_COMPARISON'
MODELS=['V70_OWNERSHIP_LOO_PAIR_V1','V61_FROZEN']
SOURCES=['SYNTH','ORIGINAL_YCB','CONTROLLED_YCB','PALLOX']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def read(p):return json.loads(Path(p).read_text())
def csvrows(p):
    with Path(p).open(newline='')as f:return list(csv.DictReader(f))
def fraction(a,b):return a/b if b else None
def numpy_scalar(value):
    if isinstance(value,np.generic):return value.item()
    raise TypeError(type(value).__name__)
def boolean(s):
    if s in [True,'True','true','1',1]:return True
    if s in [False,'False','false','0',0]:return False
    raise ValueError(s)
def load(path,keys):
    with np.load(path,allow_pickle=False)as z:return {k:z[k]for k in keys}

report_path=OUT/'SOURCE_RESULTS.json';decision_path=ROOT/'SOURCE_ADMISSION_DECISION.json'
report=read(report_path);decision=read(decision_path);refs=report['bindings'];inputs=[]
for b in [decision['source_results'],decision['prefit_policy'],decision['source_selection'],
          refs['primary'],refs['v61'],refs['VAL_pair_index'],refs['strict_reference']]:
    assert sha(b['path'])==b['sha256'];inputs.append(b)
assert Path(decision['source_results']['path']).resolve()==report_path
policy=read(decision['prefit_policy']['path'])
assert policy['frozen_before_V70_fit_or_model_outputs']is True
GTpath=V66/refs['GT_prior_binding']['path'];assert GTpath.stat().st_size==refs['GT_prior_binding']['bytes']
GTkeys=['sample_id','source','group_id','image_sha256','global_id','target_index',
        'dense_allowed','dense_valid','dense_exact_state']
g=load(GTpath,GTkeys);n=len(g['sample_id']);assert n==8590
valid=g['dense_valid'][:,4,4];allowed=g['dense_allowed'][:,4,4]
assert int(valid.sum())==8584 and (allowed[~valid]==63).all()
native_pairs=[(0,0),(1,0),(1,1),(2,0),(2,1),(2,2)]
truth=[]
for a,v in zip(allowed,valid):
    opts={native_pairs[k][0]for k in range(6)if int(a)&(1<<k)}
    truth.append(next(iter(opts))if v and len(opts)==1 else -1)
truth=np.array(truth,np.int8);assert np.array_equal(truth,g['dense_exact_state'][:,4,4])
global_id=g['global_id'][g['target_index']]
assert len(set(g['sample_id']))==8590 and len(set(g['global_id']))==1091

newkeys=['sample_id','source','group_id','image_sha256','global_id','native_valid','allowed','exact_GT',
         'raw_logit_visible_given_VE','raw_probability_visible_given_VE']
v61keys=['sample_id','source','group_id','image_sha256','global_id','native_valid','logit','pV']
d70=load(refs['primary']['path'],newkeys);d61=load(refs['v61']['path'],v61keys)
for d in [d70,d61]:
    for k in ['sample_id','source','group_id','image_sha256']:assert np.array_equal(d[k],g[k])
    assert np.array_equal(d['global_id'],global_id)and np.array_equal(d['native_valid'],valid)
assert np.array_equal(d70['allowed'],allowed)and np.array_equal(d70['exact_GT'],truth)
prob={MODELS[0]:d70['raw_probability_visible_given_VE'],MODELS[1]:d61['pV']}
logit={MODELS[0]:d70['raw_logit_visible_given_VE'],MODELS[1]:d61['logit']}
prediction={}
for m in MODELS:
    assert np.isfinite(prob[m][valid]).all()
    pr=np.full(n,-1,np.int8);pr[valid]=prob[m][valid]>=.5;prediction[m]=pr
    assert np.array_equal(pr[valid],(logit[m][valid]>=0).astype(np.int8))

per_query_path=OUT/'PER_QUERY.csv';rows=csvrows(per_query_path)
selected={m:[]for m in MODELS}
for r in rows:
    if r['model']in selected:selected[r['model']].append(r)
for m,rr in selected.items():
    assert len(rr)==8590
    for i,r in enumerate(rr):
        assert int(r['index'])==i and r['sample_id']==g['sample_id'][i]and r['source']==g['source'][i]
        assert r['global_id']==global_id[i]and r['image_sha256']==g['image_sha256'][i]
        assert boolean(r['native_valid'])==bool(valid[i])and int(r['allowed'])==allowed[i]and int(r['exact_GT'])==truth[i]
        assert int(r['prediction'])==prediction[m][i]
        assert boolean(r['conditional_known_VE'])==bool(valid[i]and truth[i]in(0,1))
        if valid[i]:assert float(r['p_visible'])==prob[m][i]and float(r['logit'])==logit[m][i]
        else:assert r['status']=='U'and r['p_visible']==r['logit']==''

mismatches=[];comparisons=0;max_numeric_error=0.
def compare(got,expected,path):
    global comparisons,max_numeric_error
    comparisons+=1
    if isinstance(got,dict):
        for key,value in got.items():compare(value,expected[key],path+'/'+key)
    elif isinstance(got,list):
        if len(got)!=len(expected):mismatches.append(path+' length')
        else:
            for k,(a,b)in enumerate(zip(got,expected)):compare(a,b,path+'/'+str(k))
    elif isinstance(got,float):
        error=abs(got-expected)if expected is not None else float('inf');max_numeric_error=max(max_numeric_error,error)
        if error>1e-14:mismatches.append(dict(path=path,got=got,expected=expected))
    elif got!=expected:mismatches.append(dict(path=path,got=got,expected=expected))

def metrics(indices,pr):
    tp=fp=fn=tn=u=pos=neg=0
    for i in indices:
        isV=truth[i]==0;p=int(pr[i]);pos+=isV;neg+=not isV
        tp+=isV and p==1;fp+=not isV and p==1;fn+=isV and p!=1;tn+=not isV and p==0;u+=p<0
    return dict(N=len(indices),positive=int(pos),negative=int(neg),TP=int(tp),FP=int(fp),FN=int(fn),TN=int(tn),U=int(u),precision=fraction(tp,tp+fp),recall=fraction(tp,pos),nonvisible_recall=fraction(tn,neg))

counts={};conditional={};knownE={};unknown={};tables={};NLL={}
for m in MODELS:
    counts[m]={};conditional[m]={};knownE[m]={};unknown[m]={};tables[m]={};NLL[m]={}
    for source in SOURCES:
        source_indices=[i for i in range(n)if g['source'][i]==source]
        ix=[i for i in source_indices if valid[i]and truth[i]>=0]
        ve=[i for i in source_indices if valid[i]and truth[i]in(0,1)]
        counts[m][source]=metrics(ix,prediction[m]);conditional[m][source]=metrics(ve,prediction[m])
        e=[i for i in source_indices if valid[i]and truth[i]==1]
        absent=sum(prediction[m][i]==0 for i in e);vis=sum(prediction[m][i]==1 for i in e);u=sum(prediction[m][i]<0 for i in e)
        knownE[m][source]=dict(N=len(e),binary_nonvisible=int(absent),visible=int(vis),U=int(u),recall=fraction(absent,len(e)))
        tab=np.zeros((3,3),np.int64)
        for i in ix:tab[truth[i],0 if prediction[m][i]==1 else(1 if prediction[m][i]==0 else 2)]+=1
        tables[m][source]=tab.tolist()
        ui=[i for i in source_indices if valid[i]and truth[i]<0]
        unknown[m][source]={k:sum(int(prediction[m][i])==v for i in ui)for k,v in [('VISIBLE',1),('ABSENT',0),('U',-1)]}
        z=logit[m];loss=[]
        for i in ve:
            q=float(z[i]);loss.append(max(q,0)+math.log1p(math.exp(-abs(q)))-int(truth[i]==0)*q)
        NLL[m][source]=dict(N=len(ve),mean=sum(loss)/len(loss))
        dest=report['models'][m]['per_source'][source]
        compare(counts[m][source],dest['visible_binary_ALL_known_VEO'],m+'/'+source+'/ALLknownVEO')
        compare(conditional[m][source],dest['conditional_known_VE'],m+'/'+source+'/conditionalVE')
        compare(knownE[m][source],dest['known_E_to_binary_nonvisible'],m+'/'+source+'/knownEabsence')
        compare(tables[m][source],dest['class_known_confusion']['counts'],m+'/'+source+'/confusion')
        compare(unknown[m][source],dest['UNKNOWN_predictions'],m+'/'+source+'/UNKNOWN')
        compare(NLL[m][source],{k:dest['conditional_known_VE']['NLL'][k]for k in ['N','mean']},m+'/'+source+'/NLL')
    compare(sum(NLL[m][s]['mean']for s in SOURCES)/4,report['models'][m]['macro_source_knownVE_NLL'],m+'/macroNLL')

pairrows=csvrows(refs['VAL_pair_index']['path']);assert len(pairrows)==3200
pair_counts=Counter();all_transitions={m:np.zeros((3,3),np.int64)for m in MODELS};paired={m:{}for m in MODELS}
bykind={}
for r in pairrows:
    assert r['split']=='VAL';i,j=int(r['original_index']),int(r['controlled_index'])
    assert g['sample_id'][i]==r['original_sample_id']and g['sample_id'][j]==r['controlled_sample_id']
    assert truth[i]==int(r['original_exact_state'])and truth[j]==int(r['controlled_exact_state'])
    pair_counts[str(int(truth[i]))+'->'+str(int(truth[j]))]+=1
    bykind.setdefault(r['pair_type'],[]).append((i,j))
    for m in MODELS:
        a,b=prediction[m][i],prediction[m][j];all_transitions[m][0 if a==1 else(1 if a==0 else 2),0 if b==1 else(1 if b==0 else 2)]+=1
for m in MODELS:
    for kind,pairs in bykind.items():
        if kind not in ['V_TO_V','V_TO_E']:continue
        mat=np.zeros((3,3),np.int64);ov=cv=ca=joint=lost=gain=0
        for i,j in pairs:
            a,b=int(prediction[m][i]),int(prediction[m][j]);ov+=a==1;cv+=b==1;ca+=b==0
            joint+=a==1 and b==(1 if kind=='V_TO_V'else 0);lost+=a==1 and b==0;gain+=a==0 and b==1
            mat[0 if a==1 else(1 if a==0 else 2),0 if b==1 else(1 if b==0 else 2)]+=1
        d=dict(N=len(pairs),original_visible=int(ov),controlled_visible=int(cv),controlled_binary_nonvisible=int(ca),joint_correct=int(joint),
            original_visible_to_controlled_nonvisible=int(lost),original_nonvisible_to_controlled_visible=int(gain),
            controlled_visible_recall=fraction(cv,len(pairs))if kind=='V_TO_V'else None,
            known_E_to_binary_nonvisible_recall=fraction(ca,len(pairs))if kind=='V_TO_E'else None,
            prediction_transition_matrix=dict(counts=mat.tolist()))
        paired[m][kind]=d;compare(d,report['paired_original_controlled'][m][kind],m+'/'+kind)

refrows=csvrows(refs['strict_reference']['path']);lookup={str(s):i for i,s in enumerate(g['sample_id'])}
strict={m:{}for m in MODELS}
for s,N in [('ORIGINAL_YCB',714),('CONTROLLED_YCB',296)]:
    chosen=[lookup[r['sample_id']]for r in refrows if r['model']=='V65_BASELINE'and r['source']==s and boolean(r['strict25_V'])]
    assert len(chosen)==N and all(valid[i]and truth[i]==0 for i in chosen)
    for m in MODELS:
        tp=sum(prediction[m][i]==1 for i in chosen);strict[m][s]=dict(N=N,TP=int(tp),FN=N-int(tp),U=0,recall=fraction(tp,N))
        compare(strict[m][s],report['strict25_visible'][m][s],m+'/'+s+'/strict25V')

primary=MODELS[0];broad=[]
for s in SOURCES[1:]:
    for metric in ['precision','recall']:
        value=counts[primary][s][metric];broad.append(dict(category='public_ALL_known_VEO_visible',source=s,metric=metric,value=value,minimum=.8,passed=value is not None and value>=.8))
for s in ['CONTROLLED_YCB','PALLOX']:
    value=knownE[primary][s]['recall'];broad.append(dict(category='known_E_binary_nonvisible',source=s,metric='recall',value=value,minimum=.6,passed=value is not None and value>=.6))
checks=broad.copy()
for s,minimum in [('ORIGINAL_YCB',679),('CONTROLLED_YCB',278)]:
    value=strict[primary][s]['TP'];checks.append(dict(category='strict25_pixel_core',source=s,metric='TP',value=value,minimum=minimum,passed=value>=minimum))
compare(checks,report['source_diagnostic_checks'],'policy10')
bp=sum(c['passed']for c in broad);lp=sum(c['passed']for c in checks)
compare(dict(pass_=bp,fail=8-bp),dict(pass_=decision['broader_eligibility']['pass'],fail=decision['broader_eligibility']['fail']),'decision/broad8')
compare(dict(pass_=lp,fail=10-lp),dict(pass_=decision['legacy_diagnostics']['pass'],fail=decision['legacy_diagnostics']['fail']),'decision/legacy10')
assert bp==5 and lp==5
assert report['eligibility_policy_result']['exploratory_human_comparison_eligible']is False
assert decision['verdict']=='HOLD_NOT_ELIGIBLE_FOR_CURRENT_COMPARISON'and decision['current_inference_allowed']is False and decision['current223_new_inference']==0

# Preserve every bound source result/decision/score/label input throughout audit.
for b in inputs:assert sha(b['path'])==b['sha256']
result=dict(schema='V70_INDEPENDENT_ACTUAL_SAVED_METRIC_RECONCILIATION_V1',completed=True,status='PASS'if not mismatches else'FAIL',
 evaluator_functions_imported_or_reused=False,methods='Scalar confusion counts from independently projected allowed-bit GT and raw saved p>=.5; independent source/class denominators, pair loops, stable scalar logit NLL and fixed policy comparisons.',
 queries=8590,native_valid=8584,native_invalid_U=6,PER_QUERY_selected_rows=17180,models=MODELS,
 ALL_known_VEO=counts,conditional_known_VE=conditional,known_E_binary_nonvisible=knownE,UNKNOWN_predictions=unknown,
 confusion_V_E_O_by_VISIBLE_ABSENT_U=tables,conditional_NLL=NLL,
 paired_VAL=dict(rows=3200,GT_transitions=dict(pair_counts),models=paired,all3200_prediction_transitions={m:a.tolist()for m,a in all_transitions.items()}),
 strict25_visible=strict,policy_recomputed=dict(broader8_pass=int(bp),broader8_fail=8-int(bp),legacy10_pass=int(lp),legacy10_fail=10-int(lp),checks=checks,
 exploratory_human_comparison_eligible=False,current_inference_allowed=False,new_current223_inference=0),
 scalar_and_structure_comparisons=comparisons,maximum_numeric_difference=max_numeric_error,mismatches=mismatches,
 bindings=[bind(report_path),bind(per_query_path),bind(decision_path),bind(__file__)]+inputs,
 GT_cache_prior_immutable_binding=refs['GT_prior_binding'],GT_array_keys_opened=GTkeys,
 raw_feature_file_hash_repeated=False,feature_RGB_mask_weight_current_reads=0,new_model_forward_calls=0,training=0,PnP=0,prior_artifacts_modified=False)
out=ROOT/'source_audit/INDEPENDENT_SAVED_METRIC_RECONCILIATION.json';assert not out.exists()
out.write_text(json.dumps(result,indent=2,allow_nan=False,default=numpy_scalar)+'\n')
print(json.dumps(dict(status=result['status'],audit=bind(out),comparisons=comparisons,mismatches=mismatches,counts=counts,conditional=conditional[primary],broad8=[bp,8-bp]),indent=2,default=numpy_scalar))
assert not mismatches
