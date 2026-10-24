"""Scalar count reconciliation for two saved source visibility baselines only."""
from pathlib import Path
from collections import Counter
import csv,hashlib,json,math
import numpy as np

ROOT=Path(__file__).resolve().parents[1];BASE=ROOT.parent;HERE=ROOT/'source_audit'
GT_KEYS=('sample_id','source','group_id','image_sha256','global_id','global_source','global_image_sha256','target_index','dense_allowed','dense_valid','dense_exact_state')
SAM_KEYS=('sample_id','source','group_id','global_id','native_valid','exact_logit','exact_visible')
MOBILE_KEYS=('sample_id','source','group_id','image_sha256','center_native_valid','center_allowed','GT_state','raw_SAM_visible_probability')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p):return dict(path=str(p),sha256=sha(p))
def arrays(path,keys):
    with np.load(path,allow_pickle=False)as z:return{k:z[k]for k in keys}
def ratio(a,b):return a/b if b else None
def arrayhash(a):
    h=hashlib.sha256((a.dtype.str+':'+repr(a.shape)).encode());h.update(np.ascontiguousarray(a).tobytes());return h.hexdigest()
def compare(actual,published,where,checked):
    if isinstance(actual,dict):
        for k,v in actual.items():compare(v,published[k],where+'.'+k,checked)
    elif isinstance(actual,list):
        assert len(actual)==len(published),where
        for i,(a,p)in enumerate(zip(actual,published)):compare(a,p,where+f'[{i}]',checked)
    elif actual is None:assert published is None,where
    elif isinstance(actual,(int,float)):
        assert abs(actual-published)<1e-14,(where,actual,published);checked.append(where)
    else:assert actual==published,(where,actual,published)

def one(source,mask,valid,state,allowed,pred):
    ids=[i for i,s in enumerate(source)if s==mask];conf=[[0,0,0]for _ in range(3)];unknown=Counter();allvalid=Counter();invalid=Counter()
    bp=Counter();marginal_unknown=0
    for i in ids:
        p=int(pred[i]);col=0 if p==1 else 1 if p==0 else 2
        if not valid[i]:invalid[['VISIBLE','ABSENT','U'][col]]+=1;continue
        allvalid[['VISIBLE','ABSENT','U'][col]]+=1
        if state[i]>=0:conf[int(state[i])][col]+=1
        else:unknown[['VISIBLE','ABSENT','U'][col]]+=1
        a=int(allowed[i]);truth=1 if a==1 else 0 if not(a&1)else -1
        if truth<0:marginal_unknown+=1
        elif p<0:bp['U']+=1
        elif truth==1:bp['TP'if p else'FN']+=1
        else:bp['FP'if p else'TN']+=1
    def binary(tp,fp,fn,tn,positive,negative,U):
        return dict(TP=tp,FP=fp,FN=fn,TN=tn,N=positive+negative,positive=positive,negative=negative,U=U,
            precision=ratio(tp,tp+fp),recall=ratio(tp,positive),nonvisible_recall=ratio(tn,negative))
    pos=sum(conf[0]);neg=sum(conf[1])+sum(conf[2]);bn=binary(conf[0][0],conf[1][0]+conf[2][0],conf[0][1],conf[1][1]+conf[2][1],pos,neg,sum(r[2]for r in conf))
    bpn=bp['TP']+bp['FN'];bnn=bp['FP']+bp['TN'];binary_partial=binary(bp['TP'],bp['FP'],bp['FN'],bp['TN'],bpn,bnn,bp['U']);binary_partial['unknown_marginal']=marginal_unknown
    def group(c):
        n=sum(conf[c]);return dict(N=n,nonvisible=conf[c][1],visible=conf[c][0],U=conf[c][2],recall=ratio(conf[c][1],n))
    V=sum(bool(valid[i])for i in ids)
    return dict(total_rows=len(ids),native_valid=V,native_invalid=len(ids)-V,known_class_GT=pos+neg,UNKNOWN_class_GT=sum(unknown.values()),
        native_valid_all63=int(sum(valid[i]and allowed[i]==63 for i in ids)),informative_partial=int(sum(valid[i]and allowed[i]!=63 for i in ids)),
        confusion=dict(GT_rows=['V','E','O'],prediction_columns=['VISIBLE','ABSENT','U'],counts=conf),
        visible_binary_on_class_known=bn,known_E_to_binary_nonvisible=group(1),known_O_to_binary_nonvisible=group(2),
        UNKNOWN_class_predictions={k:unknown[k]for k in ['VISIBLE','ABSENT','U']},all_native_valid_predictions={k:allvalid[k]for k in ['VISIBLE','ABSENT','U']},
        invalid_predictions={k:invalid[k]for k in ['VISIBLE','ABSENT','U']},binary_known_partial_diagnostic=binary_partial)

def main():
    out=HERE/'INDEPENDENT_SOURCE_METRIC_AUDIT.json';assert not out.exists()
    report_path=ROOT/'source_evaluation/SAM21_FIXED_COMPARISON/SOURCE_RESULTS.json';report=json.loads(report_path.read_text());assert report['completed']and report['source_only']
    b=report['bindings'];gtpath=BASE/'query_visibility_controlled_real_20261022_v66'/b['GT_cache_prior_binding']['path']
    assert gtpath.stat().st_size==b['GT_cache_prior_binding']['bytes']
    # Do not hash/decode the whole mixed feature+GT archive: only the whitelist.
    d=arrays(gtpath,GT_KEYS);a=d['dense_allowed'][:,4,4];valid=d['dense_valid'][:,4,4];state=d['dense_exact_state'][:,4,4]
    assert len(a)==8590 and len(d['global_id'])==1091 and len(set(d['sample_id']))==8590 and valid.sum()==8584
    assert np.array_equal(d['global_source'][d['target_index']],d['source'])and np.array_equal(d['global_image_sha256'][d['target_index']],d['image_sha256'])
    for i,(aa,v,y)in enumerate(zip(a,valid,state)):
        groups=[bool(int(aa)&bits)for bits in [1,6,56]];expected=groups.index(True)if sum(groups)==1 and v else-1
        assert int(y)==expected
    source=list(d['source'].astype(str));models={};input_bindings=[]
    sam=arrays(b['SAM_saved_npz']['path'],SAM_KEYS);assert sha(b['SAM_saved_npz']['path'])==b['SAM_saved_npz']['sha256'];input_bindings.append(b['SAM_saved_npz'])
    for k in ['sample_id','source','group_id']:assert np.array_equal(sam[k],d[k])
    assert np.array_equal(sam['global_id'],d['global_id'][d['target_index']])and np.array_equal(sam['native_valid'],valid)
    assert np.isnan(sam['exact_logit'][~valid]).all()and np.isfinite(sam['exact_logit'][valid]).all()
    sam_pred=np.full(8590,-1,np.int8);sam_pred[valid]=(sam['exact_logit'][valid]>0).astype(np.int8);assert np.array_equal(sam_pred,sam['exact_visible'])
    models['SAM21_BASE_PLUS_FROZEN']=sam_pred
    oldbind=b['baselines']['V66_CONTROLLED_PLUS_PALLOX']['output'];assert sha(oldbind['path'])==oldbind['sha256'];input_bindings.append(oldbind)
    mobile=arrays(oldbind['path'],MOBILE_KEYS)
    for k in ['sample_id','source','group_id','image_sha256']:assert np.array_equal(mobile[k],d[k])
    assert np.array_equal(mobile['center_native_valid'],valid)and np.array_equal(mobile['center_allowed'],a)and np.array_equal(mobile['GT_state'][:,0],state)
    probabilities=mobile['raw_SAM_visible_probability'];assert probabilities.shape==(8590,2)and np.isfinite(probabilities).all()
    mob_pred=np.full(8590,-1,np.int8);mob_pred[valid]=(probabilities[valid,0]>.5).astype(np.int8);models['MOBILE_SAM_RAW']=mob_pred
    with(ROOT/'source_inputs/QUERY_ID_MAP.csv').open()as f:mapping=list(csv.DictReader(f))
    assert len(mapping)==8590
    for i,q in enumerate(mapping):
        assert int(q['joined_query_index'])==i
        for k in ['sample_id','source','group_id','image_sha256']:assert q[k]==str(d[k][i])
        assert q['global_id']==str(d['global_id'][d['target_index'][i]])
        assert(q['native_input_valid']=='True')==bool(valid[i])
    checked=[];records=[]
    for model,pred in models.items():
        assert(pred[~valid]==-1).all()
        for s in ['SYNTH','ORIGINAL_YCB','CONTROLLED_YCB','PALLOX']:
            result=one(source,s,valid,state,a,pred);published=report['models'][model]['per_source'][s]['exact'];compare(result,published,model+'.'+s+'.exact',checked)
            records.append(dict(model=model,source=s,metrics=result))
    invalid_ids=d['sample_id'][~valid].tolist();assert len(invalid_ids)==6 and all(str(x).startswith('PALLOX')for x in invalid_ids)
    flat=[]
    for r in records:
        m=r['metrics'];z=dict(model=r['model'],source=r['source'],native_valid=m['native_valid'],native_invalid=m['native_invalid'],known_class_GT=m['known_class_GT'],UNKNOWN_class_GT=m['UNKNOWN_class_GT'],
            **m['visible_binary_on_class_known'],known_E_N=m['known_E_to_binary_nonvisible']['N'],known_E_absent=m['known_E_to_binary_nonvisible']['nonvisible'],known_E_binary_absence_recall=m['known_E_to_binary_nonvisible']['recall'])
        flat.append(z)
    cp=HERE/'INDEPENDENT_SOURCE_METRIC_AUDIT.csv'
    with cp.open('x',newline='')as f:w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    result=dict(PASS=True,schema='V68_SAVED_SOURCE_TWO_BASELINE_EXACT_SCALAR_RECONCILIATION_V1',scope='SAM21 and frozen rawMobileSAM exact support only; four sources; no reanalysis of otherheads/radius2/strict25.',
        published_results=ref(report_path),code=ref(Path(__file__)),saved_score_input_bindings=input_bindings,GT_archive_prior_binding=b['GT_cache_prior_binding'],
        GT_archive_whole_bytes_not_hashed_to_avoid_feature_reads=True,GT_array_whitelist=list(GT_KEYS),SAM_array_whitelist=list(SAM_KEYS),Mobile_array_whitelist=list(MOBILE_KEYS),
        GT_decoded_array_hashes={k:dict(shape=list(v.shape),dtype=str(v.dtype),sha256=arrayhash(v))for k,v in d.items()},
        canonical_rows_joined=8590,targets=1091,source_counts=dict(Counter(source)),native_valid=8584,native_invalid=6,invalid_same6IDs=invalid_ids,invalid_all_U_both_models=True,
        exact_class_state_rebuilt_from_compatible_bits=True,scalar_numeric_fields_compared=len(checked),mismatches=0,source_results=records,CSV=ref(cp),
        fixed_decisions=dict(SAM21='native float32 exactlogit>0 strict',MobileSAM='saved rawSAM pVisible>.5 strict; never >=.5'),
        ties=dict(SAM21_exact_zero=int((sam['exact_logit'][valid]==0).sum()),MobileSAM_exact_half=int((probabilities[valid,0]==.5).sum())),
        denominator_caveat='Primary P/R uses knownV/E/O class subset. Compatible-set binary-known diagnostic has a wider denominator for original/controlled YCB; class-UNKNOWN is not automatically unknownvisibility or a paddednegative.',
        mechanism_caveat='KnownE binaryabsent counts are nonvisibility on E GT, not predictedexternal cause precision/recall. Saved masks cannot establish semanticcorner/self success.',
        source_HOLD_and_all_old_selection_thresholds_preserved=True,source_model_RGB_current_feature_weight_reads_or_new_forward_training_PnP=0)
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print('SOURCE_METRIC_RECONCILIATION_PASS',sha(out),'numericfields',len(checked));print(json.dumps(flat,indent=2))

if __name__=='__main__':main()
