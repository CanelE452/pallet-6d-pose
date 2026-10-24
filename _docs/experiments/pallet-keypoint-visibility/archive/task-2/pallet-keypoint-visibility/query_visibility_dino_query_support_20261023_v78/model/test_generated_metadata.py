"""Full-sized artificial identity/GT join test; no source payload reads."""
from pathlib import Path
import argparse, hashlib, importlib.util, json
import numpy as np

R = Path(__file__).resolve().parents[1]

def b(p):
    return dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest())

def module(p, name):
    spec=importlib.util.spec_from_file_location(name,p)
    m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m

def main(output):
    assert not output.exists()
    T=module(R/'train_source_once.py','generated_v78_metadata_trainer')
    p=T.decisions()
    paired=T.imp(p['bindings']['V77_source_trainer'],'generated_v77_join_helpers')
    base=module(R.parent/'query_visibility_reference_local_support_20261023_v76/mixed_fit/train_mixed_once.py','generated_v76_join_admission')
    n,a=5203,4799
    source=np.array(['SYNTHETIC']*1003+['YCB_PUBLIC_REAL']*4200)
    split=np.array(['train']*768+['val']*235+['train']*3400+['val']*800)
    y=np.full((n,8),-1,np.int8); y[:,1]=0; y[:a,0]=1; y[:a,2]=1
    original=dict(id=np.array([f'artificial{i}' for i in range(n)]),source=source,split=split,
        label8=y,native_GTref8=np.zeros((n,8,2)),
        convex_self_proxy8_separate=np.ones((n,8),np.int8),
        near_boundary8=np.zeros((n,8),bool),rule8=np.full((n,8),'GENERATED'))
    orig=np.arange(a)
    after={k:v[orig].copy() for k,v in original.items()}
    after['id']=np.array([x+'::V77_OPAQUE_AFTER' for x in after['id']])
    after['label8'][:,0]=0
    before_primary=np.isin(y,[0,1])
    after.update(original_row_index=orig,after_row_index=np.arange(a),anchor_k=np.zeros(a,np.int64),
        pair_anchor_success=np.ones(a,bool),anchor_training_target=np.zeros(a,np.int8),
        original_primary8=before_primary[orig],query_valid8=np.ones((a,8),bool),
        reference_in_local_square8=np.ones((a,8),bool))
    count=n+a
    d=dict(frame_index=np.arange(count),id=np.r_[original['id'],after['id']],
        source=np.r_[source,after['source']],split=np.r_[split,after['split']],
        domain=np.r_[np.zeros(n,np.int8),np.ones(a,np.int8)],
        original_row_index=np.r_[np.arange(n),orig],query8_xy=np.zeros((count,8,2)),
        L=np.full(count,100.),context42=np.zeros((count,8,42),np.float32),
        query_center_valid=np.ones((count,8),bool),ROI_anyvalid=np.ones((count,8),bool),
        localany=np.ones((count,8),bool),targetany=np.ones(count,bool))
    m=T.metadata_from_compact(d,original,after,paired,base)
    assert m['N']==80016 and len(m['all_frame_id'])==10002
    assert np.array_equal(m['primary'].reshape(count,8)[:n],before_primary)
    assert m['pair_good'].all() and len(m['pair_good'])==a
    assert not m['primary'].reshape(count,8)[:,3:].any()
    rng=np.random.default_rng(20261023)
    for _ in range(64):
        ids=paired.sample(m,rng)
        assert m['primary'][ids].all() and m['train'][ids].all()
        assert all(np.sum((m['source'][ids]==s)&(m['y'][ids]==c))==24 for s in range(2) for c in range(2))
    rejected=[]
    for key in ('query8_xy','L','context42','query_center_valid','ROI_anyvalid','localany','targetany','id','split','domain','original_row_index'):
        changed={k:v.copy() for k,v in d.items()}
        arr=changed[key]
        if arr.dtype.kind in 'US': arr[n]='wrong_identity'
        elif arr.dtype.kind=='b': arr[n]=~arr[n]
        else: arr[n]+=1
        try:
            T.metadata_from_compact(changed,original,after,paired,base)
        except AssertionError:
            rejected.append(key)
        else:
            raise AssertionError(('changed_pair_fact_not_rejected',key))
    report=dict(PASS=True,test=b(Path(__file__).resolve()),trainer=b(R/'train_source_once.py'),
        artificial_frames=count,artificial_slots=count*8,identity_and_native_order_preserved=True,
        unknown_zero_admission=True,all_successful_pairs_C=True,source_class24_batches=64,
        rejected_changed_pair_fields=rejected,all_source_payload_feature_GT_weights_RGB_reads=0,
        generated_head_or_backbone_forward=0,optimizer_updates=0,current_reads=0,PnP=0)
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(PASS=True,proof=b(output))))

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--output',type=Path,default=R/'GENERATED_METADATA_JOIN_TESTS.json')
    main(ap.parse_args().output)
