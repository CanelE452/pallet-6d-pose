"""Independent saved public scalar arithmetic, without models or feature reads."""
from pathlib import Path
from collections import Counter
import hashlib,json
import numpy as np

V=Path(__file__).resolve().parent;P=V.parent
S=P/'query_visibility_public_native_supervision_control_20261024_v148'
RUN=S/'runs/C33_C34_PUBLIC_NATIVE_FIXED2000'
def binding(p):return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def read(p):return json.loads(p.read_text())
def write(p,x):
    with p.open('x')as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def sigmoid(z):return np.exp(-np.logaddexp(0.,-z))
def describe(x):
    return dict(N=int(len(x)),mean=float(x.mean()),min=float(x.min()),max=float(x.max()),median=float(np.median(x)))

if __name__=='__main__':
    policy=V/'ROOT_SOURCE_MECHANISM_POLICY_BEFORE_SAVED_ANALYSIS.json'
    assert binding(policy)['sha256']=='777c44db64d04ee439f2fc210898ccbca18deb10dee60df0a051493c4dc460ce'
    protocol=read(S/'source_fit_preparation/PROTOCOL_BEFORE_SOURCE_FIT.json')
    label=Path(protocol['public_labels']['path']);assert binding(label)==protocol['public_labels']
    rows=[json.loads(s)for s in label.read_text().splitlines()if s]
    assert len(rows)==512 and [r['global_index']for r in rows]==list(range(512))
    paths={'C32_INITIAL':RUN/'C33_PUBLIC_NATIVE_WEIGHT0_FIXED2000/INITIAL_PUBLIC512_SCORES.npz',
           'C33':RUN/'C33_PUBLIC_NATIVE_WEIGHT0_FIXED2000/FINAL_PUBLIC512_SCORES.npz',
           'C34':RUN/'C34_PUBLIC_NATIVE_WEIGHT1_FIXED2000/FINAL_PUBLIC512_SCORES.npz'}
    before={k:binding(p)for k,p in paths.items()};pack={}
    for k,p in paths.items():
        with np.load(p,allow_pickle=False)as f:pack[k]={n:f[n].copy()for n in ['raw_head_logit_before_U','global_index','native_slot','query_xy','L','split','head_forwarded','nearest_valid','prediction']}
        a=pack[k];assert np.array_equal(a['global_index'],np.arange(512))
        assert np.isfinite(a['raw_head_logit_before_U']).all()
        for i,r in enumerate(rows):
            assert int(a['native_slot'][i])==r['native_slot'] and str(a['split'][i])==r['split']
            assert np.array_equal(a['query_xy'][i],np.array(r['query_xy'])) and a['L'][i]==r['L']
            assert bool(a['nearest_valid'][i])==r['original_input_valid']
            assert bool(a['prediction'][i]<0)==(not r['original_input_valid'])
        assert a['head_forwarded'].all() and int(a['nearest_valid'].sum())==463
        if k!='C32_INITIAL':
            for n in ['native_slot','query_xy','L','split','head_forwarded','nearest_valid']:
                ref=pack['C32_INITIAL'][n];assert a[n].dtype==ref.dtype and a[n].shape==ref.shape and a[n].tobytes()==ref.tobytes()
    train=np.array([r['split']=='TRAIN_LAB' and r['semantic_state']in ['D','E'] and r['original_input_valid']for r in rows])
    masks={s:train & np.array([r['semantic_state']==s for r in rows])for s in ['D','E']}
    assert masks['D'].sum()==35 and masks['E'].sum()==8
    z=pack['C32_INITIAL']['raw_head_logit_before_U'].astype(np.float64)
    d,e=z[masks['D']],z[masks['E']]
    lossD=.5*np.logaddexp(0.,d).mean();lossE=.5*np.logaddexp(0.,-e).mean()
    gD=.5*sigmoid(d).mean();gE=.5*(sigmoid(e)-1.).mean()
    perrow=[]
    for i in np.flatnonzero(train):
        isD=rows[i]['semantic_state']=='D';n=35 if isD else 8
        loss=float(.5/n*np.logaddexp(0.,z[i] if isD else -z[i]))
        grad=float(.5/n*(sigmoid(z[i])-(0 if isD else 1)))
        perrow.append(dict(global_index=int(i),id=rows[i]['id'],native_slot=rows[i]['native_slot'],state=rows[i]['semantic_state'],initial_z=float(z[i]),weighted_BCE=loss,weighted_dL_dz=grad))
    pair=[]
    for role in ['ALL',2,3]:
        dm=masks['D'].copy();em=masks['E'].copy()
        if role!='ALL':
            nm=np.array([r['native_slot']==role for r in rows]);dm&=nm;em&=nm
        if not dm.any() or not em.any():
            pair.append(dict(role=role,D=int(dm.sum()),E=int(em.sum()),pairs=0));continue
        models={}
        for name,a in pack.items():
            zz=a['raw_head_logit_before_U'].astype(np.float64);gap=zz[em,None]-zz[dm][None,:]
            models[name]=dict(pair_gap=describe(gap.ravel()),rank_auc=float(((gap>0)+.5*(gap==0)).mean()),wrong_or_tied_order_pairs=int((gap<=0).sum()),margin1_violated_pairs=int((gap<1).sum()),softplus_margin1_mean=float(np.logaddexp(0.,1-gap).mean()))
        pair.append(dict(role=role,D=int(dm.sum()),E=int(em.sum()),pairs=int(dm.sum()*em.sum()),models=models))
    delta=pack['C34']['raw_head_logit_before_U'].astype(np.float64)-pack['C33']['raw_head_logit_before_U'].astype(np.float64)
    shifts={}
    for fold in ['TRAIN_LAB','VAL_HALL']:
        for state in ['D','E']:
            m=np.array([r['split']==fold and r['semantic_state']==state and r['original_input_valid']for r in rows]);shifts[fold+'::'+state]=describe(delta[m])
    result=dict(completed=True,PASS=True,policy=binding(policy),code=binding(Path(__file__)),source_label=binding(label),inputs=before,
        TRAIN43_native_state_counts=dict(Counter(str(r['native_slot'])+'::'+r['semantic_state']for r in rows if r['split']=='TRAIN_LAB'and r['semantic_state']in ['D','E'])),
        initial_public_TRAIN43_BCE=dict(weighted_D=float(lossD),weighted_E=float(lossE),total=float(lossD+lossE),D_fraction=float(lossD/(lossD+lossE))),
        initial_public_TRAIN43_scalar_bias_derivative=dict(D=float(gD),E=float(gE),total=float(gD+gE),D_push_magnitude_over_E=float(abs(gD/gE)),interpretation='Positive dL/db means negative-bias update under plain SGD; this scalar clue does not identify the actual full AdamW/hidden-parameter cause.'),
        initial_TRAIN43_all_rows=perrow,TRAIN43_crossclass_pairs=pair,C34_minus_C33_public_valid_shifts=shifts,
        no_model_feature_weight_RGB_current223_reads_or_fit=True,original_U49_preserved=True,selected_model=None,
        limits=['Saved F32 logits evaluated in float64 arithmetic; no actual model gradient/backward was called.','Crossclass pairs are not fixed-q counterfactual observations.','Scalar bias derivative is not full parameter-level causal attribution.','No labels changed or examples removed by query-reference error.'])
    assert before=={k:binding(p)for k,p in paths.items()}
    write(V/'ROOT_PUBLIC_SCALAR_DIAGNOSIS.json',result)
    print(json.dumps({k:result[k]for k in ['initial_public_TRAIN43_BCE','initial_public_TRAIN43_scalar_bias_derivative','TRAIN43_native_state_counts','TRAIN43_crossclass_pairs','C34_minus_C33_public_valid_shifts']},ensure_ascii=False))
