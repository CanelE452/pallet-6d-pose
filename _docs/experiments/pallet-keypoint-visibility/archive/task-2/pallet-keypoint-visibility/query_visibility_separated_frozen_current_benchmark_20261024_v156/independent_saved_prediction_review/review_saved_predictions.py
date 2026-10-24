"""Authorized saved prediction arithmetic/metadata audit; no labels or models."""
from pathlib import Path
import argparse,csv,hashlib,json
import numpy as np
import torch  # CPU sigmoid arithmetic only; no model class/weight/forward.
H=Path(__file__).resolve().parent;V=H.parent
OUT=V/'frozen_four_head_current_benchmark_v1'
MODELS=('C35','C36','C37','C38');N=1784
FIELDS=('id','native_corner','query_x','query_y','roi_side','valid','raw_query_aux_hidden_logit','raw_p_visible_FP32','authoritative_decision')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def read(p):return json.loads(Path(p).read_text())
def check(b):
    assert '_v112'not in b['path'].lower();assert bind(b['path'])==b;return Path(b['path'])
def exact(a,b):return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
def packet(p):
    with np.load(p,allow_pickle=False)as f:
        assert 'features1928'not in f.files and 'features'not in f.files
        return {k:f[k].copy()for k in f.files}
def write(p,r):
    with Path(p).open('x')as f:json.dump(r,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def as_scalar_bytes(value,dtype):return np.asarray(value,dtype=dtype).tobytes()
def main(expected):
    sealpath=OUT/'PREDICTION_SEAL.json';assert sha(sealpath)==expected
    s=read(sealpath);assert s['completed'] and s['PASS'] and tuple(s['models'])==MODELS
    assert (s['frames'],s['slots'],s['valid'],s['U'],s['additional_U'])==(223,N,1739,45,0)
    assert s['source_admission_unchanged'] and s['benchmark_only_not_admitted'] and s['selected_model']is None
    assert s['new_candidate_head_calls']==8 and s['original_C16_head_calls']==2
    assert s['encoder_R0_RGB_mask_GT_optimizer_fit_PnP_q_correction_calls']==0
    p=read(check(s['protocol']));entry=read(check(s['entry']));policy=read(check(s['future_phase_policy']))
    assert p['models_fixed']==policy['models_fixed_before_current_body_reads'] and p['models']==list(MODELS)
    assert entry['approved_future_frozen_benchmark'] and entry['resolved_protocol']==s['protocol']
    assert bind(s['code']['path'])==s['code'] and s['code']['sha256']=='8bc1caff2992f514f84c268c146aeca84b39242e2b657b4923ba516cfe9ad13d'
    pre=read(V/'independent_preexecution_review/INFERENCE_CODE_REVIEW.json');assert pre['PASS'] and pre['inference_code']==s['code']
    o=p['original'];keys=tuple(o['metadata_whitelist']);assert len(keys)==12 and 'features1928'not in keys
    assert o['input_whitelist']==['features1928',*keys]
    # Lazy metadata whitelist only: do not hash/read the feature-value payload.
    assert '_v112'not in o['feature_packet']['path'].lower()
    with np.load(o['feature_packet']['path'],allow_pickle=False)as f:meta={k:f[k].copy()for k in keys}
    reference=packet(check(o['C16_original_raw_packet']))
    for k in keys:assert exact(meta[k],reference[k]),k
    av=meta['valid'];assert av.dtype==bool and av.shape==(N,) and av.sum()==1739
    for k in('query_available','prior_valid','context_available','local_central4_available','context_Q_inside'):assert meta[k].dtype==bool and meta[k].shape==(N,)
    assert exact(av,meta['query_available']) and np.array_equal(av,meta['prior_valid']&meta['context_available']&meta['local_central4_available'])
    assert not(meta['prior_valid']&~av).any() and (~meta['prior_valid']).sum()==45
    assert len(set(meta['id'].tolist()))==223 and len(set(zip(meta['id'].tolist(),meta['native_corner'].tolist())))==N
    assert np.array_equal(meta['query_xy'],meta['original_query_xy'].astype(np.float32),equal_nan=True)
    assert np.array_equal(meta['side_L'],meta['original_side_L'].astype(np.float32),equal_nan=True)
    parity=read(check(s['C16_parity']));assert parity['completed'] and parity['PASS'] and parity['entry']==s['entry']
    assert parity['all1784_single_batch'] and parity['repeat_calls']==2 and parity['candidate_checkpoint_hash_load_or_forward_calls']==0
    assert parity['state_before']==parity['state_after'] and parity['valid']==1739 and parity['U']==45 and parity['additional_U']==0
    assert set(parity['all_raw_safe_probability_metadata_bytes_exact'])==set(keys)|set(o['C16_output_score_whitelist'])
    assert all(parity['all_raw_safe_probability_metadata_bytes_exact'].values())
    source16=read(check(o['original_C16_inference_seal']))
    assert parity['state_before']==source16['head_state_before']==source16['head_state_after']
    scorefields={'raw_head_logit_before_U','query_logit','p_visible_FP32','raw_three_logits','prediction_D0_E1_Uminus1'}
    packs={};facts={};bindings=[]
    for name in MODELS:
        record=s['models'][name];fixed=p['models_fixed'][name]
        assert record['source_freeze']==fixed['source_freeze'] and record['source_checkpoint']==fixed['source_checkpoint']
        # Source checkpoint binding is compared, never hashed or deserialized.
        freeze=read(check(fixed['source_freeze']))
        assert freeze['checkpoint']==fixed['source_checkpoint'] and not freeze['current_allowed'] and freeze['selected_model']is None
        assert not record['source_checks_all_pass'] and record['source_current_allowed_false_unchanged']
        assert record['state_before']==record['state_after']==fixed['source_state_sha']
        assert record['full1784_repeat2_all3_byteexact'] and record['head_calls']==2 and record['auxiliary_logits_never_primary_gates']
        a=packet(check(record['raw_packet']));assert set(a)==set(keys)|scorefields
        for k in keys:assert exact(a[k],meta[k]),(name,k)
        for k in('raw_head_logit_before_U','query_logit','p_visible_FP32'):assert a[k].shape==(N,) and a[k].dtype==np.float32 and np.isfinite(a[k]).all()
        raw=a['raw_head_logit_before_U'];z=a['query_logit'];pv=a['p_visible_FP32'];three=a['raw_three_logits']
        assert three.shape==(N,3) and three.dtype==np.float32 and np.isfinite(three).all()
        assert exact(raw,three[:,0].copy()) and exact(z,np.where(av,raw,0).astype(np.float32))
        # Same original CPU FP32 scalar-probability operation, without a model.
        expected_pv=torch.sigmoid(-torch.as_tensor(z)).numpy();assert exact(pv,expected_pv)
        pred=np.where(av,(z>=0).astype(np.int8),-1)
        assert exact(a['prediction_D0_E1_Uminus1'],pred) and (pred==-1).sum()==45
        assert not z[~av].any() and (pv[~av]==.5).all()
        with check(record['scores_csv']).open(newline='')as f:
            reader=csv.DictReader(f);assert tuple(reader.fieldnames)==FIELDS;rows=list(reader)
        assert len(rows)==N
        for i,r in enumerate(rows):
            assert r['id']==str(meta['id'][i]) and int(r['native_corner'])==int(meta['native_corner'][i]) and r['valid']==str(bool(av[i]))
            for key,arr in [('query_x',meta['original_query_xy'][:,0]),('query_y',meta['original_query_xy'][:,1]),('roi_side',meta['original_side_L']),('raw_query_aux_hidden_logit',z),('raw_p_visible_FP32',pv)]:
                assert as_scalar_bytes(float(r[key]),arr.dtype)==arr[i].tobytes(),(name,i,key)
            assert r['authoritative_decision']==(('D'if z[i]<0 else'E')if av[i]else'UNKNOWN')
        facts[name]=dict(rows=N,valid=1739,U=45,additional_U=0,all12_original_metadata_arrays_byteexact=True,
            raw3_finite_and_visibility_column_exact=True,safe_U_first_and_probability_FP32_exact=True,CSV_NPZ_all1784_rows_exact=True,
            source_state_before_after_matches_frozen_metadata=True,producer_fullbatch_repeat_claim_retained_not_rerun=True)
        packs[name]=a;bindings.extend([record['raw_packet'],record['scores_csv']])
    assert not packs['C35']['raw_three_logits'][:,1:].any()
    assert exact(packs['C37']['raw_three_logits'][:,1:].copy(),packs['C38']['raw_three_logits'][:,1:].copy())
    # Recheck saved candidate outputs and descriptor metadata; never checkpoints.
    for b in bindings:check(b)
    result=dict(completed=True,PASS=True,code=bind(__file__),prediction_seal=bind(sealpath),protocol=s['protocol'],entry=s['entry'],policy=s['future_phase_policy'],C16_parity=s['C16_parity'],
        candidate_outputs=bindings,models=facts,full1784_original_metadata_CSV_NPZ_and_raw3_checked=True,
        C37_C38_auxiliary_logits_all1784_byteexact=True,C35_neutral_auxiliary_logits_zero=True,
        original_C16_durable_before_candidate_access_supported_by_code_and_producer_receipt=True,
        original_C16_metadata_and_stored_reference_packet_independently_joined=True,
        learned_model_rerun_or_independent_repeat_claim=False,
        source_qualification_failed_and_not_admitted_preserved=True,selected_model=None,
        source_checkpoint_body_or_hash_reads=0,feature_values_or_full_feature_file_hash_reads=0,
        labels_RGB_V112_model_weight_optimizer_or_new_forward_reads_or_calls=0,
        CPU_FP32_sigmoid_is_scalar_arithmetic_only=True,
        limits=['No independent head reexecution or weight/state deserialization. Repeat/state claims are producer assertions checked against sealed source metadata.',
                'No groundtruth or performance evaluation in this audit; those require the separately authorized saved evaluator.',
                'Only the original feature-packet12metadata arrays were read lazily; no feature values or full feature-file checksum were read.'])
    write(H/'REVIEW.json',result);print(json.dumps(dict(review=bind(H/'REVIEW.json'),all4_PASS=True,auxiliary_crossarm_exact=True)))
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--prediction-seal-sha',required=True);args=a.parse_args();main(args.prediction_seal_sha)
