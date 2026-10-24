"""Independent NumPy replay of saved stencil arithmetic; no Torch/model/RGB."""
from pathlib import Path
import hashlib,json
import numpy as np
V=Path(__file__).resolve().parents[1]
def bind(p):return dict(path=str(p),sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest())
def read(p):return json.loads(Path(p).read_text())
def check(b):assert bind(Path(b['path']))=={k:b[k]for k in('path','sha256')};return Path(b['path'])
policy=read(V/'ROOT_C19_SOURCE_CONTROL_POLICY_BEFORE_FEATURES.json')
receipt_path=V/'stencil_features_v2/FEATURE_COMPLETE.json';r=read(receipt_path)
assert r['completed']and r['PASS']and r['base_features']==policy['bindings']['base_features']
pack=check(r['output']);basepath=check(r['base_features']);manifest=check(r['input_occurrences'])
for k in('code','common_code','protocol','entry','policy','precision_addendum'):check(r[k])
protocol=read(r['protocol']['path']);assert protocol['outer_order']==[0,1,2,3,5,6,7,8]
with np.load(pack,allow_pickle=False)as f:d={k:f[k]for k in f.files}
with np.load(basepath,allow_pickle=False)as f:b={k:f[k]for k in f.files}
assert d['raw_nodes'].shape==(4428,9,384)and d['pooled9_repeat8'].shape==d['ordered_outer8'].shape==(4428,3072)
assert all(d[k].dtype==np.float32 and np.isfinite(d[k]).all()for k in('raw_nodes','pooled9_repeat8','ordered_outer8'))
assert b['features'].shape==(4428,1928)and b['features'].dtype==np.float32 and np.isfinite(b['features']).all()
for k in('occurrence_unit','query_available','factual_query_available','context_available','global_occurrence_index','unit_Q_inside'):
    assert np.array_equal(d[k],b[k])and d[k].dtype==b[k].dtype
assert d['occurrence_unit'].shape==(4896,)and np.array_equal(np.unique(d['occurrence_unit']),np.arange(4428))
assert np.array_equal(d['global_occurrence_index'],np.arange(4896))and (~d['query_available']).sum()==51
assert d['query_available'].dtype==bool and np.array_equal(d['query_available'],d['factual_query_available']&d['context_available'])
rep=np.full(4428,-1,np.int32)
for i,u in enumerate(d['occurrence_unit']):
    if rep[u]<0:rep[u]=i
assert np.array_equal(rep,d['representative_occurrence_index'])
rows=[json.loads(s)for s in manifest.read_text().splitlines()if s]
assert len(rows)==4896 and [x['occurrence_index']for x in rows]==list(range(4896))
for k in('query_available','factual_query_available','context_available'):assert np.array_equal(d[k],np.array([x[k]for x in rows]))
# Independent F64 arithmetic, not producer/common helper imports.
raw=d['raw_nodes'].astype(np.float64)
normalize=lambda a:a/np.maximum(np.sqrt(np.sum(a*a,axis=-1,keepdims=True)),1e-12)
pooled=normalize(np.mean(raw,axis=1));ordered=normalize(raw[:,[0,1,2,3,5,6,7,8]])
pp=d['pooled9_repeat8'].reshape(4428,8,384);oo=d['ordered_outer8'].reshape(4428,8,384)
assert all(np.array_equal(pp[:,0],pp[:,j])for j in range(8))
pooldiff=float(np.max(np.abs(pp[:,0]-pooled)));outerdiff=float(np.max(np.abs(oo-ordered)))
assert pooldiff<=1e-6 and outerdiff<=1e-6
mask=d['factual_query_available'][rep] # independently corroborated against literal stored maskedP.
oldP=b['features'][:,1536:1920];assert np.array_equal(mask,np.any(oldP!=0,axis=1))
centre=normalize(raw[:,4]);expected=np.where(mask[:,None],centre,0.)
pdiff=float(np.max(np.abs(oldP-expected)));assert pdiff<=1e-6 and not oldP[~mask].any()
assert np.linalg.norm(raw,axis=2).min()>0
batches=r['parity_batches'];assert len(batches)==554
for j,bb in enumerate(batches):
    assert bb==dict(first_unit=8*j,units=min(8,4428-8*j),views=3*min(8,4428-8*j),F_bitexact=True,P_bitexact=True)
assert r['primary_calls']==554 and r['repeat_calls']==1 and r['encoder_calls']==555
assert r['first_batch_repeat_exact']and r['all_F_P_bitexact']and not r['base1928_rewritten']and r['all_source_mapping_and_availability_exact']
assert r['encoder_state_before']==r['encoder_state_after']and r['model_state_unchanged']
runtime=read(protocol['asset_runtime']['path']);assert r['encoder_state_before']==runtime['state_before_SHA256']
failure_path=V/'stencil_features/FAILED_EXECUTION.json';failure=read(failure_path);correction_path=V/'ROOT_FEATURE_API_CORRECTION_BEFORE_V2.json';correction=read(correction_path)
assert 'TypeError: all()'in failure['traceback']and not failure['automatic_retry']and correction['root_explicit_reexecution_after_fix']and not correction['training_started']and correction['failure']==bind(failure_path)
# Recheck unchanged base-file bytes after every saved-only arithmetic read.
assert bind(basepath)==r['base_features']
out=dict(completed=True,PASS=True,blockers=[],feature_pack=r['output'],feature_receipt=bind(receipt_path),
    scope='All saved public stencil/base feature values plus bound input-only metadata; no RGB, labels, weights, model import/forward, fit, current or V112.',
    dimensions=dict(raw_nodes=[4428,9,384],each_arm=[4428,3072],base=[4428,1928],occurrences=4896,dtype='float32 finite'),
    mapping=dict(original4896_order_exact=True,unit_map_exact=True,all4428_units_used=True,first_occurrence_representatives_exact=True,all_original_availability_flags_exact=True,input_U=51,new_U=0,unit_Q_inside_exact=True,input_manifest_SHA_q_L_folds_preserved=True),
    arithmetic=dict(pooled8_blocks_bitidentical=True,rowmajor_outer_order=[0,1,2,3,5,6,7,8],pool='F64 mean of all9 raw vectors then L2, compared with each saved repeat block',ordered='F64 independent L2 per outer raw vector in fixed row-major order',old_centreP='F64 normalize raw node4 and apply original factual/centre mask; mask equals storedP nonzero rows4377, unavailable51 remain exact zero',max_abs_pool_difference=pooldiff,max_abs_ordered_difference=outerdiff,max_abs_masked_centreP_difference=pdiff,independent_arithmetic_absolute_tolerance=1e-6,tolerance_scope='Only NumPy F64 vs producer GPU FP32 reduction/normalization. No tolerance for hashes, mapping, availability, pooled-block identity, source input parity or producer F/P strict stop.'),
    producer_certificate=dict(encoder_calls=555,primary_batches=554,repeat_calls=1,last_batch_units=4,last_batch_views=12,all4428_F_P_strict_GPU_parity_reported=True,encoder_state_before=r['encoder_state_before'],encoder_state_after=r['encoder_state_after'],state_matches_prior_runtime=True,first_repeat_exact=True,seconds=r['seconds'],limitation='Raw original256patch maps and CLS are not stored here. This audit independently replays normalization/packing and centreP coherence; it does not independently rerun encoder/raw sampling/F parity or rehash model parameters.'),
    base1928=dict(original_file_SHA_unchanged=True,binding=r['base_features'],rewritten=False),
    preserved_V1=dict(failure=bind(failure_path),root_API_correction=bind(correction_path),reason='Torch2.1.1 tuple-dimension Tensor.all API incompatibility; explicit new V2 directory/root entry, original failure retained.'),
    bindings=dict(policy=r['policy'],precision_addendum=r['precision_addendum'],producer=r['code'],common=r['common_code'],protocol=r['protocol'],actual_entry=r['entry'],input_manifest=r['input_occurrences'],audit_code=bind(Path(__file__))),
    actual_reviewer_model_encoder_forward_fit_RGB_weights_current_V112_PnP=0,original_files_changed=False,no_training_or_promotion_decision=True,fully_solved=False)
with(Path(__file__).parent/'REVIEW.json').open('x')as f:json.dump(out,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
print(json.dumps(dict(completed=True,PASS=True,review=bind(Path(__file__).parent/'REVIEW.json'),pooldiff=pooldiff,outerdiff=outerdiff,maskedPdiff=pdiff)))
