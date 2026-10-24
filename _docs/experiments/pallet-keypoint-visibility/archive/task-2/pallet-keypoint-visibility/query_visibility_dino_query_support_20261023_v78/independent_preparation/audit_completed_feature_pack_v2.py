"""Independent V78 saved feature QA; no RGB/P3/model forward or GT read.

Explicit run only after feature PACK_SEAL. Input masks here are factual
canvas-validity arrays, not instance ownership/semantic annotation masks.
"""
from pathlib import Path
from collections import Counter
from functools import lru_cache
import argparse,hashlib,json,time
import numpy as np
from dino_pool_audit_math import expected_valid_fractions,patch_weights,assert_fp16_storage,VALID_FRACTION_ABS_BOUND
R=Path(__file__).resolve().parent;V=R.parent;OUT=V/'feature_pack'
COPY=['context42','query8_xy','L','query_center_valid','ROI_anyvalid','ROI_center_valid','query_box_xyxy','target_crop_center_xy','target_crop_side','target_context_valid','native_index','initial8_norm','initial8_norm_valid','query_valid_cached','pure_continuous_query_valid','nearest_pixel_valid','query_missing']
EXTRA=['local_features','target_features','validity_fractions','frame_index','original_row_index','domain','id','source','split','localany','targetany']
EXPECTED_DECISION_SHA='a008fc7d35e9aa1d0d4b7d631a796f238aaba7f0aa052fe7da087080b70115e9'
EXPECTED_PROTOCOL_SHA='b33a87e92070d7283abd45bf196fc7cd25a3439c987758bf448d2451c5411396'

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for a in iter(lambda:f.read(1<<20),b''):h.update(a)
 return h.hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def load(p):return json.loads(Path(p).read_text())
def check(b):assert sha(b['path'])==b['sha256'],b['path']
def same(a,b):
 a=np.asarray(a);b=np.asarray(b);return np.array_equal(a,b,equal_nan=True)if a.dtype.kind in 'fc'and b.dtype.kind in 'fc'else np.array_equal(a,b)
def save(p,x):
 with p.open('x')as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')

def actual(output):
 start=time.monotonic();output.mkdir(parents=True,exist_ok=False)
 derivation=load(R/'VALIDITY_NUMERIC_BOUND_DERIVATION.json');assert derivation['observed_max_used_to_choose_bound']is False
 weight_bounds={x['native_size']:x['avgpool196_absolute_bound']for x in derivation['bounds_by_native_size']}
 prot=load(V/'feature_preparation/FEATURE_PROTOCOL.json');assert sha(V/'feature_preparation/FEATURE_PROTOCOL.json')==EXPECTED_PROTOCOL_SHA
 assert prot['decisions']['sha256']==EXPECTED_DECISION_SHA and prot['frames']==10002 and prot['query_slots']==80016
 assert prot['forward_fields']==['rgb_uint8','canvas_valid_bool']and prot['new_training_current_GT_PnP']==0
 for b in prot['bindings']+prot['official_code']+[prot['weight']]:check(b)
 seal=load(OUT/'PACK_SEAL.json');assert seal['completed']and seal['PASS']and seal['GT_current_fit_PnP']==0
 for k in ['manifest','lineage','inference_receipt','code','pooling','protocol']:check(seal[k])
 assert seal['protocol']==bind(V/'feature_preparation/FEATURE_PROTOCOL.json')
 man=load(OUT/'PACK_MANIFEST.json');receipt=load(OUT/'FEATURE_INFERENCE_RECEIPT.json')
 assert man['completed']and man['frame_count']==10002 and man['query_count']==80016
 assert receipt['completed']and receipt['PASS']and receipt['weights_unchanged']and receipt['state_digest_before']==receipt['state_digest_after']
 assert receipt['unique_source_frames']==10002 and receipt['unique_views_encoded']==90018
 assert receipt['no_patch_tokens_stored']and receipt['all_model_parameters_float32']and receipt['no_autocast']and receipt['TF32']is False and receipt['GT_reference_mask_label_current_reads']==receipt['new_training_PnP']==0
 assert receipt['weights']==prot['weight']
 proofs=receipt['fixed_batch_repeat_and_pool_proofs'];needed=set();regimes=set()
 for p in proofs:
  check(p['file']);assert p['repeat_token_bit_differences']==0 and p['independent_float64_pool_max_abs_diff']<=2e-5
  regimes.update((x,p['kind'])for x in p['regimes'])
  with np.load(p['file']['path'],allow_pickle=False)as z:needed.update(int(x)for x in z['frame_index'])
 assert regimes=={(r,k)for r in prot['actual_fixed_repeat_regimes']for k in ['local','target']}
 metadata={b['name']:[json.loads(x)for x in Path(b['lineage']['path']).read_text().splitlines()]for b in prot['banks']}
 banks={b['name']:b for b in prot['banks']}
 input_chunks={b['name']:{c['start']:c for c in load(b['manifest']['path'])['chunks']}for b in prot['banks']}
 preserved={};feature_proof={};counts=Counter();allids=[];allrows=[];pos=0;ordinary_calls=0;fraction_mismatch=0;fraction_max=0.;fraction_bound_violations=0;zero_support_flips=0;zero_feature_violations=0;finite_elements=0
 for chunk in man['chunks']:
  check(chunk['file']);assert chunk['start']==pos and chunk['end']>pos
  bank=banks[chunk['input_bank']];assert chunk['input_bank_start']==bank['start'];inp=input_chunks[bank['name']][chunk['input_start']]
  assert chunk['input_chunk']==inp['file']and chunk['input_start']==inp['start']and chunk['input_end']==inp['end']
  check(inp['file']);F=chunk['end']-chunk['start'];assert chunk['start']==bank['start']+inp['start']and F==inp['end']-inp['start']
  meta=metadata[bank['name']][inp['start']:inp['end']];assert len(meta)==F
  with np.load(inp['file']['path'],allow_pickle=False)as iz, np.load(chunk['file']['path'],allow_pickle=False)as z:
   assert set(z.files)==set(COPY+EXTRA)
   for k in COPY:assert z[k].dtype==iz[k].dtype and same(z[k],iz[k]),('Changed copied input',k,pos)
   local=z['local_features'];target=z['target_features'];frac=z['validity_fractions']
   assert local.shape==(F,8,2,384)and target.shape==(F,384)and frac.shape==(F,8,3)
   assert local.dtype==target.dtype==frac.dtype==np.float16
   assert np.isfinite(local).all()and np.isfinite(target).all()and np.isfinite(frac).all()and(frac>=0).all()and(frac<=1).all();finite_elements+=local.size+target.size+frac.size
   assert np.array_equal(z['frame_index'],np.arange(pos,chunk['end']))and(z['domain']==bank['domain']).all()
   source=np.array([r.get('source','SYNTHETIC')for r in meta]);split=np.array(['val'if r['split']in('val','validation','VALIDATION')else'train'for r in meta]);ids=np.array([r['id']for r in meta])
   original=np.array([pos+j if bank['domain']==0 else r['original_row_index']for j,r in enumerate(meta)])
   for k,a in [('id',ids),('source',source),('split',split),('original_row_index',original)]:assert np.array_equal(z[k],a),(k,pos)
   lv=iz['crop_valid96'];tv=iz['target_valid128'];assert lv.dtype==tv.dtype==np.bool_
   assert np.array_equal(z['localany'],lv.reshape(F,8,-1).any(2))and np.array_equal(z['targetany'],tv.reshape(F,-1).any(1))
   ref=expected_valid_fractions(lv,tv);err=abs(ref-frac.astype(np.float64));fraction_max=max(fraction_max,float(err.max(initial=0)));fraction_mismatch+=int((ref.astype(np.float16)!=frac).sum());fraction_bound_violations+=int((err>VALID_FRACTION_ABS_BOUND).sum());zero_support_flips+=int(((ref==0)!=(frac==0)).sum())
   zero_feature_violations+=int(np.count_nonzero(local[:,:,0][frac[:,:,0]==0]))+int(np.count_nonzero(local[:,:,1][frac[:,:,1]==0]))+int(np.count_nonzero(target[frac[:,0,2]==0]))
   for j in range(F):
    gi=pos+j;allids.append(str(ids[j]));row=dict(frame_index=gi,id=str(ids[j]),source=str(source[j]),split=str(split[j]),domain=bank['domain'],original_row_index=int(original[j]),input_bank=bank['name'],input_row_index=inp['start']+j);allrows.append(row);counts[(str(source[j]),str(split[j]),bank['domain'])]+=1
    if gi in needed:
     preserved[gi]=(lv[j].copy(),tv[j].copy());feature_proof[gi]=(local[j].copy(),target[j].copy(),frac[j].copy())
  ordinary_calls+=(F*8+31)//32+(F+31)//32;pos=chunk['end']
  if len(allrows)%1000<128:print(json.dumps(dict(frames=pos,total=10002,seconds=round(time.monotonic()-start,2))),flush=True)
 assert pos==10002 and len(set(allids))==10002 and fraction_bound_violations==zero_support_flips==zero_feature_violations==0
 assert receipt['backbone_forward_calls']==ordinary_calls+receipt['repeat_forward_calls']and receipt['repeat_forward_calls']==len(proofs)
 published_counts=[dict(source=s,split=p,domain=d,frames=n)for(s,p,d),n in sorted(counts.items())];assert man['source_split_domain_counts']==published_counts
 lineage=[json.loads(x)for x in Path(man['lineage']['path']).read_text().splitlines()];assert lineage==allrows
 proof_results=[]
 for proof in proofs:
  with np.load(proof['file']['path'],allow_pickle=False)as z:pv={k:z[k]for k in z.files}
  frame=pv['frame_index'];query=pv['query_index'];kind=proof['kind']
  if kind=='local':
   stored=np.stack([feature_proof[int(i)][0][int(k)]for i,k in zip(frame,query)]);sf=np.stack([feature_proof[int(i)][2][int(k),:2]for i,k in zip(frame,query)]);masks=np.stack([preserved[int(i)][0][int(k),0]for i,k in zip(frame,query)])
  else:
   assert (query==-1).all();stored=np.stack([feature_proof[int(i)][1]for i in frame]);sf=np.array([feature_proof[int(i)][2][0,2]for i in frame]);masks=np.stack([preserved[int(i)][1][0]for i in frame])
  quant=assert_fp16_storage(pv['pooled_fp32'],stored);fq=assert_fp16_storage(pv['fractions_fp32'],sf)
  independent_w=patch_weights(masks);w=pv['token_validity_weights'];assert w.dtype==np.float32 and w.shape==independent_w.shape
  wd=float(np.max(abs(w.astype(float)-independent_w)));assert wd<=weight_bounds[masks.shape[-1]]
  if kind=='local':fw=np.stack([w[:,7:9,7:9].astype(float).mean((1,2)),w.astype(float).mean((1,2))],1)
  else:fw=w.astype(float).mean((1,2))
  fd=float(np.max(abs(fw-pv['fractions_fp32'])));assert fd<=1e-6
  proof_results.append(dict(kind=kind,regimes=proof['regimes'],file=proof['file'],views=len(frame),pooled_quantization=quant,fraction_quantization=fq,independent_valid_weights_max_abs_diff=wd,analytic_weight_bound=weight_bounds[masks.shape[-1]],within_original_1e_6_engineering_assertion=wd<=1e-6,independent_weight_mean_vs_GPU_fraction_max_abs_diff=fd,repeat_tokens_bit_differences=0,producer_transient_NumPy_pool_max_abs_diff=proof['independent_float64_pool_max_abs_diff']))
 report=dict(PASS=True,scope='SOURCE saved features and input-only coordinate/context/canvas-validity arrays; no semantic masks/GT/RGB/P3/model forward decoded.',code=bind(__file__),independent_math=bind(R/'dino_pool_audit_math.py'),protocol=bind(V/'feature_preparation/FEATURE_PROTOCOL.json'),feature_seal=bind(OUT/'PACK_SEAL.json'),frames=10002,query_slots=80016,unique_backbone_views=90018,all_forward_metadata_joins_exact=True,all_copied_fields_dtype_and_values_bit_exact=COPY,source_split_domain_counts=published_counts,all_feature_elements_finite=finite_elements,independent_fraction_FP64_vs_stored_FP16_max_abs=fraction_max,independent_fraction_cast_bit_mismatch_elements=fraction_mismatch,independent_fraction_engineering_bound=VALID_FRACTION_ABS_BOUND,independent_fraction_bound_violations=fraction_bound_violations,zero_support_status_flips=zero_support_flips,nonzero_zero_support_pooled_elements=zero_feature_violations,fixed_FP32_to_FP16_proofs=proof_results,original_1e_6_weight_engineering_assertion_status='FAILED_AND_PRESERVED; not retrospectively PASS',numeric_bound_derivation=bind(R/'VALIDITY_NUMERIC_BOUND_DERIVATION.json'),original_failed_audit_code=bind(R/'provenance/FAILED_FEATURE_QA_WEIGHT_FP64_TOLERANCE_V1/audit_completed_feature_pack.py'),floating_difference_interpretation='FP32 interpolation/196avgpool arithmetic differs slightly from independent FP64; fixed CPU32/GPU32 diagnostic confirms same recipe. No source/model/threshold changed.',backbone_state_before_after_equal=True,checkpoint_file_unchanged=True,patch_token_vectors_archived=False,independent_pooling_evidence='Producer freshly repeated actual batches and compared transient actual tokens+w with NumPy64 sum; this auditor independently reconstructs resize/validity weights and casts from saved FP32 pooled vectors. Full actual patchtokens are deliberately not archived, and no new forward was run.',limitations=['Fractional canvas validity is an input sampling mask, not target ownership or semantic visibility.','Pooling weights do not prevent invalid black padding from affecting other ViT tokens through global attention.','Generic backbone/pooling/head/initialization change together; this is a new candidate rather than a DINO-only causal ablation.','Source labels/public q emulation remain unchanged conditional proxies; successful feature QA is not task success.'],GT_model_RGB_P3_current_forward_fit_PnP=0,seconds=time.monotonic()-start)
 save(output/'INDEPENDENT_FEATURE_QA.json',report);save(output/'AUDIT_SEAL.json',dict(PASS=True,bindings=[bind(p)for p in sorted(output.iterdir())if p.is_file()],prior_artifacts_modified=False));print(json.dumps(dict(PASS=True,report=bind(output/'INDEPENDENT_FEATURE_QA.json'))))

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args();actual(a.output)
if __name__=='__main__':main()
