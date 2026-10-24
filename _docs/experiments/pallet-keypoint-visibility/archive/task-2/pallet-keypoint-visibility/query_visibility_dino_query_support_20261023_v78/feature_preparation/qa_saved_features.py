"""Saved-feature QA only. No DINO/GT/model/weights/RGB/body-feature reads."""
from pathlib import Path
import hashlib,json
import numpy as np

HERE=Path(__file__).resolve().parent;OUT=HERE.parent/'feature_pack'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb')as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def bind(p):return dict(path=str(p),sha256=sha(p))
def check(b):assert sha(b['path'])==b['sha256'],b['path']
def load(p):return json.loads(Path(p).read_text())
def rows(p):return[json.loads(x)for x in Path(p).read_text().splitlines()]
def write(p,d):
 with open(p,'x')as f:f.write(json.dumps(d,indent=2)+'\n')

seal=load(OUT/'PACK_SEAL.json');assert seal['PASS']and seal['completed']
for k in ('manifest','lineage','inference_receipt','code','pooling','protocol'):check(seal[k])
man=load(seal['manifest']['path']);prot=load(seal['protocol']['path']);receipt=load(seal['inference_receipt']['path'])
assert man['completed']and man['frame_count']==10002 and man['query_count']==80016
assert receipt['PASS']and receipt['weights_unchanged']and receipt['state_digest_before']==receipt['state_digest_after']and receipt['unique_views_encoded']==90018
keys=[k for k in prot['input_NPZ_fields_whitelist']if k not in ('rgb96','crop_valid96','target_rgb128','target_valid128')]
line=rows(man['lineage']['path']);assert len(line)==10002
source={b['name']:rows(b['lineage']['path'])for b in prot['banks']}
proof_entries=receipt['fixed_batch_repeat_and_pool_proofs'];proofs=[]
for p in proof_entries:
 check(p['file']);assert p['repeat_token_bit_differences']==0 and p['independent_float64_pool_max_abs_diff']<=prot['independent_pool_atol']
 with np.load(p['file']['path'],allow_pickle=False)as z:proofs.append((p,{k:z[k]for k in z.files}))
seen=set();checked=0;proof_rows=0;rounding_max=0.;compressed=0
for c in man['chunks']:
 assert c['start']==checked;check(c['file']);check(c['input_chunk']);compressed+=Path(c['file']['path']).stat().st_size
 with np.load(c['file']['path'],allow_pickle=False)as z:a={k:z[k]for k in z.files}
 with np.load(c['input_chunk']['path'],allow_pickle=False)as z:
  original={k:z[k]for k in keys}
  localany=z['crop_valid96'].reshape(len(a['frame_index']),8,-1).any(2)
  targetany=z['target_valid128'].reshape(len(a['frame_index']),-1).any(1)
 for k in keys:assert np.array_equal(a[k],original[k],equal_nan=True),k
 assert np.array_equal(localany,a['localany'])and np.array_equal(targetany,a['targetany'])
 f=c['end']-c['start'];assert a['local_features'].shape==(f,8,2,384)and a['target_features'].shape==(f,384)and a['validity_fractions'].shape==(f,8,3)
 for k in ('local_features','target_features','validity_fractions'):assert a[k].dtype==np.float16 and np.isfinite(a[k]).all()
 assert a['context42'].dtype==np.float32 and np.isfinite(a['context42']).all()
 assert a['validity_fractions'].min()>=0 and a['validity_fractions'].max()<=1
 assert np.array_equal(a['validity_fractions'][:,:,2],np.broadcast_to(a['validity_fractions'][:,:1,2],(f,8)))
 assert not a['local_features'][~a['localany']].any()and not a['target_features'][~a['targetany']].any()
 for j in range(f):
  idx=c['start']+j;r=line[idx];src=source[c['input_bank']][c['input_start']+j]
  assert int(a['frame_index'][j])==idx and r['frame_index']==idx and r['id']==str(a['id'][j])==src['id']
  assert str(a['source'][j])==r['source']==src.get('source','SYNTHETIC')
  assert str(a['split'][j])==r['split']==('val'if src['split']in('val','validation','VALIDATION')else'train')
  assert int(a['original_row_index'][j])==r['original_row_index']==(idx if r['domain']==0 else src['original_row_index'])
  assert int(a['domain'][j])==r['domain'];assert r['id']not in seen;seen.add(r['id'])
 for info,p in proofs:
  for j,idx in enumerate(p['frame_index']):
   if not c['start']<=idx<c['end']:continue
   ii=int(idx-c['start']);qq=int(p['query_index'][j]);ref=p['pooled_fp32'][j]
   stored=a['local_features'][ii,qq]if info['kind']=='local'else a['target_features'][ii]
   assert np.array_equal(stored,ref.astype(np.float16))
   rr=float(np.max(np.abs(stored.astype(np.float32)-ref)));rounding_max=max(rounding_max,rr)
   assert np.all(np.abs(stored.astype(np.float32)-ref)<=np.abs(ref)*2**-11+2**-25)
   fr=a['validity_fractions'][ii,qq,:2]if info['kind']=='local'else a['validity_fractions'][ii,0,2]
   assert np.array_equal(fr,p['fractions_fp32'][j].astype(np.float16));proof_rows+=1
 checked=c['end']
assert checked==10002 and proof_rows==sum(len(p['frame_index'])for _,p in proofs)
qa=dict(completed=True,PASS=True,manifest=bind(OUT/'PACK_MANIFEST.json'),pack_seal=bind(OUT/'PACK_SEAL.json'),protocol=bind(HERE/'FEATURE_PROTOCOL.json'),code=bind(Path(__file__)),
 all_frames=checked,all_queries=checked*8,all_geometry_context_flags_exact=True,all_ID_order_domain_original_lineage_exact=True,all_features_finite_FP16=True,
 all_local_target_validity_flags_exact=True,all_empty_canvas_features_zero=True,all_target_fractions_shared_perframe=True,
 actual_guard_proof_batches=len(proofs),actual_guard_FP32_to_FP16_rows_checked=proof_rows,FP16_rounding_max_abs_error=rounding_max,
 first_fixed_batches_repeat_token_differences=0,actual_independent_pool_max_abs_difference=max(p['independent_float64_pool_max_abs_diff']for p,_ in proofs),
 NPZ_bytes=compressed,source_GT_RGB_P3_weights_current_reads=0,new_model_forward_fit_PnP=0,
 scope='Exact source metadata/context/flags and saved dtype/rounding QA. Independent all-mask fraction audit is separate; no new feature inference.')
write(OUT/'INPUT_FEATURE_QA.json',qa)
write(OUT/'FINAL_FEATURE_SEAL.json',dict(completed=True,PASS=True,manifest=bind(OUT/'PACK_MANIFEST.json'),QA=bind(OUT/'INPUT_FEATURE_QA.json'),pack_seal=bind(OUT/'PACK_SEAL.json'),protocol=bind(HERE/'FEATURE_PROTOCOL.json'),inference_receipt=bind(OUT/'FEATURE_INFERENCE_RECEIPT.json'),QA_code=bind(Path(__file__)),GT_current_fit_PnP=0))
print(json.dumps(bind(OUT/'FINAL_FEATURE_SEAL.json')))
