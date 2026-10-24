"""Saved arrays/arithmetic only. No model, raw RGB decoder or GT packet."""
from pathlib import Path
import json,hashlib
import numpy as np
V=Path(__file__).resolve().parent.parent;O=Path(__file__).resolve().parent

def read(p):return json.loads(Path(p).read_text())
def rows(p):return[json.loads(s)for s in Path(p).read_text().splitlines()if s]
def b(p):return dict(path=str(p),sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest())
def chk(d):assert b(Path(d['path']))['sha256']==d['sha256']
def arrsha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def write(n,x):
 with(O/n).open('x')as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
receipt=V/'material_spatial_features/FEATURE_COMPLETE.json'
assert b(receipt)['sha256']=='f6d61a866a9be8b46b8e60928d221838cf507b82a253b00cdc7864a908502839'
write('SCOPE.json',dict(scope='Saved feature packet/input-only lineage/facts and7already-rendered RGB/valid/context array hashes only. No new encoder, model, raw RGB decode, teacher packet/GT label array, current/V112, fit or PnP. Renderer receipt is read only to consume crop indices and file/hash fields; embedded teacher/evidence fields are not used.',receipt=b(receipt)))
f=read(receipt);assert f['completed']and f['PASS']and f['model_state_unchanged']and f['first_batch_repeat_exact']
for k in('output','manifest','inputs','new_query_lineage','input_RGB_spatial_facts','protocol','code'):chk(f[k])
manifest=read(f['manifest']['path']);proto=read(f['protocol']['path'])
assert manifest['old_features']==f['old_features_reused']==proto['old_output']and manifest['old_reused_by_binding_only']
old=read(manifest['old_receipt']['path']);assert old['output']==f['old_features_reused']and old['features_shape']==[11652,1928]and old['occurrences']==41984
inp=rows(f['inputs']['path']);lin=rows(f['new_query_lineage']['path']);facts=rows(f['input_RGB_spatial_facts']['path'])
with np.load(f['output']['path'],allow_pickle=False)as z:a={k:z[k]for k in z.files}
x=a['features'];fa=a['factual_query_available'];ca=a['context_available'];aa=a['query_available'];qi=a['unit_Q_inside']
assert len(inp)==len(facts)==400 and len(lin)==3200 and x.shape==(3200,1928)and x.dtype==np.float32 and np.isfinite(x).all()
assert all(a[k].dtype==np.int32 for k in('occurrence_unit','global_occurrence_index','global_input_unit_index'))
assert all(a[k].dtype==bool for k in('query_available','factual_query_available','context_available','unit_Q_inside'))
assert np.array_equal(a['occurrence_unit'],np.arange(3200))and np.array_equal(a['global_occurrence_index'],np.arange(41984,45184))and np.array_equal(a['global_input_unit_index'],np.arange(11652,14852))
assert np.array_equal(aa,fa&ca)and np.array_equal(x[:,1927],qi.astype(np.float32))
expected_p=[];inside=[];folds={};invalid_finite_nonzero=0
for i,(r,fc)in enumerate(zip(inp,facts)):
 assert r['instance_row_index']==fc['instance_row_index']==i and r['input_id']==fc['input_id']and r['plan_row_index']==fc['plan_row_index']and r['image']==fc['image']and r['split']==fc['split']and r['original_split']==fc['original_split']
 q=np.asarray(r['initial_points'],np.float64);box=np.asarray(r['query_box_xyxy'],np.float64)
 assert np.array_equal(box,np.r_[q.min(0),q.max(0)])and r['query_box_xyxy']==fc['query_box_xyxy']
 uv=(q-box[:2])/(box[2:]-box[:2]);ins=((uv>=0)&(uv<=1)).all(1)&~(q==[-1,-1]).all(1);inside.extend(ins)
 folds[r['split']]=folds.get(r['split'],0)+1
 assert np.array_equal(x[i*8:i*8+8,1924],np.full(8,np.float32(fc['target_valid_fraction'])))
 for k in range(8):
  u=8*i+k;row=lin[u];fact=fc['queries'][k]
  assert row['local_occurrence_index']==row['local_input_unit_index']==u and row['global_occurrence_index']==41984+u and row['global_input_unit_index']==11652+u
  assert row['instance_row_index']==i and row['logical_aux_slot']==k and row['input_id']==r['input_id']and row['query_id']==r['input_id']+'::q'+str(k)and row['query_xy']==r['initial_points'][k]and row['L']==r['roi_side']and row['image']==r['image']and row['split']==r['split']and row['original_split']==r['original_split']and row['query_box_xyxy']==r['query_box_xyxy']
  assert row['factual_query_available']==bool(fa[u])and row['context_available']==bool(ca[u])and row['query_available']==bool(aa[u])and row['context_Q_inside']==bool(qi[u])
  assert fact['logical_aux_slot']==k and fa[u]==(fact['q_continuous']and fact['q_nearest']and fact['q_center_valid']and fact['RGB_central4_valid'])and ca[u]==(fc['target_valid_fraction']>0)
  expected_p.append(fact['RGB_central4_valid'])
assert folds=={'TRAIN':300,'VALIDATION':100}and np.array_equal(qi,inside)
assert int(aa.sum())==3064 and int((~fa).sum())==136 and ca.all()
blocks=[x[:,i*384:(i+1)*384].astype(np.float64)for i in range(5)];gates=[x[:,1924]>0,qi,x[:,1925]>0,x[:,1926]>0,np.array(expected_p)]
normerrors={}
for n,block,valid in zip('TQFCP',blocks,gates):
 norm=np.linalg.norm(block,axis=1);err=np.max(abs(norm-valid.astype(np.float64)));assert err<5e-7;normerrors[n]=float(err)
coserrors={}
for col,(i,j)in zip(range(1920,1924),[(0,3),(0,2),(1,3),(1,4)]):
 u,v=blocks[i],blocks[j];want=np.sum(u*v,axis=1)/np.maximum(np.linalg.norm(u,axis=1)*np.linalg.norm(v,axis=1),1e-24);err=float(np.max(abs(want-x[:,col])));assert err<5e-7;coserrors[str(col)]=err
assert ((x[:,1924:1927]>=0)&(x[:,1924:1927]<=1)).all()
# Only7 saved already-rendered crop arrays are read, never source images/depth/masks.
render_path=V/'fixed_material_rgb_review/RENDER_RECEIPTS.json';render=read(render_path);assert render['completed']and render['selected_cases']==7
matches=[]
for row in render['rendered']:
 u=int(row['slot_index']);i,k=divmod(u,8);fr=facts[i];fact=fr['queries'][k];binding=next(d for d in row['files']if d['path'].endswith('EXACT_RAW_CROPS_AND_VALID.npz'));chk(binding)
 with np.load(binding['path'],allow_pickle=False)as z:rr=z['rgb96'];vv=z['crop_valid96'];tt=z['context224_fp32'];tv=z['context_valid224']
 assert row['input_id']==inp[i]['input_id']and row['q']==inp[i]['initial_points'][k]and row['L']==inp[i]['roi_side']
 assert arrsha(rr)==fact['RGB96_sha256']==row['exact_raw_crop_values_sha256']and arrsha(vv)==fact['valid96_sha256']and arrsha(tt)==fr['target_RGB224_F32_sha256']
 assert x[u,1924]==np.float32(tv.mean())and x[u,1925]==np.float32(vv.mean())and x[u,1926]==np.float32(vv[:,32:64,32:64].mean())
 assert fa[u]==bool(fact['q_center_valid']and vv[:,47:49,47:49].all())
 matches.append(dict(slot_index=u,input_id=inp[i]['input_id'],saved_arrays=binding,RGB96_sha256=arrsha(rr),valid96_sha256=arrsha(vv),context224_F32_sha256=arrsha(tt),fractions_bitexact=True))
review=dict(completed=True,PASS=True,blockers=[],scope='Independent saved-only input join, feature arithmetic and seven existing crop-array replay; no encoder/head/model/raw RGB/GT label packet/current/V112/fit/PnP.',feature_complete=b(receipt),packet=f['output'],new400frames_3200queryunits_distinct=True,new3200local_map_exact=True,global14852unit_45184occurrence_offsets_exact=True,all_ID_q_L_box_order_original_folds_preserved=True,fold_frames=folds,fold_occurrences=dict(TRAIN=2400,VALIDATION=800),query_available=3064,factual_U=136,context_U=0,extra_context_U=0,Q_inside=3200,Qoutside_not_extra_U=True,old_features_reference_exact=manifest['old_features'],old_feature_tensor_not_deserialized_or_copied=True,feature_norm_max_errors=normerrors,feature_cosine_max_errors=coserrors,support_fraction_bounds_and_flags_exact=True,independent_saved_crop_matches=matches,saved_crop_cases_verified=7,producer_runtime_proof=dict(model_state_unchanged=f['model_state_unchanged'],first_batch_repeat_exact=f['first_batch_repeat_exact'],max_views=f['max_views'],encoder_calls=f['encoder_calls']),limits=['No DINO/token/head recomputation. Saved vector normalization/cosines/support/lineage and selected7 crop bytes are verified; feature-to-image provenance for other3193 queries relies on pinned producer/runtime proof.','No teacher label array, depth or mask is opened or judged. Renderer metadata contains embedded annotation/evidence fields; only selected crop identity/hash fields were consumed.','No source-training admission or natural/generalization proof is created by this engineering audit.'],own_code=b(Path(__file__)),new_model_or_pixel_forward_fit_private_PnP=0)
write('REVIEW.json',review);write('SEAL.json',dict(completed=True,PASS=True,review=b(O/'REVIEW.json'),files=[b(O/'SCOPE.json'),b(Path(__file__))]))
print(json.dumps(dict(review=b(O/'REVIEW.json'),seal=b(O/'SEAL.json'),saved_crop_matches=len(matches),available=int(aa.sum())),indent=2))
