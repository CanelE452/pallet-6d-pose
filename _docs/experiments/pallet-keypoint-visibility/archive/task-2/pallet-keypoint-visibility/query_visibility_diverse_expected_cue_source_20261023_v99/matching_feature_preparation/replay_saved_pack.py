from pathlib import Path
import json,hashlib,numpy as np
V=Path(__file__).resolve().parent.parent;H=V/'matching_feature_preparation';O=V/'matching_feature_pack'
def read(p):return json.loads(Path(p).read_text())
def rows(p):return [json.loads(x)for x in Path(p).read_text().splitlines()if x]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return {'path':str(p.resolve()),'sha256':sha(p)}
def check(b):assert sha(b['path'])==b['sha256'];return Path(b['path'])
def write(p,x):
 with p.open('x')as f:json.dump(x,f,indent=2);f.write('\n')
s=read(O/'INPUT_PACK_SEAL.json');m=read(check(s['manifest']));qa=read(check(s['QA']));line=rows(check(s['input_lineage']));facts=rows(check(s['frame_facts']));p=read(check(s['protocol']));render=read(check(p['source_manifest']));qs=rows(check(p['queries']));orig=rows(check(p['occurrences']))
assert s['completed']and s['PASS']and qa['PASS']and len(line)==len(orig)==2304 and len(facts)==288
assert m['model_state_before']==m['model_state_after']and m['source_predict_calls']==288 and m['first_repeat_calls']==m['blank_warmup_calls']==1
count=zero=available=0
for ci,(c,rc)in enumerate(zip(m['chunks'],render['chunks'])):
 with np.load(check(c['file']),allow_pickle=False)as z,np.load(check(rc['file']),allow_pickle=False)as a:
  assert set(z.files)=={'p3_roi','rgb96','crop_valid96'}
  assert z['p3_roi'].shape==(192,65,9,9)and z['p3_roi'].dtype==np.float16
  assert z['rgb96'].shape==(192,3,96,96)and z['rgb96'].dtype==np.uint8
  assert z['crop_valid96'].shape==(192,1,96,96)and z['crop_valid96'].dtype==bool
  x=z['p3_roi'];rgb=z['rgb96'];vv=z['crop_valid96']
  assert np.array_equal(rgb,a['rgb96'].reshape(192,3,96,96))and np.array_equal(vv,a['crop_valid96'].reshape(192,1,96,96))
  assert np.isfinite(x).all()
  for j in range(192):
   i=ci*192+j;fi,k=divmod(i,8);bi,si=divmod(fi,3);r=line[i];q=np.array(qs[bi]['query8_xy'][k]);L=qs[bi]['L'][k]
   for key,val in orig[i].items():assert r[key]==val
   assert r['occurrence_index']==r['input_unit_index']==i and r['chunk_index']==ci and r['index_in_chunk']==j and r['query_xy']==q.tolist()and r['L']==L
   u=(np.arange(9)+.5)/9-.5;xx,yy=np.meshgrid(q[0]+u*L,q[1]+u*L);v=(xx>=0)&(xx<640)&(yy>=0)&(yy<640)&np.bool_(not(q==-1).all())
   assert v.dtype==bool and np.array_equal(x[j,64],v.astype(np.float16));assert not x[j,:64][:,~v].any()
   central=bool(vv[j,0,47:49,47:49].all());fp=r['input_facts'];assert fp['ROI_center_valid']==bool(v[4,4])and fp['RGB_central4_valid']==central and fp['operator_query_available']==bool(v[4,4]and central)
   assert r['p3_roi_sha256']==hashlib.sha256(x[j].tobytes()).hexdigest()
   available+=fp['operator_query_available'];zero+=int(not v.any());count+=1
assert count==2304 and sum(r['split']=='TRAIN'for r in line)==1536
r=dict(completed=True,PASS=True,code=bind(Path(__file__)),producer_seal=bind(O/'INPUT_PACK_SEAL.json'),all_queries=count,all_frames=288,chunks12=True,all_q_L_source_order_original_metadata_exact=True,all2304_RGB96_valid_bitexact_renderer=True,all_channel64_native_valid_and_outside_zero=True,all_input_refs_and_P3_value_SHA=True,factual_operator_available=available,no_valid_ROI_queries=zero,model_state_unchanged=m['model_state_before'],source_predict_calls=288,repeat_calls=1,warmup_calls=1,actual_seconds=m['seconds'],peak_GPU_allocated_bytes=m['peak_GPU_allocated_bytes'],new_model_forward_fit_GT_private_current_PnP=0)
write(H/'SAVED_PACK_REPLAY.json',r)
write(H/'FINAL_MATCHING_FEATURE_SEAL.json',dict(completed=True,PASS=True,producer_seal=bind(O/'INPUT_PACK_SEAL.json'),saved_replay=bind(H/'SAVED_PACK_REPLAY.json'),code=p['code'],generated=p['generated'],protocol=bind(H/'FEATURE_PROTOCOL_BEFORE_ACTUAL_EXECUTION.json'),execution_bindings=bind(H/'SOURCE_FEATURE_EXECUTION_BINDINGS.json'),execution_log=bind(H/'ACTUAL_FEATURE_EXECUTION.log'),actual_terminal_exit=0,no_actual_failure_restart=True,actual_source_predict=288,actual_repeat=1,actual_warmup=1,new_head_fit_private_current_PnP=0))
print(json.dumps(dict(final=bind(H/'FINAL_MATCHING_FEATURE_SEAL.json'),output_bytes=sum(f.stat().st_size for f in O.rglob('*')if f.is_file()),summary=r)))
