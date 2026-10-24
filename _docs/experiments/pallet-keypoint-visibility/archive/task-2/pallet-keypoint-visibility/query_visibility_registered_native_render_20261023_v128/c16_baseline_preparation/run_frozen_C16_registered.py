"""Inert source-only adapter: exact C11 image features + immutable C16 head.
GT/owner/reference arrays are never opened; saved evaluation follows prediction seal.
"""
import os,sys
os.environ['XFORMERS_DISABLED']='1';os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8');sys.dont_write_bytecode=True
from pathlib import Path
import argparse,ast,csv,hashlib,importlib.util,json,time,traceback,collections
import numpy as np
H=Path(__file__).resolve().parent;V=H.parent;P=V.parent
PROTOCOL=H/'FROZEN_C16_PROTOCOL_BEFORE_FORWARD.json';OUT=V/'frozen_C16_registered_baseline'
POLICY=V/'ROOT_RENDER_POLICY_BEFORE_PIXELS.json';POLICY_SHA='fc314702da18a0ff1e36e77fc20ee6db8de8d535b79b71a8473463d41179aaf6'
V125=P/'query_visibility_boundary_query_source_20261023_v125'
PARENT=V125/'feature_baseline_preparation/run_frozen_boundary_source.py';PARENT_SHA='c59d986a29668c04e19500ac1418062ad1912fdcdcfd5ed0e56209e8225be1e1'
PARENT_PROTOCOL=V125/'feature_baseline_preparation/FEATURE_BASELINE_PROTOCOL_BEFORE_FORWARD.json'
BATCH=8;VARIANTS=('BEFORE','COVER','SHAM');STYLES=('WOOD_BOARD_BLOCK','POLYMER_RIB_LEG')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def check(b):assert sha(b['path'])==b['sha256'],b['path'];return Path(b['path'])
def read(p):return json.loads(Path(p).read_text())
def write(p,v):
 with Path(p).open('x')as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def jsonl(p,rows):
 with Path(p).open('x')as f:
  for r in rows:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
def ash(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def mod(p,name):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def pure(p,names):
 nodes=[n for n in ast.parse(Path(p).read_text()).body if isinstance(n,ast.FunctionDef)and n.name in names];assert {n.name for n in nodes}==set(names)
 ns={'np':np};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(p)+'::pure_reuse','exec'),ns);return ns
def identity(r):
 value=r['id']if'id'in r else(r['occurrence_id']if'occurrence_id'in r else r['idstr'])
 for alias in('id','occurrence_id','idstr'):
  if alias in r:assert r[alias]==value
 return value
def validate_rows(rs,expected=432):
 assert len(rs)==expected and len({identity(r)for r in rs})==expected
 groups=collections.OrderedDict();scenes={}
 forbidden={'reference_xy','native_anchor3D','teacher_label','state','part_id','owner','witnesses','candidate_aux_hidden_label_GT_only','continuous_region_records','uncertainty_bbox'}
 for i,r in enumerate(rs):
  assert r['occurrence_index']==r['rgb96_row']==i and not forbidden.intersection(r)
  q8=np.asarray(r['initial_points'],np.float64);q=np.asarray(r['query_xy'],np.float64);slot=r['carrier_slot']
  assert q8.shape==(8,2)and q.shape==(2,)and isinstance(slot,int)and 0<=slot<8 and slot==r['logical_slot']and np.array_equal(q8[slot],q)and np.isfinite(q8).all()
  good=~(q8==[-1,-1]).all(1);g=q8[good];assert len(g)>=2
  box=np.r_[g.min(0),g.max(0)];assert np.all(box[2:]>box[:2])and r['context_box_xyxy']==box.tolist()
  assert r['L']==float(np.clip(.25*np.max(np.ptp(g,axis=0)),16,max(r['raw_hw'])))
  assert r['raw_hw']==[256,256]and r['style']in STYLES and r['variant']in VARIANTS and r['jitter_index']in range(9)
  assert r['full_RGB_format']=='NPZ_RGB_UINT8_HWC'and r['full_RGB_key']=='RGB_uint8'
  key=(r['style'],r['variant']);assert scenes.setdefault(key,r['full_RGB'])==r['full_RGB']
  groups.setdefault(r['query_group_id'],[]).append(r)
 assert set(scenes)=={(s,v)for s in STYLES for v in VARIANTS}and len(groups)==expected//3
 for group in groups.values():
  assert len(group)==3 and {r['variant']for r in group}==set(VARIANTS)
  for r in group:
   for k in('style','logical_slot','carrier_slot','jitter_index','offset_xy','query_xy','initial_points','L','context_box_xyxy','raw_hw'):assert r[k]==group[0][k],k
 return groups
def stored_layout(rgb,valid,N):
 assert rgb.dtype==np.uint8 and rgb.shape==(N,96,96,3)and valid.dtype==bool and valid.shape==(N,96,96)
 return rgb.transpose(0,3,1,2),valid[:,None]
def unit_key(r):
 return json.dumps([r['decoded_RGB_sha256'],r['raw_hw'],r['query_xy'],r['L'],r['context_box_xyxy'],r['rgb96_sha256'],r['valid96_sha256']],separators=(',',':'))
def load_native_rgb(binding,key,hw,decoded):
 # NPZ is lazy: only this whitelisted image member is read, never owner/depth/GT arrays.
 assert key=='RGB_uint8';check(binding)
 with np.load(binding['path'],allow_pickle=False)as f:rgb=np.array(f[key],copy=True)
 assert rgb.dtype==np.uint8 and rgb.shape==tuple(hw)+(3,)and ash(rgb)==decoded
 return rgb
def authorize(entry):
 e=read(entry);assert e['approved_source_C16_baseline']is True and all(e[k]is False for k in('fit_allowed','current_allowed','V112_allowed','initializer_allowed','PnP_allowed','automatic_retry'))
 for k in('code','protocol','generated','render_policy','render_complete','crop_complete','input_rows','stored_crops'):check(e[k])
 assert sha(POLICY)==POLICY_SHA and sha(PARENT)==PARENT_SHA
 p=read(PROTOCOL);assert e['code']==p['code']==bind(__file__)and e['protocol']==bind(PROTOCOL)and e['render_policy']==p['render_policy']==bind(POLICY)and e['generated']==p['generated'];assert read(e['generated']['path'])['PASS']
 for k in('parent_code','parent_protocol','C11_producer','C11_protocol','asset_runtime','asset_plan','weight','C8_preprocess','C9_target_preprocess','local_crop_code','target_crop_code','source_model_freeze','checkpoint','head_class_code'):check(p[k])
 for b in p['official_source_bindings']:check(b)
 rc=read(e['render_complete']['path']);cc=read(e['crop_complete']['path']);assert rc['completed']and rc['PASS']and rc['scenes']==6 and cc['completed']and cc['PASS']and cc['input_manifest']==e['input_rows']and cc['stored_crops']==e['stored_crops']and cc['render_complete']==e['render_complete']
 # GT_only descriptor in CROP_COMPLETE is deliberately neither hashed nor opened.
 freeze=read(p['source_model_freeze']['path']);assert freeze['completed']and freeze['selected_update']==4500 and freeze['checkpoint']==p['checkpoint']and freeze['head_class_code']==p['head_class_code']and freeze['threshold_logit']==0 and not freeze['current_used_for_fit_or_selection']
 packet=read(e['input_rows']['path']);assert packet['completed'];rs=packet['rows'];validate_rows(rs)
 allowed={b['sha256']:b['path']for b in rc['outputs']}
 for r in rs:assert allowed.get(r['full_RGB']['sha256'])==r['full_RGB']['path']
 return e,p,rs
def run(entry):
 e,p,rs=authorize(entry);N=len(rs);assert not OUT.exists(),'Preserve failure; no automatic restart.'
 import torch
 assert torch.__version__==p['runtime']['torch']=='2.1.1+cu118'and torch.cuda.is_available()and Path(sys.executable).resolve()==Path(p['runtime']['executable']).resolve()
 OUT.mkdir();write(OUT/'STARTED.json',dict(entry=bind(entry),teacher_body_or_hash_read=False,source_only=True));started=time.monotonic()
 try:
  torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
  producer=mod(p['C11_producer']['path'],'v128_exact_C11');c8=mod(p['C8_preprocess']['path'],'v128_exact_C8');c9=mod(p['C9_target_preprocess']['path'],'v128_exact_C9');tm=mod(p['target_crop_code']['path'],'v128_target_sampler');crop=producer.pure_crop();availability=pure(PARENT,('factual_availability',))['factual_availability']
  with np.load(e['stored_crops']['path'],allow_pickle=False)as f:stored_rgb=f['rgb96'];stored_valid=f['valid96'];index=f['occurrence_index']
  assert np.array_equal(index,np.arange(N));rgb96,valid96=stored_layout(stored_rgb,stored_valid,N);lookup={};reps=[];unitmap=[]
  for i,r in enumerate(rs):
   assert ash(stored_rgb[i])==r['rgb96_sha256']and ash(stored_valid[i])==r['valid96_sha256'];key=unit_key(r)
   if key not in lookup:lookup[key]=len(reps);reps.append(i)
   else:j=reps[lookup[key]];assert np.array_equal(rgb96[i],rgb96[j])and np.array_equal(valid96[i],valid96[j])
   unitmap.append(lookup[key])
  um=np.array(unitmap,np.int32);U=len(reps);assert np.array_equal(np.unique(um),np.arange(U));cache={}
  def native(r):
   key=(r['full_RGB']['path'],r['full_RGB']['sha256'])
   if key not in cache:cache[key]=load_native_rgb(r['full_RGB'],r['full_RGB_key'],r['raw_hw'],r['decoded_RGB_sha256'])
   return cache[key]
  sys.path.insert(0,p['official_repo']);from dinov2.hub.backbones import dinov2_vits14_reg
  encoder=dinov2_vits14_reg(pretrained=False);state=torch.load(p['weight']['path'],map_location='cpu',weights_only=True);encoder.load_state_dict(state,strict=True);del state;encoder=encoder.cuda().float().eval().requires_grad_(False);encstate=producer.state_sha(encoder);assert encstate==read(p['asset_runtime']['path'])['state_before_SHA256']
  features=[];facts=[];factual=[];context=[];Q=[];calls=0;repeated=False
  for start in range(0,U,BATCH):
   batch=reps[start:start+BATCH];n=len(batch);lr=[];lv=[];tr=[];tv=[];fractions=[];central4=[]
   for ix in batch:
    r=rs[ix];image=native(r);a,v=crop(image,np.asarray(r['query_xy'],np.float64),r['L'],96);assert np.array_equal(a,rgb96[ix])and np.array_equal(v,valid96[ix]),('Exact native local96 replay',ix)
    target,vt=tm.sample_target_context(image,r['context_box_xyxy']);fa,ca,aa,c4=availability(r['query_xy'],r['raw_hw'],v,vt)
    assert fa==r['factual_query_available']and ca==r['context_available']and aa==r['query_available'];frac=[np.float32(vt.mean()),np.float32(v.mean()),np.float32(v[:,32:64,32:64].mean())]
    lr.append(a);lv.append(v);tr.append(target.transpose(2,0,1));tv.append(vt[None]);fractions.append(frac);central4.append(c4);factual.append(fa);context.append(ca)
    facts.append(dict(input_unit_index=len(facts),representative_occurrence_index=ix,full_RGB=r['full_RGB'],query_xy=r['query_xy'],L=r['L'],context_box_xyxy=r['context_box_xyxy'],stored_RGB96_HWC_sha256=ash(stored_rgb[ix]),stored_valid96_HW_sha256=ash(stored_valid[ix]),canonical_RGB96_CHW_sha256=ash(a),canonical_valid96_1HW_sha256=ash(v),central32_CHW_sha256=ash(a[:,32:64,32:64]),target_RGB224_F32_sha256=ash(target),target_valid_sha256=ash(vt),fractions=[float(t)for t in frac],factual_query_available=fa,context_available=ca,query_available=aa))
   local,_=c8.preprocess_views(torch.as_tensor(np.stack(lr),device='cuda'),torch.as_tensor(np.stack(lv),device='cuda'));target=c9.preprocess_target(torch.as_tensor(np.stack(tr),device='cuda'),torch.as_tensor(np.stack(tv),device='cuda'));views=torch.cat((target,local),0);assert views.shape==(3*n,3,224,224)and len(views)<=24
   with torch.inference_mode():
    z=encoder.forward_features(views);calls+=1
    if not repeated:
     again=encoder.forward_features(views);calls+=1;assert all(torch.equal(z[k],again[k])for k in('x_norm_clstoken','x_norm_patchtokens'));repeated=True
    cl=z['x_norm_clstoken'];pt=z['x_norm_patchtokens'];x,inside=producer.assemble(cl[:n],pt[:n],cl[n:2*n],pt[n:2*n],cl[2*n:],[rs[i]['query_xy']for i in batch],[rs[i]['context_box_xyxy']for i in batch],fractions,central4)
   features.append(x.cpu().numpy());Q.extend(inside.tolist())
  x=np.concatenate(features);fa=np.array(factual,bool);ca=np.array(context,bool);qi=np.array(Q,bool);a=(fa&ca)[um]
  assert x.shape==(U,1928)and x.dtype==np.float32 and np.isfinite(x).all()and np.array_equal(x[:,1927],qi.astype(np.float32))and not x[~qi,384:768].any()and calls==(U+7)//8+1
  for i,r in enumerate(rs):assert bool(fa[um[i]])==r['factual_query_available']and bool(ca[um[i]])==r['context_available']and bool(a[i])==r['query_available']
  assert producer.state_sha(encoder)==encstate and all(not t.requires_grad for t in encoder.parameters())and all(not t.training for t in encoder.modules());del encoder
  ff=OUT/'SOURCE_FEATURES_INPUT_ONLY.npz';np.savez_compressed(ff,features=x,occurrence_unit=um,query_available=a,factual_query_available=fa[um],context_available=ca[um],unit_Q_inside=qi,global_occurrence_index=np.arange(N,dtype=np.int32));factfile=OUT/'UNIQUE_INPUT_UNIT_FACTS_INPUT_ONLY.jsonl';jsonl(factfile,facts)
  fr=OUT/'FEATURE_COMPLETE.json';write(fr,dict(completed=True,PASS=True,entry=bind(entry),code=e['code'],protocol=e['protocol'],input_manifest=e['input_rows'],stored_crops=e['stored_crops'],output=bind(ff),inputfacts=bind(factfile),occurrences=N,unique_input_units=U,features_shape=[U,1928],encoder_calls=calls,primary_calls=(U+7)//8,repeat_calls=1,first_batch_repeat_exact=True,encoder_state_before=encstate,encoder_state_after=encstate,model_state_unchanged=True,canonical_local96_replay_all_exact=True,native_npz_members_read=['RGB_uint8'],GT_body_or_hash_read=False))
  headmod=mod(p['head_class_code']['path'],'v128_frozen_C16_head');payload=torch.load(p['checkpoint']['path'],map_location='cpu',weights_only=True);assert payload['update']==4500 and payload['head_class_code']==p['head_class_code'];head=headmod.SpatialTargetQueryHead().cuda().float().eval().requires_grad_(False);head.load_state_dict(payload['state_dict'],strict=True);del payload;assert sum(t.numel()for t in head.parameters())==221825;hs=producer.state_sha(head)
  with torch.inference_mode():raw=head(torch.as_tensor(x,device='cuda'));again=head(torch.as_tensor(x,device='cuda'));assert torch.equal(raw,again)
  raw=raw.cpu().numpy().astype(np.float32);assert raw.shape==(U,)and np.isfinite(raw).all()and producer.state_sha(head)==hs;z=np.where(a,raw[um],0).astype(np.float32);pred=np.where(a,(z>=0).astype(np.int8),-1).astype(np.int8)
  pf=OUT/'RAW_C16_PREDICTIONS_INPUT_ONLY.npz';np.savez_compressed(pf,query_logit=z,raw_head_logit_before_U=raw[um],prediction=pred,query_available=a,unit_raw_logit=raw,occurrence_unit=um,global_occurrence_index=np.arange(N,dtype=np.int32),occurrence_id=np.array([identity(r)for r in rs]))
  table=OUT/'C16_PREDICTIONS_INPUT_ONLY.csv';rows=[dict(occurrence_index=i,occurrence_id=identity(r),query_group_id=r['query_group_id'],style=r['style'],variant=r['variant'],logical_slot=r['logical_slot'],jitter_index=r['jitter_index'],qx=r['query_xy'][0],qy=r['query_xy'][1],L=r['L'],query_available=bool(a[i]),hidden_logit=float(z[i]),prediction=int(pred[i]))for i,r in enumerate(rs)]
  with table.open('x',newline='')as f:writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
  seal=OUT/'PREDICTION_SEAL.json';write(seal,dict(completed=True,PASS=True,entry=bind(entry),code=e['code'],protocol=e['protocol'],source_model_freeze=p['source_model_freeze'],checkpoint=p['checkpoint'],selected_update=4500,input_manifest=e['input_rows'],crop_complete=e['crop_complete'],render_complete=e['render_complete'],feature_receipt=bind(fr),features=bind(ff),predictions=bind(pf),scoreCSV=bind(table),rows=N,unique_units=U,available=int(a.sum()),factual_U=int((~fa[um]).sum()),context_U=int((~ca[um]).sum()),final_U=int((~a).sum()),threshold_logit=0,decoder='D z<0; E z>=0; factual/context unavailable U first',head_calls=2,head_repeat_exact=True,head_state_before=hs,head_state_after=hs,model_state_unchanged=True,GT_body_or_hash_read=False,no_label_or_qpixel_membership_in_features=True,source_registered_only_not_current_performance=True,seconds=time.monotonic()-started,fit_current_V112_PnP=0));print(json.dumps(dict(completed=True,prediction_seal=bind(seal),rows=N,unique_units=U,available=int(a.sum()))),flush=True)
 except BaseException:write(OUT/'FAILED_NO_RESTART.json',dict(completed=False,traceback=traceback.format_exc(),automatic_retry=False));raise
def prepare():
 assert sha(POLICY)==POLICY_SHA and sha(PARENT)==PARENT_SHA;parent=read(PARENT_PROTOCOL);proof=H/'GENERATED_BASELINE_TESTS.json';assert read(proof)['PASS']and read(proof)['code']==bind(__file__)
 freeze=read(parent['source_model_freeze']['path']);assert freeze['selected_update']==4500 and freeze['threshold_logit']==0 and not freeze['current_used_for_fit_or_selection']
 keys=('asset_runtime','asset_plan','weight','official_repo','official_revision','official_source_bindings','runtime','C8_preprocess','C9_target_preprocess','local_crop_code','target_crop_code','C11_producer','C11_protocol','source_model_freeze','checkpoint','head_class_code')
 p={k:parent[k]for k in keys};p.update(completed=True,code=bind(__file__),generated=bind(proof),render_policy=bind(POLICY),parent_code=bind(PARENT),parent_protocol=bind(PARENT_PROTOCOL),output=str(OUT),feature_operator=parent['feature_operator'],availability=parent['availability'],input_contract='INPUT_ROWS.json completed/rows432, HWC96/HWvalid stored hashes; canonical CHW transpose only; onecontext+full+central max24views; no teacher/owner/reference arrays.',native_member_whitelist=['RGB_uint8'],checkpoint_policy='Immutable C16 source-selected4500, frozen before this registered source; raw0, descriptive only.',evaluation_boundary='GT-only bodies/hashes never accessed by this inference adapter; join only after PREDICTION_SEAL.',actual_source_pixels_features_weights_encoder_head_fit_current_V112_calls=0);write(PROTOCOL,p)
 placeholder=lambda name:dict(path=str(V/'source_crops_v1'/name),sha256='ROOT_BIND_AFTER_CROP_COMPLETION')
 template=dict(approved_source_C16_baseline=False,fit_allowed=False,current_allowed=False,V112_allowed=False,initializer_allowed=False,PnP_allowed=False,automatic_retry=False,code=p['code'],protocol=bind(PROTOCOL),generated=p['generated'],render_policy=p['render_policy'],render_complete=dict(path=str(V/'actual_render_v1/RENDER_COMPLETE.json'),sha256='ROOT_BIND_COMPLETED_RENDER'),crop_complete=placeholder('CROP_COMPLETE.json'),input_rows=placeholder('INPUT_ROWS.json'),stored_crops=placeholder('RGB96_INPUT_ONLY.npz'));write(H/'ROOT_ENTRY_TEMPLATE.json',template)
 print(json.dumps(dict(code=p['code'],protocol=bind(PROTOCOL),template=bind(H/'ROOT_ENTRY_TEMPLATE.json'),command=p['runtime']['executable']+' '+str(Path(__file__).resolve())+' --entry ROOT_ACTUAL_C16_BASELINE_ENTRY.json',actual_forward=0)),flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');ap.add_argument('--entry',type=Path);a=ap.parse_args()
 if a.prepare:prepare()
 elif a.entry:run(a.entry)
 else:raise SystemExit('Use --prepare or --entry ROOT_ACTUAL_C16_BASELINE_ENTRY.json')
