"""Frozen C111 RGB-only native features, fixed230 frames/all1840 slots.
No R0/P3/classifier/GT calls. Exact17 views/frame plus sealed old10 replay.
"""
import os,sys
os.environ['XFORMERS_DISABLED']='1';os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8');sys.dont_write_bytecode=True
from pathlib import Path
import argparse,ast,hashlib,importlib.util,json,time,traceback
import numpy as np
H=Path(__file__).resolve().parent;V=H.parent;P=V.parent
POLICY=V/'ROOT_POLICY_BEFORE_NEW_SOURCE_BODIES.json';POLICY_SHA='46b3b077edfc78d113dcd0b38771bb3df96053a3741002ac774bc1c9866e2334'
V147=P/'query_visibility_public_native_features_20261024_v147'
PARENT=V147/'feature_preparation/PROTOCOL_BEFORE_DINO_FORWARD.json'
HELPER=V147/'feature_preparation/encode_saved_public512.py';HELPER_SHA='2809691ec4b773c7c336733579b072d67460f3720c71adcdba17b2f9d528b17c'
INTAKE=V/'full_source_v1/FULL_SOURCE_INTAKE_COMPLETE.json';R0=V/'actual_full230_initializer_v1/PREDICTION_SEAL.json'
V152=P/'query_visibility_mr6d_actual_query_bridge_20261024_v152';OLD_R0=V152/'actual_mr6d_initializer_v1/PREDICTION_SEAL.json';OLD_R0_SHA='041ba0669c2067dbae43c4885dc3b9c897fcd66c286482dfacc2d493ebb6db08'
OLD_CROPS=V152/'original_query_evidence/EVIDENCE_RENDER_COMPLETE.json';OLD_CROPS_SHA='b2c5f3ea7d17c4e1f8f0aa174b4714e7e261e8e2e545ec233e1b4fcf7bf21d20'
PROTOCOL=H/'FEATURE_PROTOCOL_BEFORE_DINO_FORWARD.json';INPUTS=H/'FEATURE_FRAMES230_INPUT_ONLY.json';GENERATED=H/'FEATURE_GENERATED_CHECKS.json';OUT=V/'full230_native_DINO_v1';N=1840;FRAMES=230
BOX_MODE='FROZEN_INITIAL_Q8_UNCLIPPED_UNEXPANDED_MINMAX'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for block in iter(lambda:f.read(1<<20),b''):h.update(block)
 return h.hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def check(b):assert sha(b['path'])==b['sha256'],b['path'];return Path(b['path'])
def read(p):return json.loads(Path(p).read_text())
def lines(p):return[json.loads(s)for s in Path(p).read_text().splitlines()if s]
def write(p,x):
 with Path(p).open('x')as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def ash(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def exact(a,b):return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
def clean(x):
 if isinstance(x,np.ndarray):return clean(x.tolist())
 if isinstance(x,np.generic):return clean(x.item())
 if isinstance(x,float):return x if np.isfinite(x)else None
 if isinstance(x,dict):return{k:clean(v)for k,v in x.items()}
 if isinstance(x,(tuple,list)):return[clean(v)for v in x]
 return x

def operators(p):
 # Only NumPy/Torch math definitions are imported; no encoder/model construction.
 assert sha(HELPER)==HELPER_SHA
 spec=importlib.util.spec_from_file_location('v153_exact_C111_math',HELPER);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 return m.operators(p)

def input_packet(intake_binding,r0_binding):
 intake=read(check(intake_binding));r0=read(check(r0_binding));assert intake['completed']and intake['PASS']and r0['completed']and r0['PASS']and r0['GT_read']is False
 assert intake['frames']==r0['frames']==230 and r0['rows']==1840 and intake['policy']==r0['policy']==bind(POLICY)
 assert r0['canonical_transport_seal']==intake_binding
 manifest=read(check(intake['RGB_manifest']));rows=manifest['rows'];assert len(rows)==230
 files={Path(x['path']).name:x for x in r0['outputs']};props=lines(check(files['FROZEN_PROPOSALS_ALL230_INPUT_ONLY.jsonl']));queries=lines(check(files['ORIGINAL_NATIVE_QUERY1840_INPUT_ONLY.jsonl']))
 assert len(props)==230 and len(queries)==1840
 frames=[]
 for i,(r,s)in enumerate(zip(rows,props)):
  assert r['frame_index']==s['frame_index']==i
  for k in ('id','source_id','source','split','raw_hw','canonical_RGB_NPY','decoded_RGB_sha256'):assert r[k]==s[k]
  selected=None if not s['candidates']else int(np.argmax([c['score']for c in s['candidates']]))
  assert selected==s['selected_index']
  qq=queries[i*8:i*8+8]
  for k,q in enumerate(qq):assert q['frame_index']==i and q['frame_id']==r['id']and q['source_id']==r['source_id']and q['split']==r['split']and q['native_slot']==k and q['query_id']==r['id']+':native'+str(k)
  frames.append({k:r[k]for k in ('frame_index','id','source_id','split','raw_hw','canonical_RGB_NPY','decoded_RGB_sha256')})
 assert sum(r['split']=='SOURCE_DEVELOPMENT'for r in frames)==185 and sum(r['split']=='SOURCE_VALIDATION'for r in frames)==45
 return dict(completed=True,RGB_GT_free=True,frames=frames,query_npz=files['ORIGINAL_NATIVE_QUERY1840_AND_VALID9X9_INPUT_ONLY.npz'],queries=files['ORIGINAL_NATIVE_QUERY1840_INPUT_ONLY.jsonl'],proposals=files['FROZEN_PROPOSALS_ALL230_INPUT_ONLY.jsonl'],intake=intake_binding,initializer=r0_binding,context_box_mode=BOX_MODE)

def generated(p,op):
 import torch
 torch.set_num_threads(1)
 rgb=np.arange(40*50*3,dtype=np.uint8).reshape(40,50,3)
 a,v=op['crop'](rgb,[2.,3.],16.,96);assert a.shape==(3,96,96)and v.shape==(1,96,96)
 A=np.stack([a]*8);M=np.stack([v]*8);B=A.copy();B[~np.broadcast_to(M,A.shape)]=251
 lv,_=op['preprocess_views'](torch.from_numpy(A),torch.from_numpy(M));other,_=op['preprocess_views'](torch.from_numpy(B),torch.from_numpy(M));assert torch.equal(lv,other)
 q=np.array([[10.,12.],[30.,12.],[30.,28.],[10.,28.],[12.,14.],[28.,14.],[28.,26.],[12.,26.]])
 box=op['initial_context_box'](q);assert box.tolist()==[10.,12.,30.,28.]
 assert op['initial_context_box'](np.full((8,2),-1.))is None
 target,vt=op['sample_target_context'](rgb,box);tv=op['preprocess_target'](torch.from_numpy(target.transpose(2,0,1)[None]),torch.from_numpy(vt[None,None]));assert torch.cat((tv,lv),0).shape==(17,3,224,224)
 cl=torch.arange(17*384,dtype=torch.float32).reshape(17,384)+1;pt=torch.arange(17*256*384,dtype=torch.float32).reshape(17,256,384)/1e4+1
 fractions=np.array([[vt.mean(),v.mean(),v[:,32:64,32:64].mean()]]*8,np.float32)
 z,inside=op['assemble'](cl[:1].repeat_interleave(8,0),pt[:1].repeat_interleave(8,0),cl[1:9],pt[1:9],cl[9:],q,[box]*8,fractions,[True]*8)
 assert z.shape==(8,1928)and torch.isfinite(z).all()
 nans=np.full((8,2),np.nan);zero=np.zeros((8,3),np.float32)
 zz,ii=op['assemble'](cl[:1].repeat_interleave(8,0),pt[:1].repeat_interleave(8,0),cl[1:9],pt[1:9],cl[9:],nans,[[np.nan]*4]*8,zero,[False]*8)
 assert not zz.any()and not ii.any()
 # A sentinel can lie geometrically inside a negative q8 context, but the
 # original C111 sampler explicitly excludes [-1,-1] before inside testing.
 mixed=q.copy();mixed[0]=[-1.,-1.];mixed[1]=[-5.,-5.];mbox=op['initial_context_box'](mixed)
 assert mbox[0]<-1<mbox[2]and mbox[1]<-1<mbox[3]
 mz,mi=op['assemble'](cl[:1].repeat_interleave(8,0),pt[:1].repeat_interleave(8,0),cl[1:9],pt[1:9],cl[9:],mixed,[mbox]*8,np.ones((8,3),np.float32),[True]*8)
 construct=np.isfinite(mixed).all(1)&~(mixed==[-1,-1]).all(1)
 masked=torch.where(torch.as_tensor(construct)[:,None],mz,torch.zeros_like(mz))
 assert not mi[0]and not masked[0].any()and np.array_equal(masked[:,1927].numpy(),mi.astype(np.float32))
 write(GENERATED,dict(completed=True,PASS=True,code=bind(__file__),invented_only=True,exact_C111_C8_C9_operator_reuse=True,invalid_RGB_meanfill_invariance=True,view_order_target_full8_central8=True,GT_free_q8_unclipped_minmax=True,noquery_all8_zero_placeholder_features=True,mixed_sentinel_inside_negative_context_remains_undefined=True,source_RGB_features_weights_GT_model_body_reads_or_calls=0))

def prepare():
 assert sha(POLICY)==POLICY_SHA and sha(OLD_R0)==OLD_R0_SHA and sha(OLD_CROPS)==OLD_CROPS_SHA and sha(HELPER)==HELPER_SHA
 parent=read(PARENT);packet=input_packet(bind(INTAKE),bind(R0))
 p=dict(completed=True,prepared_only=True,code=bind(__file__),policy=bind(POLICY),parent_protocol=bind(PARENT),helper_code=bind(HELPER),
  **{k:parent[k]for k in ('C111_producer','initial_context_code','native_grid_code','C8_preprocess','C9_target_preprocess','local_crop_code','target_crop_code','asset_runtime','asset_plan','weight','official_repo','official_revision','official_source_bindings','runtime','expected_encoder_state')},
  source_intake=bind(INTAKE),source_initializer=bind(R0),prior_fixed10_R0=bind(OLD_R0),prior_fixed10_factual_crops=bind(OLD_CROPS),rows=N,frames=FRAMES,output=str(OUT),context_box_mode=BOX_MODE,
  views_per_call=17,primary_encoder_calls=230,fixed10_replay_calls=10,total_encoder_calls=240,
  R0_P3_or_classifier_calls=0,labels_pose_K_depth_masks_reference_reads=0,
  fixed10_parity_scope='Prior sealed R0/canonical/factual96 bytes; MR10 has no prior DINO cache. New full-run fixed10 features/raw tokens replayed bitexact from recomputed identical input17views.',
  feature_layout={**parent['feature_layout'],'availability':'Full230 original continuous validity; nearest/context/central4/Qinside/constructed stored separately, no classifier gating in this phase'},availability='query_available is original native continuous flag only (V147-compatible metadata), nearest/context/central4/Qinside/feature_constructed remain separate; no new classifier U rule',
  noquery='All1840 kept. Nonfinite/sentinel/L-invalid local crops are zero+constructedfalse. Fixed17-view frame shape uses meanfilled placeholders; all unconstructed feature rows zeroed, never semantic supervision.',automatic_retry=False)
 for k in ('C111_producer','initial_context_code','native_grid_code','C8_preprocess','C9_target_preprocess','local_crop_code','target_crop_code','asset_runtime','asset_plan'):check(p[k])
 op=operators(p);generated(p,op);write(INPUTS,packet);p['inputs']=bind(INPUTS);p['generated']=bind(GENERATED);write(PROTOCOL,p)
 t=H/'ROOT_FEATURE_ENTRY_TEMPLATE.json';write(t,dict(approved_actual_DINO_feature_extraction=False,code=p['code'],policy=p['policy'],protocol=bind(PROTOCOL),inputs=p['inputs'],generated=p['generated'],source_only=True,fit_allowed=False,classifier_allowed=False,initializer_allowed=False,current_allowed=False,PnP_allowed=False,semantic_pose_depth_reads_allowed=False,automatic_retry=False,output=str(OUT)))
 write(H/'FEATURE_PREPARATION_READY.json',dict(completed=True,prepared_only=True,code=bind(__file__),protocol=bind(PROTOCOL),generated=p['generated'],inputs=p['inputs'],template=bind(t),actual_learned_calls=0))
 print(json.dumps(dict(completed=True,prepared_only=True,protocol=bind(PROTOCOL),code=bind(__file__),template=bind(t))),flush=True)

def run(entry):
 e=read(entry);assert e['approved_actual_DINO_feature_extraction']is True and e['source_only']is True
 for k in ('fit_allowed','classifier_allowed','initializer_allowed','current_allowed','PnP_allowed','semantic_pose_depth_reads_allowed','automatic_retry'):assert e[k]is False
 p=read(check(e['protocol']));assert e['protocol']==bind(PROTOCOL)
 for k in ('code','policy','inputs','generated'):assert e[k]==p[k];check(p[k])
 assert e['code']==bind(__file__)and e['policy']==bind(POLICY)and sha(POLICY)==POLICY_SHA and e['output']==p['output']==str(OUT)and not OUT.exists()
 for k in ('parent_protocol','helper_code','C111_producer','initial_context_code','native_grid_code','C8_preprocess','C9_target_preprocess','local_crop_code','target_crop_code','asset_runtime','asset_plan','weight','source_intake','source_initializer','prior_fixed10_R0','prior_fixed10_factual_crops'):check(p[k])
 for k in p['official_source_bindings']:check(k)
 packet=input_packet(p['source_intake'],p['source_initializer']);assert packet==read(check(p['inputs']));frames=packet['frames']
 with np.load(check(packet['query_npz']),allow_pickle=False)as f:qdata={k:np.array(f[k],copy=True)for k in f.files}
 assert np.array_equal(qdata['global_index'],np.arange(N))and qdata['query_xy'].shape==(N,2)
 oldseal=read(check(p['prior_fixed10_R0']));oldfiles={Path(b['path']).name:check(b)for b in oldseal['outputs']};oldprops=lines(oldfiles['FROZEN_PROPOSALS_ALL10_INPUT_ONLY.jsonl'])
 oldevidence=read(check(p['prior_fixed10_factual_crops']));oldpackets={r['source_id']:r['crop_packet']for r in oldevidence['sheets']};assert len(oldpackets)==10
 oldids=[r['source_id']for r in oldprops];assert set(oldids)==set(oldpackets)and set(oldids)<=set(r['source_id']for r in frames)
 import torch
 assert str(torch.__version__)==p['runtime']['torch']=='2.1.1+cu118'and np.__version__=='1.26.4'and Path(sys.executable).resolve()==Path(p['runtime']['executable']).resolve()and torch.cuda.is_available()
 OUT.mkdir();write(OUT/'INPUT_SEAL_BEFORE_ENCODER.json',dict(completed=True,PASS=True,entry=bind(entry),inputs=p['inputs'],initializer=p['source_initializer'],intake=p['source_intake'],rows=N,frames=FRAMES,GT_read=False))
 write(OUT/'STARTED.json',dict(completed=False,entry=bind(entry),automatic_retry=False));started=time.monotonic()
 try:
  torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
  torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
  op=operators(p);sys.path.insert(0,p['official_repo']);from dinov2.hub.backbones import dinov2_vits14_reg
  encoder=dinov2_vits14_reg(pretrained=False);weights=torch.load(p['weight']['path'],map_location='cpu',weights_only=True);assert tuple(weights['register_tokens'].shape)==(1,4,384)
  encoder.load_state_dict(weights,strict=True);del weights;encoder=encoder.cuda().float().eval().requires_grad_(False)
  assert sum(t.numel()for t in encoder.parameters())==22058112
  before=op['state_sha'](encoder);assert before==p['expected_encoder_state']
  calls=0;checked=[];saved_old={};facts=[];crop_manifest=[];features=[];allinside=[];allconstruct=[];allc4=[];allctx=[];allfrac=[];allboxes=[]
  def encode(i,save):
   nonlocal calls
   r=frames[i];sl=slice(i*8,i*8+8);q=qdata['query_xy'][sl];L=qdata['L'][sl]
   rgb=np.load(check(r['canonical_RGB_NPY']),allow_pickle=False);assert rgb.dtype==np.uint8 and list(rgb.shape)==r['raw_hw']+[3]and ash(rgb)==r['decoded_RGB_sha256']
   construct=qdata['finite_non_sentinel'][sl]&np.isfinite(L)&(L>0)
   local=np.zeros((8,3,96,96),np.uint8);vmask=np.zeros((8,1,96,96),bool)
   for k in range(8):
    if construct[k]:local[k],vmask[k]=op['crop'](rgb,q[k],L[k],96)
   hwc=np.ascontiguousarray(local.transpose(0,2,3,1));valid=vmask[:,0]
   if r['source_id']in oldpackets:
    old=oldpackets[r['source_id']]
    with np.load(check(old),allow_pickle=False)as f:assert exact(hwc,f['rgb96'])and exact(valid,f['valid96'])
    checked.append(old)
   box=op['initial_context_box'](q)
   if op['valid_box'](box):target,vt=op['sample_target_context'](rgb,box)
   else:target=np.zeros((224,224,3),np.float32);vt=np.zeros((224,224),bool)
   boxes=[box if box is not None else[np.nan]*4]*8
   frac=np.array([[np.float32(vt.mean()),np.float32(vmask[k].mean()),np.float32(vmask[k,:,32:64,32:64].mean())]for k in range(8)],np.float32)
   c4=np.array([bool(vmask[k,:,47:49,47:49].all())for k in range(8)])
   lv,_=op['preprocess_views'](torch.as_tensor(local,device='cuda'),torch.as_tensor(vmask,device='cuda'))
   tv=op['preprocess_target'](torch.as_tensor(target.transpose(2,0,1)[None],device='cuda'),torch.as_tensor(vt[None,None],device='cuda'))
   views=torch.cat((tv,lv),0);assert views.shape==(17,3,224,224)and views.dtype==torch.float32
   with torch.inference_mode():
    z=encoder.forward_features(views);calls+=1;cl,pt=z['x_norm_clstoken'],z['x_norm_patchtokens']
    x,inside=op['assemble'](cl[:1].repeat_interleave(8,0),pt[:1].repeat_interleave(8,0),cl[1:9],pt[1:9],cl[9:],q,boxes,frac,c4)
    x=torch.where(torch.as_tensor(construct,device='cuda')[:,None],x,torch.zeros_like(x))
   xn=x.cpu().numpy();raw={k:z[k].cpu().numpy().copy()for k in ('x_norm_clstoken','x_norm_patchtokens')}if r['source_id']in oldids else None
   if save:
    cropfile=OUT/f'FRAME_{i:04d}_FACTUAL96_INPUT_ONLY.npz';np.savez_compressed(cropfile,rgb96=hwc,valid96=valid,global_index=np.arange(i*8,i*8+8,dtype=np.int32),crop_constructed=construct)
    crop_manifest.append(dict(frame_index=i,source_id=r['source_id'],crop_packet=bind(cropfile)))
    for k in range(8):facts.append(clean(dict(global_index=i*8+k,frame_index=i,frame_id=r['id'],source_id=r['source_id'],split=r['split'],native_slot=k,query_id=str(qdata['query_id'][i*8+k]),query_xy=q[k],L=L[k],feature_constructed=construct[k],native_continuous_valid=qdata['continuous_valid'][i*8+k],nearest_valid=qdata['nearest_valid'][i*8+k],context_box_xyxy=box,Q_inside=inside[k],local_central4_available=c4[k],context_available=bool(vt.any()),RGB96_HWC_sha256=ash(hwc[k]),RGB96_CHW_sha256=ash(local[k]),valid96_sha256=ash(valid[k]),target224_F32_HWC_sha256=ash(target),target_valid_sha256=ash(vt),fractions=frac[k])))
   checked.append(r['canonical_RGB_NPY'])
   return xn,np.array(inside,bool),construct,c4,bool(vt.any()),frac,np.array(box if box is not None else[np.nan]*4,np.float64),raw
  for i,r in enumerate(frames):
   x,inside,construct,c4,ctx,frac,box,raw=encode(i,True)
   features.append(x);allinside.extend(inside);allconstruct.extend(construct);allc4.extend(c4);allctx.extend([ctx]*8);allfrac.extend(frac);allboxes.extend([box]*8)
   if r['source_id']in oldids:saved_old[r['source_id']]=(x.copy(),raw)
   if(i+1)%25==0:print(json.dumps(dict(frames=i+1,rows=(i+1)*8,encoder_calls=calls,seconds=time.monotonic()-started)),flush=True)
  byid={r['source_id']:i for i,r in enumerate(frames)};parity=[]
  for sid in oldids:
   new,*rest=encode(byid[sid],False);raw=rest[-1];old,oldraw=saved_old[sid]
   assert exact(new,old)and all(exact(raw[k],oldraw[k])for k in oldraw)
   parity.append(dict(source_id=sid,frame_index=byid[sid],all8_features1928_SHA256=ash(new),features_and_raw_tokens_bitexact=True,prior_factual96_RGB_valid_bytes_exact=True))
  assert calls==240 and len(parity)==10
  x=np.concatenate(features);inside=np.asarray(allinside,bool);constructed=np.asarray(allconstruct,bool);available=qdata['continuous_valid'].copy()
  assert x.shape==(N,1928)and x.dtype==np.float32 and np.isfinite(x).all()and not x[~constructed].any()
  assert np.array_equal(x[:,1927],inside.astype(np.float32))and not x[~inside,384:768].any()
  after=op['state_sha'](encoder);assert before==after and all(not t.requires_grad for t in encoder.parameters())and all(not m.training for m in encoder.modules())
  for b in checked+[p['weight'],packet['query_npz'],p['source_initializer'],p['source_intake']]:check(b)
  output=OUT/'MR6D_NATIVE1840_FEATURES_INPUT_ONLY.npz'
  np.savez_compressed(output,features=x,global_index=qdata['global_index'],frame_index=qdata['frame_index'],native_slot=qdata['native_slot'],query_id=qdata['query_id'],frame_id=qdata['frame_id'],source_id=np.repeat([r['source_id']for r in frames],8),split=qdata['split'],query_xy=qdata['query_xy'],L=qdata['L'],no_detection=qdata['no_detection'],query_available=available,native_continuous_valid=available,nearest_valid=qdata['nearest_valid'],feature_constructed=constructed,local_central4_available=np.asarray(allc4,bool),context_available=np.asarray(allctx,bool),support_fractions=np.asarray(allfrac,np.float32),context_box_xyxy=np.asarray(allboxes,np.float64),unit_Q_inside=inside,occurrence_unit=np.arange(N,dtype=np.int32))
  factfile=OUT/'FEATURE_INPUT_FACTS1840_INPUT_ONLY.jsonl'
  with factfile.open('x')as f:
   for row in facts:f.write(json.dumps(row,allow_nan=False)+'\n')
  cropfile=OUT/'CROP_PACKETS230_INPUT_ONLY.json';write(cropfile,dict(completed=True,frames=230,rows=1840,sheets=crop_manifest))
  parfile=OUT/'FIXED10_INPUT_AND_DINO_REPEAT_PARITY.json';write(parfile,dict(completed=True,PASS=True,prior_R0=p['prior_fixed10_R0'],prior_factual96=p['prior_fixed10_factual_crops'],no_prior_MR10_DINO_cache_claim=True,frames=10,rows=80,replay_encoder_calls=10,rows_compared=parity))
  write(OUT/'FEATURE_COMPLETE.json',dict(completed=True,PASS=True,code=p['code'],policy=p['policy'],protocol=bind(PROTOCOL),entry=bind(entry),inputs=p['inputs'],input_seal=bind(OUT/'INPUT_SEAL_BEFORE_ENCODER.json'),source_initializer=p['source_initializer'],source_intake=p['source_intake'],features=bind(output),inputfacts=bind(factfile),crop_manifest=bind(cropfile),fixed10_parity=bind(parfile),rows=N,frames=230,features_shape=[N,1928],available=int(available.sum()),feature_constructed=int(constructed.sum()),all1840_preserved=True,context_box_mode=BOX_MODE,primary_encoder_calls=230,fixed10_replay_calls=10,total_encoder_calls=calls,views_per_call=17,frozen_encoder_state_before=before,frozen_encoder_state_after=after,encoder_state_unchanged=True,weight_and_inputs_unchanged=True,prior_MR10_canonical_factual96_and_new_features_replay_bitexact=True,register_features_used=False,GT_pose_K_mask_depth_label_reference_reads=0,classifier_R0_P3_fit_private_PnP_calls=0,seconds=time.monotonic()-started))
  print(json.dumps(dict(completed=True,PASS=True,feature_complete=bind(OUT/'FEATURE_COMPLETE.json'))),flush=True)
 except BaseException:
  write(OUT/'FAILED_EXECUTION.json',dict(completed=False,traceback=traceback.format_exc(),automatic_retry=False));raise
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');ap.add_argument('--entry',type=Path);a=ap.parse_args()
 if a.prepare and not a.entry:prepare()
 elif a.entry and not a.prepare:run(a.entry)
 else:raise SystemExit('Use --prepare after sealed initializer; actual root --entry separately')
