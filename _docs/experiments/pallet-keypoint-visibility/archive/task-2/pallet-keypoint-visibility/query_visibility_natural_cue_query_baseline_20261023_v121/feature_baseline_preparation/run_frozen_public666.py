"""Frozen C14 PUBLIC666 source baseline. No fit/current/initializer/GT forward."""
import os,sys
os.environ['XFORMERS_DISABLED']='1';os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8');sys.dont_write_bytecode=True
from pathlib import Path
import argparse,ast,csv,hashlib,importlib.util,json,time,traceback
import numpy as np
import torch
H=Path(__file__).resolve().parent;V=H.parent;P=V.parent
POLICY=V/'ROOT_PUBLIC_BASELINE_POLICY_BEFORE_FORWARD.json';PROTOCOL=H/'BASELINE_PROTOCOL_BEFORE_FORWARD.json';OUT=V/'frozen_C14_public_baseline'
C11=P/'query_visibility_spatial_target_query_features_20261023_v111/feature_preparation/prepare_spatial_query_features.py';C11_SHA='cbf2f1cad553673b727c39fbe5c9c4263dfbdac35dba0aeea086f956e74ff19e'
C11_PROTOCOL=C11.parent/'FEATURE_PROTOCOL_BEFORE_SOURCE_FORWARD.json'
N=666;BATCH=8
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for z in iter(lambda:f.read(1<<20),b''):h.update(z)
 return h.hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def check(b):assert sha(b['path'])==b['sha256'],b['path'];return Path(b['path'])
def read(p):return json.loads(Path(p).read_text())
def rows(p):return[json.loads(l)for l in Path(p).read_text().splitlines()if l]
def write(p,x):
 with Path(p).open('x')as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def writecsv(p,rs):
 with Path(p).open('x',newline='')as f:
  w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
def mod(p,n):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def ash(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def validate_rows(rs):
 assert len(rs)==N and [r['occurrence_index']for r in rs]==list(range(N))and len({r['query_id']for r in rs})==N
 assert sum(r['split']=='TRAIN'for r in rs)==270 and sum(r['split']=='VALIDATION'for r in rs)==396
 for i,r in enumerate(rs):
  assert r['pair_index']==i//18 and r['jitter_index']==(i%18)//2 and r['view']==('before'if i%2==0 else'after')
  assert r['offset_xy']==[(-3,0,3)[r['jitter_index']%3],(-3,0,3)[r['jitter_index']//3]]
  assert r['source_emulated_not_initializer']and r['fixed_carrier_slot']==0 and not r['carrier_slot_is_native_part_identity']
  q=np.asarray(r['initial_points'],np.float64);assert q.shape==(8,2)and np.isfinite(q).all()and np.array_equal(q[0],r['query_xy'])
  assert r['L']==float(np.clip(.25*np.max(np.ptp(q,axis=0)),16,max(r['raw_hw'])))
  good=~(q==[-1,-1]).all(1);assert r['context_box_xyxy']==np.r_[q[good].min(0),q[good].max(0)].tolist()
  assert not any(k in r for k in('original_reference_xy_unchanged','original_teacher_label_unchanged','teacher_label','reference_xy','uncertainty_bbox','mask','depth','pose'))
  if i%2:
   for k in('query_xy','L','initial_points','context_box_xyxy','offset_xy','split'):assert r[k]==rs[i-1][k]
 return True
def pure_query_validity(path):
 fn=next(n for n in ast.parse(Path(path).read_text()).body if isinstance(n,ast.FunctionDef)and n.name=='query_validity');ns={'np':np};exec(compile(ast.Module(body=[fn],type_ignores=[]),str(path)+'::query_validity_only','exec'),ns);return ns['query_validity']

def authorize(entry):
 e=read(entry);assert e['approved_frozen_PUBLIC_source_baseline']is True and all(e[k]is False for k in('fit_allowed','current_allowed','V112_allowed','initializer_allowed','PnP_allowed'))
 for k in('code','protocol','policy','input_manifest'):check(e[k])
 p=read(e['protocol']['path']);assert e['code']==bind(__file__)and p['code']==e['code']and e['protocol']==bind(PROTOCOL)and e['policy']==p['policy']==bind(POLICY)and e['input_manifest']==p['input_manifest']
 assert p['C11_producer']==bind(C11)and sha(C11)==C11_SHA
 policy=read(POLICY);assert policy['threshold_logit']==0 and p['source_model_freeze']==policy['model_freeze']and p['checkpoint']==policy['checkpoint']
 for k in('C11_producer','C11_protocol','asset_runtime','asset_plan','weight','C8_preprocess','C9_target_preprocess','local_crop_code','target_crop_code','source_model_freeze','checkpoint','head_class_code'):check(p[k])
 for b in p['official_source_bindings']:check(b)
 freeze=read(p['source_model_freeze']['path']);assert freeze['completed']and freeze['selected_update']==1200 and freeze['threshold_logit']==0 and freeze['checkpoint']==p['checkpoint']and freeze['head_class_code']==p['head_class_code']and not freeze['current_used_for_fit_or_selection']
 rs=rows(e['input_manifest']['path']);validate_rows(rs)
 # No weak evidence body/hash is read by this stage.
 return e,p,rs

def run(entry):
 e,p,rs=authorize(entry);assert not OUT.exists(),'One invocation; preserve failures, no automatic restart'
 assert torch.__version__==p['runtime']['torch']=='2.1.1+cu118'and torch.cuda.is_available()and Path(sys.executable).resolve()==Path(p['runtime']['executable']).resolve()
 producer=mod(p['C11_producer']['path'],'v121_C11_operator');c8=mod(p['C8_preprocess']['path'],'v121_C8_preprocess');c9=mod(p['C9_target_preprocess']['path'],'v121_C9_preprocess');tm=mod(p['target_crop_code']['path'],'v121_target_math');crop=producer.pure_crop();query_valid=pure_query_validity(p['local_crop_code']['path'])
 OUT.mkdir();write(OUT/'STARTED.json',dict(entry=bind(entry),source_only=True,weak_evidence_body_or_hash_read=False,no_automatic_restart=True));started=time.monotonic()
 try:
  torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
  sys.path.insert(0,p['official_repo']);from dinov2.hub.backbones import dinov2_vits14_reg
  encoder=dinov2_vits14_reg(pretrained=False);state=torch.load(p['weight']['path'],map_location='cpu',weights_only=True);encoder.load_state_dict(state,strict=True);del state;encoder=encoder.cuda().float().eval().requires_grad_(False)
  encstate=producer.state_sha(encoder);assert encstate==read(p['asset_runtime']['path'])['state_before_SHA256'];torch.cuda.reset_peak_memory_stats()
  imagecache={};features=[];rgbrows=[];validrows=[];facts=[];calls=0;repeat=False;factual=[];contexts=[];qinside=[]
  for start in range(0,N,BATCH):
   batch=rs[start:start+BATCH];n=len(batch);lr=[];lv=[];tr=[];tv=[];fra=[];c4=[]
   for r in batch:
    rb=r['full_RGB'];key=(rb['path'],rb['sha256'])
    if key not in imagecache:
     from PIL import Image
     check(rb);rgb=np.array(Image.open(rb['path']).convert('RGB'),dtype=np.uint8,copy=True);assert list(rgb.shape[:2])==r['raw_hw']and ash(rgb)==r['decoded_RGB_sha256'];imagecache[key]=rgb
    rgb=imagecache[key];a,v=crop(rgb,r['query_xy'],r['L'],96);t,vt=tm.sample_target_context(rgb,r['context_box_xyxy']);qvalid,continuous,nearest=query_valid(np.asarray(r['query_xy'],np.float64)[None],r['raw_hw']);central4=bool(v[:,47:49,47:49].all());fact=bool(qvalid[0]and central4);ctx=bool(vt.any())
    assert bool(qvalid[0])==r['q_native_valid']and bool(continuous[0])==r['original_query_continuous_valid']and bool(nearest[0])==r['query_nearest_pixel_valid']and central4==r['local_central4_available']and fact==r['factual_query_available']and ctx==r['context_available_geometry']and(fact and ctx)==r['query_available_geometry']
    fractions=[np.float32(vt.mean()),np.float32(v.mean()),np.float32(v[:,32:64,32:64].mean())]
    for x,k in zip(fractions,('target_valid_fraction_geometry','local_valid_fraction_geometry','central_valid_fraction_geometry')):assert x==np.float32(r[k])
    lr.append(a);lv.append(v);tr.append(t.transpose(2,0,1));tv.append(vt[None]);fra.append(fractions);c4.append(central4);factual.append(fact);contexts.append(ctx)
    facts.append(dict(occurrence_index=r['occurrence_index'],query_id=r['query_id'],full_RGB=rb,decoded_RGB_sha256=ash(rgb),raw_hw=r['raw_hw'],query_xy=r['query_xy'],L=r['L'],context_box_xyxy=r['context_box_xyxy'],RGB96_sha256=ash(a),valid96_sha256=ash(v),central32_sha256=ash(a[:,32:64,32:64]),central_valid32_sha256=ash(v[:,32:64,32:64]),target_RGB224_F32_sha256=ash(t),target_valid_sha256=ash(vt),target_fraction=float(fractions[0]),local_fraction=float(fractions[1]),central_fraction=float(fractions[2]),q_native_valid=bool(qvalid[0]),local_central4_available=central4,factual_query_available=fact,context_available=ctx))
   local,_=c8.preprocess_views(torch.as_tensor(np.stack(lr),device='cuda'),torch.as_tensor(np.stack(lv),device='cuda'));target=c9.preprocess_target(torch.as_tensor(np.stack(tr),device='cuda'),torch.as_tensor(np.stack(tv),device='cuda'));views=torch.cat((target,local),0);assert views.shape==(3*n,3,224,224)and len(views)<=24
   with torch.inference_mode():
    z=encoder.forward_features(views);calls+=1
    if not repeat:
     z2=encoder.forward_features(views);calls+=1;assert all(torch.equal(z[k],z2[k])for k in('x_norm_clstoken','x_norm_patchtokens'));repeat=True
    cl=z['x_norm_clstoken'];pt=z['x_norm_patchtokens'];xx,inside=producer.assemble(cl[:n],pt[:n],cl[n:2*n],pt[n:2*n],cl[2*n:], [r['query_xy']for r in batch],[r['context_box_xyxy']for r in batch],fra,c4)
   for r,q in zip(batch,inside):assert bool(q)==r['context_Q_inside_geometry']
   features.append(xx.cpu().numpy());qinside.extend(inside.tolist());rgbrows.extend(lr);validrows.extend(lv)
   if(start//BATCH+1)%20==0:print(json.dumps(dict(input_rows=start+n,total=N,encoder_calls=calls,seconds=time.monotonic()-started)),flush=True)
  x=np.concatenate(features);rgbrows=np.stack(rgbrows);validrows=np.stack(validrows);factual=np.asarray(factual,bool);contexts=np.asarray(contexts,bool);available=factual&contexts;qinside=np.asarray(qinside,bool)
  assert x.shape==(N,1928)and x.dtype==np.float32 and np.isfinite(x).all()and np.array_equal(x[:,1927],qinside.astype(np.float32))and not x[~qinside,384:768].any()and calls==85 and len(imagecache)==44
  assert producer.state_sha(encoder)==encstate and all(not z.training for z in encoder.modules())and all(not z.requires_grad for z in encoder.parameters());del encoder
  ff=OUT/'SOURCE_FEATURES666_INPUT_ONLY.npz';np.savez_compressed(ff,features=x,occurrence_unit=np.arange(N,dtype=np.int32),query_available=available,factual_query_available=factual,context_available=contexts,unit_Q_inside=qinside,global_occurrence_index=np.arange(N,dtype=np.int32))
  cf=OUT/'INPUT_CROPS666_INPUT_ONLY.npz';np.savez_compressed(cf,rgb96=rgbrows,valid96=validrows,central32=rgbrows[:,:,32:64,32:64],central_valid32=validrows[:,:,32:64,32:64],global_occurrence_index=np.arange(N,dtype=np.int32));assert rgbrows.nbytes==18413568
  factfile=OUT/'INPUT_RGB_AND_SPATIAL_FACTS666_INPUT_ONLY.jsonl'
  with factfile.open('x')as f:
   for i,r in enumerate(facts):r['context_Q_inside']=bool(qinside[i]);f.write(json.dumps(r,separators=(',',':'))+'\n')
  write(OUT/'FEATURE_COMPLETE.json',dict(completed=True,PASS=True,code=e['code'],protocol=e['protocol'],entry=bind(entry),input_manifest=e['input_manifest'],features=bind(ff),crops=bind(cf),inputfacts=bind(factfile),features_shape=[N,1928],RGB96_shape=list(rgbrows.shape),central32_shape=[N,3,32,32],encoder_calls=calls,primary_calls=84,repeat_calls=1,encoder_state_before=encstate,encoder_state_after=encstate,model_state_unchanged=True,first_batch_repeat_exact=repeat,weak_evidence_body_or_hash_read=False,new_initializer_fit_current_V112_PnP=0))
  headmod=mod(p['head_class_code']['path'],'v121_exact_C11_head');payload=torch.load(p['checkpoint']['path'],map_location='cpu',weights_only=True);assert payload['update']==1200 and payload['head_class_code']==p['head_class_code'];head=headmod.SpatialTargetQueryHead().cuda().float().eval().requires_grad_(False);head.load_state_dict(payload['state_dict'],strict=True);del payload;assert sum(z.numel()for z in head.parameters())==221825;hs=producer.state_sha(head)
  with torch.inference_mode():raw=head(torch.as_tensor(x,device='cuda'));again=head(torch.as_tensor(x,device='cuda'));assert torch.equal(raw,again)
  raw=raw.cpu().numpy().astype(np.float32);assert raw.shape==(N,)and np.isfinite(raw).all()and producer.state_sha(head)==hs;logit=np.where(available,raw,0).astype(np.float32);pred=np.where(available,(logit>=0).astype(np.int8),-1).astype(np.int8)
  pf=OUT/'RAW_QUERY_PREDICTIONS666.npz';np.savez_compressed(pf,query_logit=logit,raw_head_logit_before_U=raw,prediction=pred,query_available=available,global_occurrence_index=np.arange(N,dtype=np.int32),query_id=np.array([r['query_id']for r in rs]))
  score=[]
  for i,r in enumerate(rs):score.append(dict(occurrence_index=i,query_id=r['query_id'],pair_index=r['pair_index'],pair_id=r['pair_id'],jitter_index=r['jitter_index'],dx=r['offset_xy'][0],dy=r['offset_xy'][1],view=r['view'],raw_split=r['raw_split'],split=r['split'],query_x=r['query_xy'][0],query_y=r['query_xy'][1],L=r['L'],query_available=bool(available[i]),raw_query_aux_hidden_logit=float(logit[i]),prediction=int(pred[i])))
  sc=OUT/'QUERY_PREDICTIONS666_INPUT_ONLY.csv';writecsv(sc,score)
  write(OUT/'PREDICTION_SEAL.json',dict(completed=True,PASS=True,code=e['code'],protocol=e['protocol'],entry=bind(entry),source_model_freeze=p['source_model_freeze'],checkpoint=p['checkpoint'],input_manifest=e['input_manifest'],feature_receipt=bind(OUT/'FEATURE_COMPLETE.json'),features=bind(ff),crops=bind(cf),inputfacts=bind(factfile),predictions=bind(pf),scoreCSV=bind(sc),rows=N,available=int(available.sum()),factual_U=int((~factual).sum()),context_U=int((~contexts).sum()),final_U=int((~available).sum()),head_calls=2,head_repeat_exact=True,head_state_before=hs,head_state_after=hs,model_state_unchanged=True,threshold_logit=0,D_rule='available AND z<0; else E if available; U first',weak_evidence_body_or_hash_read=False,seconds=time.monotonic()-started,peak_GPU_bytes=torch.cuda.max_memory_allocated(),new_initializer_fit_current_V112_PnP=0))
  print(json.dumps(dict(completed=True,prediction_seal=bind(OUT/'PREDICTION_SEAL.json'))),flush=True)
 except BaseException:
  write(OUT/'FAILED_EXECUTION.json',dict(completed=False,traceback=traceback.format_exc(),no_automatic_restart=True));raise

def metrics(y,p):
 cm=np.zeros((2,3),int)
 for t,z in zip(y,p):cm[int(t),2 if z<0 else int(z)]+=1
 div=lambda a,b:float(a/b)if b else None
 return dict(CM_D_E_rows_D_E_U_columns=cm.tolist(),D_recall=div(cm[0,0],cm[0].sum()),E_recall=div(cm[1,1],cm[1].sum()),D_precision=div(cm[0,0],cm[:,0].sum()),E_precision=div(cm[1,1],cm[:,1].sum()),U=int(cm[:,2].sum()),U_rate=div(cm[:,2].sum(),cm.sum()),count=int(cm.sum()))
def evaluate_saved(entry):
 e,p,rs=authorize(entry);assert e['allow_weak_evaluation_after_prediction_seal']is True
 seal=read(OUT/'PREDICTION_SEAL.json');assert seal['completed']and seal['PASS']and seal['input_manifest']==e['input_manifest']and seal['checkpoint']==p['checkpoint']and not seal['weak_evidence_body_or_hash_read']
 for k in('features','crops','inputfacts','predictions','scoreCSV'):check(seal[k])
 # The first weak-sidecar body/hash read occurs after the completed prediction seal.
 gt=rows(check(p['weak_evidence']));assert len(gt)==N
 with np.load(seal['predictions']['path'],allow_pickle=False)as f:z=f['query_logit'];pr=f['prediction'];a=f['query_available'];ix=f['global_occurrence_index'];ids=f['query_id']
 assert np.array_equal(ix,np.arange(N))and np.array_equal(pr,np.where(a,(z>=0).astype(np.int8),-1))
 for i,(r,g)in enumerate(zip(rs,gt)):
  for k in('occurrence_index','query_id','pair_index','pair_id','jitter_index','view','raw_split'):assert r[k]==g[k]
  assert ids[i]==r['query_id']and g['original_teacher_label_unchanged']==(0 if r['view']=='before'else 1)and not g['human_gold']and not g['new_labels_created']
 y=np.array([g['original_teacher_label_unchanged']for g in gt],np.int8);centre=np.array([r['offset_xy']==[0,0]for r in rs]);summary={}
 for fold in('ALL','TRAIN','VALIDATION'):
  keep=np.array([fold=='ALL'or r['split']==fold for r in rs])
  summary[fold]={'all9_correlated_offsets':metrics(y[keep],pr[keep]),'centre_offset0':metrics(y[keep&centre],pr[keep&centre])}
 diagnostics=[]
 for pair in range(37):
  base=pair*18;b=np.arange(base,base+18,2);af=b+1;mid=base+8
  row=dict(pair_index=pair,pair_id=rs[base]['pair_id'],raw_split=rs[base]['raw_split'],cue_id=gt[base]['cue_id'],centre_BEFORE_z=float(z[mid]),centre_AFTER_z=float(z[mid+1]),centre_BEFORE_prediction=int(pr[mid]),centre_AFTER_prediction=int(pr[mid+1]),all9_both_correct=int(((pr[b]==0)&(pr[af]==1)).sum()),centre_both_correct=bool(pr[mid]==0 and pr[mid+1]==1))
  for name,indices,c in(('BEFORE',b,mid),('AFTER',af,mid+1)):
   row.update({name+'_z_min':float(z[indices].min()),name+'_z_max':float(z[indices].max()),name+'_D_count':int((pr[indices]==0).sum()),name+'_E_count':int((pr[indices]==1).sum()),name+'_U_count':int((pr[indices]<0).sum()),name+'_flips_vs_centre':int(((pr[indices]!=pr[c])&a[indices]&a[c]).sum())})
  diagnostics.append(row)
 out=OUT/'saved_weak_evaluation';out.mkdir();writecsv(out/'PER_PAIR37_SHIFT_DIAGNOSTICS.csv',diagnostics)
 write(out/'SOURCE_RESULTS.json',dict(completed=True,source_model_freeze=p['source_model_freeze'],checkpoint=p['checkpoint'],prediction_seal=bind(OUT/'PREDICTION_SEAL.json'),input_manifest=e['input_manifest'],weak_evidence=p['weak_evidence'],rows=N,pairs37=37,physical_frame_pairs22=22,RGB44=44,metrics=summary,perpair_shift_diagnostics=bind(out/'PER_PAIR37_SHIFT_DIAGNOSTICS.csv'),all333_offset_pairs_both_correct=sum(d['all9_both_correct']for d in diagnostics),centre37_pairs_both_correct=sum(d['centre_both_correct']for d in diagnostics),limits=['Repeated development-exposed PUBLIC data/weakAI labels, not new independent data or human/nativegold.','Nine shifts retain region/cue weaklabel, not target-pixel membership; all9 are correlated, centre reported separately.','Provided reference-derived query emulation, not actual initializer correspondence or coordinate verification.','No checkpoint/threshold/normalizer/arm selection or training from results.'],model_fit_current_V112_PnP=0))
 write(out/'EVALUATION_SEAL.json',dict(completed=True,PASS=True,prediction_seal=bind(OUT/'PREDICTION_SEAL.json'),weak_evidence=p['weak_evidence'],outputs=[bind(out/'SOURCE_RESULTS.json'),bind(out/'PER_PAIR37_SHIFT_DIAGNOSTICS.csv')],new_model_forward=0));print(json.dumps(dict(completed=True,evaluation_seal=bind(out/'EVALUATION_SEAL.json'))),flush=True)

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--entry',type=Path,required=True);a.add_argument('--evaluate-saved',action='store_true');args=a.parse_args()
 if args.evaluate_saved:evaluate_saved(args.entry)
 else:run(args.entry)
