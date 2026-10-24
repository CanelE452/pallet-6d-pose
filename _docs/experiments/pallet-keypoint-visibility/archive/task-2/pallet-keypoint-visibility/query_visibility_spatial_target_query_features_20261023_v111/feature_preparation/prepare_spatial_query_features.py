"""Input-only frozen DINO producer. No model/data execution on import."""
import os,sys
os.environ['XFORMERS_DISABLED']='1';os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8');sys.dont_write_bytecode=True
from pathlib import Path
from collections import OrderedDict
import argparse,ast,hashlib,importlib.util,json,time,traceback
import numpy as np
import torch
from torch.nn import functional as F
H=Path(__file__).resolve().parent;V=H.parent;P=V.parent
DESIGN=V/'ROOT_C11_DESIGN_BEFORE_SOURCE_FORWARD.json';DESIGN_SHA='f60d3d84ecc0fd3305864aae7e3f834969e1170e857f71541f8f900c06382daa'
C8=P/'query_visibility_frozen_rgb_head_20261023_v105/frozen_rgb_feature_preparation/prepare_frozen_rgb_features.py'
C9=P/'query_visibility_target_relative_rgb_head_20261023_v108/context_feature_preparation/prepare_target_context_features.py'
LOCAL=P/'query_visibility_public_cue_inputs_20261023_v90/producer/spatial_input_math.py'
TARGET=P/'query_visibility_target_context_rgb_20261023_v107/target_context_math.py'
OLD=P/'query_visibility_query_anchored_cue_20261023_v96/actual_source_pair_input_pack/PACK_MANIFEST.json'
SYN=P/'query_visibility_diverse_expected_cue_source_20261023_v99/matching_feature_pack/PACK_MANIFEST.json'
PROTOCOL=H/'FEATURE_PROTOCOL_BEFORE_SOURCE_FORWARD.json';OUT=V/'spatial_query_features'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for x in iter(lambda:f.read(1<<20),b''):h.update(x)
 return h.hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def check(b):assert sha(b['path'])==b['sha256'],b['path'];return Path(b['path'])
def read(p):return json.loads(Path(p).read_text())
def rows(p):return[json.loads(x)for x in Path(p).read_text().splitlines()if x]
def write(p,r):
 with Path(p).open('x')as f:json.dump(r,f,indent=2,allow_nan=False);f.write('\n')
def mod(p,n):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def pure_crop():
 tree=ast.parse(LOCAL.read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='crop');ns={'np':np};exec(compile(ast.Module(body=[fn],type_ignores=[]),str(LOCAL)+'::crop_only','exec'),ns);return ns['crop']

def sample_Q_raw(patch,queries,boxes,support):
 """Final grid is F32(2*uF64-1). Closed box, border, alignFalse."""
 n=len(patch)
 if patch.shape!=(n,256,384)or patch.dtype!=torch.float32:raise ValueError('Raw FP32 patchtokens[N,256,384] required')
 q=np.asarray(queries,np.float64);b=np.asarray(boxes,np.float64);support=np.asarray(support,bool)
 if q.shape!=(n,2)or b.shape!=(n,4)or support.shape!=(n,):raise ValueError('Input geometry shape')
 finite=np.isfinite(q).all(1)&np.isfinite(b).all(1)&~(q==[-1,-1]).all(1);positive=(b[:,2:]>b[:,:2]).all(1);good=finite&positive&support
 u=np.zeros((n,2),np.float64);u[good]=(q[good]-b[good,:2])/(b[good,2:]-b[good,:2]);inside=good&(u>=0).all(1)&(u<=1).all(1)
 grid=torch.as_tensor((2*u-1).astype(np.float32),device=patch.device).view(n,1,1,2)
 p=patch.reshape(n,16,16,384).permute(0,3,1,2)
 value=F.grid_sample(p,grid,mode='bilinear',padding_mode='border',align_corners=False)[:,:,0,0]
 value=torch.where(torch.as_tensor(inside,device=patch.device)[:,None],value,torch.zeros_like(value))
 return value,inside

def assemble(ctxcls,ctxpatch,fullcls,fullpatch,centralcls,queries,boxes,fractions,central4):
 n=len(ctxcls)
 if any(t.dtype!=torch.float32 for t in(ctxcls,ctxpatch,fullcls,fullpatch,centralcls)):raise ValueError('FP32 descriptors only')
 if ctxcls.shape!=fullcls.shape or ctxcls.shape!=centralcls.shape or ctxcls.shape!=(n,384)or fullpatch.shape!=(n,256,384):raise ValueError('DINO output shapes')
 frac=np.asarray(fractions,np.float32);c4=np.asarray(central4,bool)
 if frac.shape!=(n,3)or c4.shape!=(n,)or not np.isfinite(frac).all()or not((frac>=0)&(frac<=1)).all():raise ValueError('Support shapes/ranges')
 rawq,inside=sample_Q_raw(ctxpatch,queries,boxes,frac[:,0]>0)
 rawp=fullpatch.reshape(n,16,16,384)[:,7:9,7:9].mean((1,2))
 normalize=lambda x,g:torch.where(torch.as_tensor(g,device=x.device)[:,None],F.normalize(x.float(),p=2,dim=1,eps=1e-12),torch.zeros_like(x))
 T=normalize(ctxcls,frac[:,0]>0);Q=normalize(rawq,inside);Fv=normalize(fullcls,frac[:,1]>0);C=normalize(centralcls,frac[:,2]>0);Pv=normalize(rawp,c4)
 cosine=lambda a,b:F.cosine_similarity(a,b,dim=1,eps=1e-12)
 scalars=torch.stack((cosine(T,C),cosine(T,Fv),cosine(Q,C),cosine(Q,Pv),torch.as_tensor(frac[:,0],device=T.device),torch.as_tensor(frac[:,1],device=T.device),torch.as_tensor(frac[:,2],device=T.device),torch.as_tensor(inside,dtype=torch.float32,device=T.device)),1)
 out=torch.cat((T,Q,Fv,C,Pv,scalars),1)
 assert out.shape==(n,1928)and torch.isfinite(out).all()
 return out,inside

def state_sha(model):
 h=hashlib.sha256()
 for k,t in sorted(model.state_dict().items()):
  a=t.detach().cpu().contiguous();h.update(k.encode());h.update(str(a.dtype).encode());h.update(str(tuple(a.shape)).encode());h.update(a.numpy().tobytes())
 return h.hexdigest()

def prepare():
 assert sha(DESIGN)==DESIGN_SHA
 d=read(DESIGN);runtime=read(check(d['encoder']['runtime_receipt']));plan=read(check(runtime['plan']));assert runtime['completed']and runtime['PASS']
 g=H/'GENERATED_FEATURE_TESTS.json';assert read(g)['PASS']and read(g)['code']==bind(__file__)
 om=read(OLD);sm=read(SYN);units=rows(check(om['input_units']));base=rows(check(om['base_input_lineage']));syn=rows(check(sm['input_lineage']));refs=[]
 assert len(units)==1348 and len(base)==960 and len(syn)==2304
 for i,r in enumerate(units):
  assert r['input_unit_index']==i;ir=r['input_ref']
  if ir['kind']=='BASE_ROW':rr=base[ir['slot_index']]['before_input_refs'];rgb=rr['rgb96'];valid=rr['crop_valid96']
  else:
   assert ir['kind']=='CHANGED_CHUNK';rgb=dict(file=ir['file'],key='rgb96',index=ir['index']);valid=dict(file=ir['file'],key='crop_valid96',index=ir['index'])
  refs.append(dict(input_unit_index=i,rgb96_ref=rgb,valid96_ref=valid,rgb96_sha256=r['tensor_value_sha256']['rgb96'],valid96_sha256=r['tensor_value_sha256']['crop_valid96']))
 for i,r in enumerate(syn):
  assert r['input_unit_index']==i;chunk=sm['chunks'][r['chunk_index']]['file'];j=r['index_in_chunk'];refs.append(dict(input_unit_index=1348+i,rgb96_ref=dict(file=chunk,key='rgb96',index=j),valid96_ref=dict(file=chunk,key='crop_valid96',index=j),rgb96_sha256=r['rgb96_sha256'],valid96_sha256=r['crop_valid96_sha256']))
 rf=H/'OLD3652_RGB_UNIT_REFS_INPUT_ONLY.jsonl'
 with rf.open('x')as f:
  for r in refs:f.write(json.dumps(r,separators=(',',':'))+'\n')
 contexts=rows(check(d['source']['old_unit_contexts']));assert len(contexts)==3652 and all(r['input_unit_index']==i for i,r in enumerate(contexts))
 c10r=read(check(d['source']['C10_feature_receipt']));assert c10r['completed']and c10r['PASS'];new=rows(check(d['source']['new_queries']));assert len(new)==1000
 for r in new:assert len(r['initial_points'])==8 and r['split']=='TRAIN'and r['source_emulation']is True and r['native_mapping']=='UNKNOWN'
 p=dict(completed=True,design=bind(DESIGN),code=bind(__file__),generated=bind(g),asset_runtime=d['encoder']['runtime_receipt'],asset_plan=runtime['plan'],weight=d['encoder']['weight'],official_repo=plan['local_official_repo'],official_revision=plan['local_git_revision'],official_source_bindings=plan['source_bindings'],C8_preprocess=bind(C8),C9_target_preprocess=bind(C9),local_crop_code=bind(LOCAL),target_crop_code=bind(TARGET),old_manifest=bind(OLD),synthetic_manifest=bind(SYN),old_RGB_refs=bind(rf),old_contexts=d['source']['old_unit_contexts'],new_inputs=d['source']['new_queries'],C10_receipt=d['source']['C10_feature_receipt'],C10_metadata_only_pack=c10r['output'],new_query_lineage=c10r['new_query_lineage'],new_RGB_query_facts=c10r['new_RGB_query_facts'],output=str(OUT),schema=dict(features=[11652,1928],occurrences=41984,layout=d['features'],batch_old_units=8,batch_old_views=24,batch_new_context_rows=1,batch_new_views=17,max_views=32),runtime=runtime['runtime'],preparation_model_load_pixel_or_feature_read_sourceforward_fit_current=0)
 write(PROTOCOL,p);write(H/'ROOT_FEATURE_ENTRY_TEMPLATE.json',dict(approved_actual_source_feature_extraction=False,source_only=True,current_allowed=False,fit_allowed=False,PnP_allowed=False,code=bind(__file__),protocol=bind(PROTOCOL),design=bind(DESIGN),generated=bind(g),output=str(OUT)))
 print(json.dumps(dict(code=bind(__file__),protocol=bind(PROTOCOL),template=bind(H/'ROOT_FEATURE_ENTRY_TEMPLATE.json'),command=f'{sys.executable} {Path(__file__).resolve()} --entry ROOT_APPROVED_ENTRY.json')),flush=True)

class InputCache:
 def __init__(self):self.arrays={};self.checked=set();self.images=OrderedDict()
 def verify(self,b):
  if b['path']not in self.checked:check(b);self.checked.add(b['path'])
  return Path(b['path'])
 def array(self,r):
  p=self.verify(r['file']);key=(str(p),r['key'])
  if key not in self.arrays:
   with np.load(p,allow_pickle=False)as f:self.arrays[key]=np.array(f[r['key']],copy=True)
  return np.array(self.arrays[key][r['index']],copy=True)
 def image(self,b,fmt,hw,expected=None):
  path=self.verify(b)
  if str(path)in self.images:rgb=self.images.pop(str(path));self.images[str(path)]=rgb
  else:
   if fmt=='CANONICAL_RGB_UINT8_HWC_NPY':rgb=np.load(path,allow_pickle=False)
   else:
    from PIL import Image
    rgb=np.array(Image.open(path).convert('RGB'),dtype=np.uint8,copy=True)
   self.images[str(path)]=rgb
   while len(self.images)>4:self.images.popitem(last=False)
  assert rgb.dtype==np.uint8 and list(rgb.shape)==hw+[3]
  decoded=hashlib.sha256(rgb.tobytes()).hexdigest()
  if expected is not None:assert decoded==expected
  return rgb,decoded

def run(entry):
 e=read(entry);assert e['approved_actual_source_feature_extraction']is True and e['source_only']and e['current_allowed']is False and e['fit_allowed']is False and e['PnP_allowed']is False
 for k in('code','protocol','design','generated'):check(e[k])
 assert e['code']==bind(__file__)and e['protocol']==bind(PROTOCOL)and e['design']==bind(DESIGN)and sha(DESIGN)==DESIGN_SHA
 p=read(PROTOCOL);assert p['code']==e['code']and p['generated']==e['generated']and read(check(e['generated']))['PASS']and not OUT.exists()and Path(e['output']).resolve()==OUT.resolve()
 for k in('asset_runtime','asset_plan','weight','C8_preprocess','C9_target_preprocess','local_crop_code','target_crop_code','old_manifest','synthetic_manifest','old_RGB_refs','old_contexts','new_inputs','C10_receipt','C10_metadata_only_pack','new_query_lineage','new_RGB_query_facts'):check(p[k])
 for b in p['official_source_bindings']:check(b)
 assert torch.__version__==p['runtime']['torch']=='2.1.1+cu118'and torch.cuda.is_available()and Path(sys.executable).resolve()==Path(p['runtime']['executable']).resolve()
 with np.load(p['C10_metadata_only_pack']['path'],allow_pickle=False)as f:meta={k:np.array(f[k],copy=True)for k in('occurrence_unit','query_available','factual_query_available','context_available','global_occurrence_index')}
 assert meta['occurrence_unit'].shape==(41984,)and np.array_equal(meta['occurrence_unit'][33984:],np.arange(3652,11652))and np.array_equal(meta['query_available'],meta['factual_query_available']&meta['context_available'])
 refs=rows(p['old_RGB_refs']['path']);ctx=rows(p['old_contexts']['path']);new=rows(p['new_inputs']['path']);newfacts=rows(p['new_RGB_query_facts']['path']);assert len(refs)==len(ctx)==3652 and len(new)==len(newfacts)==1000
 crop=pure_crop();tm=mod(p['target_crop_code']['path'],'c11_target_math');c8=mod(p['C8_preprocess']['path'],'c11_preprocess_local_only');c9=mod(p['C9_target_preprocess']['path'],'c11_preprocess_target_only')
 OUT.mkdir(exist_ok=False);write(OUT/'STARTED.json',dict(entry=bind(entry),no_automatic_restart=True));started=time.monotonic()
 try:
  torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
  sys.path.insert(0,p['official_repo']);from dinov2.hub.backbones import dinov2_vits14_reg
  model=dinov2_vits14_reg(pretrained=False);state=torch.load(p['weight']['path'],map_location='cpu',weights_only=True);loaded=model.load_state_dict(state,strict=True);assert not loaded.missing_keys and not loaded.unexpected_keys and tuple(state['register_tokens'].shape)==(1,4,384);del state
  model=model.cuda().float().eval().requires_grad_(False);assert sum(x.numel()for x in model.parameters())==22058112;before=state_sha(model);assert before==read(p['asset_runtime']['path'])['state_before_SHA256']
  torch.cuda.reset_peak_memory_stats();cache=InputCache();features=[];insidebits=[];cert=[];calls=0;repeat=False
  def encode(views):
   nonlocal calls,repeat
   assert views.dtype==torch.float32 and len(views)<=32
   with torch.inference_mode():
    out=model.forward_features(views);calls+=1
    if not repeat:
     again=model.forward_features(views);calls+=1
     for k in('x_norm_clstoken','x_norm_patchtokens'):assert torch.equal(out[k],again[k])
     repeat=True
   assert out['x_norm_clstoken'].shape==(len(views),384)and out['x_norm_patchtokens'].shape==(len(views),256,384)
   return out
  for start in range(0,3652,8):
   rs=ctx[start:start+8];N=len(rs);lr=[];lv=[];tr=[];tv=[];qs=[];boxes=[];fra=[];c4=[]
   for i,r in enumerate(rs):
    u=start+i;rf=refs[u];a=cache.array(rf['rgb96_ref']);v=cache.array(rf['valid96_ref']);assert a.dtype==np.uint8 and a.shape==(3,96,96)and v.dtype==bool and v.shape==(1,96,96)
    assert hashlib.sha256(a.tobytes()).hexdigest()==rf['rgb96_sha256']and hashlib.sha256(v.tobytes()).hexdigest()==rf['valid96_sha256']
    if r['box_xyxy']is None:t=np.zeros((224,224,3),np.float32);vt=np.zeros((224,224),bool)
    else:rgb,_=cache.image(r['full_RGB'],r['full_RGB_format'],r['raw_hw'],r['decoded_RGB_sha256']);t,vt=tm.sample_target_context(rgb,r['box_xyxy'])
    lr.append(a);lv.append(v);tr.append(t.transpose(2,0,1));tv.append(vt[None]);qs.append(r['query_xy']if r['query_xy']is not None else[np.nan,np.nan]);boxes.append(r['box_xyxy']if r['box_xyxy']is not None else[np.nan]*4);fra.append([np.float32(vt.mean()),np.float32(v.mean()),np.float32(v[:,32:64,32:64].mean())]);c4.append(bool(v[:,47:49,47:49].all()))
   local,_=c8.preprocess_views(torch.as_tensor(np.stack(lr),device='cuda'),torch.as_tensor(np.stack(lv),device='cuda'));target=c9.preprocess_target(torch.as_tensor(np.stack(tr),device='cuda'),torch.as_tensor(np.stack(tv),device='cuda'));raw=encode(torch.cat((target,local),0));cls=raw['x_norm_clstoken'];patch=raw['x_norm_patchtokens']
   with torch.inference_mode():z,inside=assemble(cls[:N],patch[:N],cls[N:2*N],patch[N:2*N],cls[2*N:],qs,boxes,fra,c4)
   features.append(z.cpu().numpy());insidebits.extend(inside.tolist())
   for i in range(N):cert.append(dict(input_unit_index=start+i,RGB96_sha256=refs[start+i]['rgb96_sha256'],valid96_sha256=refs[start+i]['valid96_sha256'],target_RGB224_float32_sha256=hashlib.sha256(np.ascontiguousarray(tr[i]).tobytes()).hexdigest(),Q_inside=bool(inside[i])))
   if(start//8+1)%100==0:print(json.dumps(dict(old_units=start+N,total=3652,seconds=time.monotonic()-started)),flush=True)
  for i,(r,fc)in enumerate(zip(new,newfacts)):
   assert fc['input_id']==r['input_id'];rgb,decoded=cache.image(r['image'],'PNG_RGB',r['raw_hw'],fc['decoded_RGB_sha256']);t,vt=tm.sample_target_context(rgb,r['query_box_xyxy']);lr=[];lv=[];fra=[];c4=[]
   assert hashlib.sha256(np.ascontiguousarray(t).tobytes()).hexdigest()==fc['target_RGB224_float32_sha256']
   for k,q in enumerate(r['initial_points']):
    a,v=crop(rgb,q,r['roi_side'],96);f=fc['queries'][k];assert hashlib.sha256(a.tobytes()).hexdigest()==f['RGB96_sha256']and hashlib.sha256(v.tobytes()).hexdigest()==f['valid96_sha256'];lr.append(a);lv.append(v);fra.append([np.float32(vt.mean()),np.float32(v.mean()),np.float32(v[:,32:64,32:64].mean())]);c4.append(bool(v[:,47:49,47:49].all()))
   local,_=c8.preprocess_views(torch.as_tensor(np.stack(lr),device='cuda'),torch.as_tensor(np.stack(lv),device='cuda'));target=c9.preprocess_target(torch.as_tensor(t.transpose(2,0,1)[None],device='cuda'),torch.as_tensor(vt[None,None],device='cuda'));raw=encode(torch.cat((target,local),0));cls=raw['x_norm_clstoken'];patch=raw['x_norm_patchtokens']
   with torch.inference_mode():z,inside=assemble(cls[:1].repeat_interleave(8,0),patch[:1].repeat_interleave(8,0),cls[1:9],patch[1:9],cls[9:],r['initial_points'],[r['query_box_xyxy']]*8,fra,c4)
   features.append(z.cpu().numpy());insidebits.extend(inside.tolist())
   for k in range(8):cert.append(dict(input_unit_index=3652+i*8+k,input_id=r['input_id'],query_index=k,RGB96_sha256=fc['queries'][k]['RGB96_sha256'],valid96_sha256=fc['queries'][k]['valid96_sha256'],decoded_RGB_sha256=decoded,Q_inside=bool(inside[k])))
   if(i+1)%100==0:print(json.dumps(dict(new_input_rows=i+1,total=1000,seconds=time.monotonic()-started)),flush=True)
  x=np.concatenate(features).astype(np.float32);insidebits=np.asarray(insidebits,bool);assert x.shape==(11652,1928)and np.isfinite(x).all()and np.array_equal(x[:,1927],insidebits.astype(np.float32))and not x[~insidebits,384:768].any()
  after=state_sha(model);assert before==after and repeat and all(not m.training for m in model.modules())and all(not p.requires_grad for p in model.parameters())and sha(p['weight']['path'])==p['weight']['sha256']
  output=OUT/'SOURCE_SPATIAL_QUERY_FEATURES.npz';np.savez_compressed(output,features=x,unit_Q_inside=insidebits,**meta)
  cf=OUT/'INPUT_RGB_AND_SPATIAL_FACTS_INPUT_ONLY.jsonl'
  with cf.open('x')as f:
   for r in cert:f.write(json.dumps(r,separators=(',',':'))+'\n')
  write(OUT/'FEATURE_COMPLETE.json',dict(completed=True,PASS=True,output=bind(output),protocol=e['protocol'],entry=bind(entry),code=e['code'],design=e['design'],new_query_lineage=p['new_query_lineage'],input_RGB_spatial_facts=bind(cf),C10_metadata_receipt=p['C10_receipt'],C10_feature_values_read=False,features_shape=[11652,1928],occurrences=41984,all_C10_mapping_availability_exact=True,Q_outside_never_adds_U=True,Q_inside_units=int(insidebits.sum()),Q_outside_units=int((~insidebits).sum()),first_batch_repeat_exact=True,model_state_unchanged=True,frozen_encoder_state_before=before,frozen_encoder_state_after=after,encoder_batch_calls=calls,max_views=24,register_features_used=False,seconds=time.monotonic()-started,peak_GPU_bytes=torch.cuda.max_memory_allocated(),labels_masks_depth_pose_reference_nativeGT_read=0,new_head_fit_private_current_PnP=0))
  print(json.dumps(dict(completed=True,receipt=bind(OUT/'FEATURE_COMPLETE.json'))),flush=True)
 except BaseException:
  write(OUT/'FAILED_EXECUTION.json',dict(completed=False,traceback=traceback.format_exc(),no_automatic_restart=True));raise
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');ap.add_argument('--entry',type=Path);a=ap.parse_args()
 if a.prepare:prepare()
 elif a.entry:run(a.entry)
 else:raise SystemExit('Use --prepare or --entry ROOT_APPROVED_ENTRY')
