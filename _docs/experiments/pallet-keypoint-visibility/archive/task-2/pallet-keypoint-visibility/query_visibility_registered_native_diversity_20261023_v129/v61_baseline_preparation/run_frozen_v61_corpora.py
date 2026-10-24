"""Source-only frozen v61 on dynamic native/public chunked corpora.

No model/payload I/O on import or preparation. Root runs --entry. The head sees
only exact V90 P3 ROI65, never RGB96, teacher, semantic slot or owner/depth.
"""
from pathlib import Path
import argparse,ast,collections,hashlib,importlib.util,json,os,sys,time,traceback
import numpy as np
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;V=H.parent;P=V.parent
PARENT=P/'query_visibility_registered_native_render_20261023_v128/v61_baseline_preparation/BASELINE_PROTOCOL_BEFORE_FORWARD.json'
PROTOCOL=H/'V61_PROTOCOL_BEFORE_SOURCE_FORWARD.json';GENERATED=H/'GENERATED_PREP_CHECK.json';OUT=V/'v61_frozen_source_baseline_v1'
EXPECTED={'NATIVE':55296,'PUBLIC_AUX':918};CHUNK=4096
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def bind(p):return {'path':str(Path(p).resolve()),'sha256':sha(p)}
def check(b):assert bind(b['path'])==b,b['path'];return Path(b['path'])
def read(p):return json.loads(Path(p).read_text())
def write(p,x):
 with Path(p).open('x')as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def ash(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def module(p,name):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def state_sha(model):
 h=hashlib.sha256()
 for k,v in sorted(model.state_dict().items()):
  a=v.detach().cpu().contiguous().numpy();h.update(k.encode()+b'\0'+str(a.dtype).encode()+b'\0'+str(a.shape).encode()+b'\0'+a.tobytes())
 return h.hexdigest()
def decode(z,a):return np.where(a,np.where(z>=0,0,1),-1).astype(np.int8)
def proxy_ast(path,expected):
 node=next(n for n in ast.parse(Path(path).read_text()).body if isinstance(n,ast.ClassDef)and n.name=='ProxyHead')
 assert hashlib.sha256(ast.dump(node,include_attributes=False).encode()).hexdigest()==expected
 return ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[]))
def validate_rows(rows):
 assert rows and [r['occurrence_index']for r in rows]==list(range(len(rows)))and len({r['id']for r in rows})==len(rows)
 forbidden={'reference_xy','raw_state','teacher_label','owner','part_id','witnesses','error_px'}
 for r in rows:
  assert not forbidden.intersection(r);q=np.array(r['query_xy'],np.float64);q8=np.array(r['initial_points'],np.float64)
  assert q.shape==(2,)and q8.shape==(8,2)and np.isfinite(q8).all()and np.array_equal(q,q8[r['carrier_slot']])and np.isfinite(r['L'])and r['L']>0
  assert r['full_RGB_format']=='NPZ_RGB_UINT8_HWC'and r['full_RGB_key']=='RGB_uint8'and r['rgb96_row']>=0
def merge(corpora):
 rows=[];origin=[];groups=collections.OrderedDict()
 for name,rr in corpora:
  validate_rows(rr)
  for i,r in enumerate(rr):
   g=len(rows);rows.append(r);origin.append((name,i));k=(r['full_RGB']['path'],r['full_RGB']['sha256'])
   groups.setdefault(k,[]).append(g);first=rows[groups[k][0]]
   assert r['decoded_RGB_sha256']==first['decoded_RGB_sha256']and r['raw_hw']==first['raw_hw']
 assert len({name+'::'+r['id']for(name,_),r in zip(origin,rows)})==len(rows)
 return rows,origin,list(groups.values())
class ROIChunks:
 def __init__(self,out):
  self.out=out;self.x=np.empty((CHUNK,65,9,9),np.float16);self.i=np.empty(CHUNK,np.int32);self.n=0;self.total=0;self.chunks=[]
 def add(self,x,indices):
  lo=0
  while lo<len(x):
   n=min(CHUNK-self.n,len(x)-lo);self.x[self.n:self.n+n]=x[lo:lo+n];self.i[self.n:self.n+n]=indices[lo:lo+n];self.n+=n;self.total+=n;lo+=n
   if self.n==CHUNK:self.flush()
 def flush(self):
  if self.n:
   p=self.out/f'ROIS_{len(self.chunks):03d}.npz';np.savez_compressed(p,p3_roi=self.x[:self.n],global_index=self.i[:self.n]);self.chunks.append({'file':bind(p),'count':self.n});self.n=0
def authorize(entry):
 e=read(entry);assert e['approved_source_frozen_v61']is True
 assert all(e[k]is False for k in ('fit','current223','V112','PnP','initializer_reselection','automatic_retry'))
 for k in ('code','protocol','generated'):check(e[k])
 p=read(PROTOCOL);assert e['code']==p['code']==bind(__file__)and e['protocol']==bind(PROTOCOL)and e['generated']==p['generated']
 assert read(e['generated']['path'])['PASS']and read(e['generated']['path'])['code']==e['code']
 for b in p['assets'].values():check(b)
 for b in p['source_policies'].values():check(b)
 assert {d['name']:d['expected_rows']for d in e['corpora']}==EXPECTED
 assert [d['name']for d in e['corpora']]==list(EXPECTED),'Combined order is NATIVE then PUBLIC_AUX.'
 corpora=[]
 for d in e['corpora']:
  check(d['complete']);check(d['input_rows']);done=read(d['complete']['path']);assert done['completed']and done['PASS']and done['input_manifest']==d['input_rows']
  packet=read(d['input_rows']['path']);assert packet['completed']and len(packet['rows'])==d['expected_rows'];corpora.append((d['name'],packet['rows']))
 assert not OUT.exists(),'Preserve earlier completed/failed run; no automatic restart.'
 return e,p,corpora
def run(entry):
 e,p,corpora=authorize(entry)
 import torch,cv2,PIL
 assert torch.__version__=='2.1.1+cu118'and np.__version__=='1.26.4'and cv2.__version__=='4.9.0'and PIL.__version__=='12.2.0'
 assert Path(sys.executable).resolve()==Path(p['runtime']['python']).resolve()and torch.cuda.is_available()
 rows,origin,groups=merge(corpora);N=len(rows);assert N==56214
 OUT.mkdir();(OUT/'roi_chunks').mkdir();write(OUT/'STARTED.json',{'entry':bind(entry),'rows':N,'unique_full_RGB_bindings':len(groups),'automatic_retry':False,'GT_body_or_hash_read':False})
 started=time.monotonic();back=None
 try:
  torch.set_num_threads(1);cv2.setNumThreads(1);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
  os.environ['YOLO_CONFIG_DIR']=str(H/'runtime/ultralytics')
  math=module(p['assets']['spatial_math']['path'],'v129_v90_spatial_math');bm=module(p['assets']['backbone_code']['path'],'v129_v63_backbone')
  back=bm.FrozenYoloFeatures(p['assets']['backbone_checkpoint']['path'],device='cuda:0')
  with torch.inference_mode():back.predict(np.zeros((512,512,3),np.uint8),already_padded=False,cpu_features=True)
  bs=state_sha(back.yolo.model);ns={'nn':torch.nn};exec(compile(proxy_ast(p['assets']['head_class']['path'],p['ProxyHead_AST_SHA']),p['assets']['head_class']['path'],'exec'),ns)
  head=ns['ProxyHead']();assert sum(v.numel()for v in head.parameters())==28033
  head.load_state_dict(torch.load(p['assets']['head_checkpoint']['path'],map_location='cpu'),strict=True);head.requires_grad_(False).eval().cuda();hs=state_sha(head)
  zall=np.empty(N,np.float32);native=np.zeros(N,bool);any9=native.copy();central4=native.copy();seen=native.copy();chunker=ROIChunks(OUT/'roi_chunks');cache=collections.OrderedDict();checked=set();facts=[];headcalls=0;repeated=False
  def stored(r):
   b=r['rgb96_chunk'];key=(b['path'],b['sha256'])
   if key not in cache:
    if key not in checked:check(b);checked.add(key)
    with np.load(b['path'],allow_pickle=False)as f:cache[key]=(np.array(f['rgb96'],copy=True),np.array(f['valid96'],copy=True),np.array(f['occurrence_index'],copy=True))
    if len(cache)>2:cache.popitem(last=False)
   cache.move_to_end(key);a,v,ix=cache[key];j=r['rgb96_row'];assert ix[j]==r['occurrence_index']and ash(a[j])==r['rgb96_sha256']and ash(v[j])==r['valid96_sha256'];return a[j],v[j]
  for gi,indices in enumerate(groups):
   r=rows[indices[0]];check(r['full_RGB'])
   # Lazy NPZ access reads exactly RGB_uint8; never owner/depth/reference/state.
   with np.load(r['full_RGB']['path'],allow_pickle=False)as f:rgb=np.array(f['RGB_uint8'],copy=True)
   assert rgb.dtype==np.uint8 and rgb.shape==tuple(r['raw_hw'])+(3,)and ash(rgb)==r['decoded_RGB_sha256']
   with torch.inference_mode():captured=back.predict(np.ascontiguousarray(rgb[...,::-1]),already_padded=False,cpu_features=True)
   a=math.expected_affine(r['raw_hw']);gain,offset=bm.canvas_affine(captured['canvas_shape'],captured['input_shape'])
   assert captured['added_border']==100 and list(captured['canvas_shape'])==a['canvas_shape']and list(captured['input_shape'])==a['input_shape']and gain==a['gain']and np.array_equal(offset,a['offset'])
   h,w=a['p3_actual_hw'];plane=captured['p3'].numpy();assert plane.dtype==np.float16 and plane.shape==(64,80,80)and not plane[:,h:,:].any()and not plane[:,:,w:].any()
   q=np.array([rows[i]['query_xy']for i in indices],np.float64);L=np.array([rows[i]['L']for i in indices],np.float64);x,av,v=math.sample_p3(plane[:,:h,:w],q,L,r['raw_hw'],gain,offset,100)
   assert np.isfinite(x).all()and np.array_equal(x[:,64].astype(bool),v)
   nvalid,cont,near=math.query_validity(q,r['raw_hw']);native[indices]=nvalid;any9[indices]=av
   for k,i in enumerate(indices):
    crop,valid=math.crop(rgb,q[k],L[k],96);a96,v96=stored(rows[i]);assert np.array_equal(crop.transpose(1,2,0),a96)and np.array_equal(valid[0],v96)
    central4[i]=bool(valid[0,47:49,47:49].all());old=rows[i]['original_input_availability'];assert bool(cont[k])==old['continuous']and bool(near[k])==old['nearest']and bool(av[k])==old['ROI_any_valid']and central4[i]==old['central4']
   with torch.inference_mode():
    for lo in range(0,len(x),128):
     inp=torch.from_numpy(np.ascontiguousarray(x[lo:lo+128])).float().cuda();zz=head(inp).float().cpu().numpy();headcalls+=1
     if not repeated:assert np.array_equal(zz,head(inp).float().cpu().numpy());repeated=True
     zall[np.asarray(indices[lo:lo+128])]=zz
   assert not seen[indices].any();seen[indices]=True;chunker.add(x,np.asarray(indices,np.int32))
   facts.append({'image_group_index':gi,'full_RGB':r['full_RGB'],'decoded_RGB_sha256':r['decoded_RGB_sha256'],'raw_hw':r['raw_hw'],'row_count':len(indices),'global_indices':indices,'canvas_shape':list(captured['canvas_shape']),'input_shape':list(captured['input_shape']),'P3_actual_shape':[64,h,w],'added_border':100,'gain':gain,'offset':offset.tolist(),'full_P3_persisted':False})
   if(gi+1)%16==0 or gi+1==len(groups):print(json.dumps({'fullRGB_complete':gi+1,'uniqueRGB':len(groups),'rows_complete':int(seen.sum()),'rows':N,'elapsed_seconds':time.monotonic()-started}),flush=True)
   del x,rgb,captured
  chunker.flush();assert chunker.total==N and seen.all()and np.isfinite(zall).all()
  assert state_sha(back.yolo.model)==bs and state_sha(head)==hs and all(not v.requires_grad for v in back.yolo.model.parameters())and all(not v.requires_grad for v in head.parameters())
  assert all(not m.training for m in back.yolo.model.modules())and all(not m.training for m in head.modules());back.close();back=None
  available=native&any9;pred=decode(zall,available);names=np.array([o[0]for o in origin]);local=np.array([o[1]for o in origin],np.int32);ids=np.array([r['id']for r in rows]);globalids=np.array([o[0]+'::'+r['id']for o,r in zip(origin,rows)])
  pf=OUT/'V61_PREDICTIONS_INPUT_ONLY.npz';np.savez_compressed(pf,raw_visible_logit=zall,raw_hidden_logit=-zall,prediction_D0_E1_Uminus1=pred,query_available=available,native_query_valid=native,ROI_any_valid=any9,RGB_central4_available=central4,shared_C16_query_available=np.array([r['query_available']for r in rows],bool),global_index=np.arange(N,dtype=np.int32),corpus=names,source_occurrence_index=local,occurrence_id=ids,global_occurrence_id=globalids,query_xy=np.array([r['query_xy']for r in rows],np.float64),L=np.array([r['L']for r in rows],np.float64))
  fp=OUT/'FULL_RGB_FEATURE_FACTS_INPUT_ONLY.json';write(fp,{'completed':True,'rows':facts})
  rp=OUT/'ROI_PACK_MANIFEST.json';write(rp,{'completed':True,'rows':N,'array':'p3_roi float16[N,65,9,9]','chunks':chunker.chunks,'chunk_index_key':'global_index, potentially noncontiguous because identical fullRGB bindings are grouped; scores always original combined order.','maximum_chunk_rows':CHUNK,'full_P3_persisted':False})
  for b in p['assets'].values():check(b)
  seal=OUT/'PREDICTION_SEAL.json';write(seal,{'completed':True,'PASS':True,'entry':bind(entry),'code':e['code'],'protocol':e['protocol'],'corpora':e['corpora'],'predictions':bind(pf),'ROI_pack':bind(rp),'input_facts':bind(fp),'checkpoint':p['assets']['head_checkpoint'],'backbone_checkpoint':p['assets']['backbone_checkpoint'],'rows':N,'unique_full_RGB_bindings':len(groups),'source_backbone_calls':len(groups),'blank_setup_warmup_calls':1,'head_batches':headcalls,'head_repeat_batches':1,'head_repeat_exact':True,'available':int(available.sum()),'factual_U':int((~available).sum()),'shared_C16_availability_differences':int((available!=np.array([r['query_available']for r in rows],bool)).sum()),'all_saved_RGB96_valid_replay_exact':True,'backbone_state_unchanged':True,'head_state_unchanged':True,'threshold_visible_logit':0,'decoder':'Visible z>=0 D, z<0 E; native continuous&nearest&ROIany unavailable U. Signedzero D.','GT_body_or_hash_read':False,'fit_current_V112_PnP_initializer_reselection':0,'seconds':time.monotonic()-started})
  print(json.dumps({'completed':True,'seal':bind(seal),'rows':N,'unique_full_RGB':len(groups),'seconds':time.monotonic()-started}),flush=True)
 except BaseException:write(OUT/'FAILED_NO_RESTART.json',{'traceback':traceback.format_exc(),'automatic_retry':False});raise
 finally:
  if back is not None:back.close()
def generated():
 z=np.array([-1.,-np.nextafter(0.,1.),-0.,0.,np.nextafter(0.,1.),1.]);assert decode(z,np.ones(6,bool)).tolist()==[1,1,0,0,0,0]and decode(z,np.zeros(6,bool)).tolist()==[-1]*6
 def invented(n,path):
  return [{'occurrence_index':i,'id':str(i),'query_xy':[20.,20.],'initial_points':[[20.,20.]]*8,'carrier_slot':0,'L':16.,'raw_hw':[64,64],'full_RGB':{'path':path,'sha256':'0'*64},'decoded_RGB_sha256':'1'*64,'full_RGB_format':'NPZ_RGB_UINT8_HWC','full_RGB_key':'RGB_uint8','rgb96_row':i}for i in range(n)]
 rr,origin,gg=merge([('A',invented(3,'same')),('B',invented(2,'same'))]);assert len(rr)==5 and origin==[('A',0),('A',1),('A',2),('B',0),('B',1)]and gg==[[0,1,2,3,4]]
 positions=[];n=0
 for size in (72,153,5000,1):
  left=size
  while left:take=min(CHUNK-n,left);positions.append(take);left-=take;n=(n+take)%CHUNK
 assert sum(positions)==5226 and max(positions)<=4096
 assert (65*32*9+32)+(32*32*9+32)+33==28033
 json.dumps({'origin':origin,'groups':gg,'chunks':positions,'available':int(np.ones(5,bool).sum())},allow_nan=False)
 return {'completed':True,'PASS':True,'code':bind(__file__),'checks':{'signedzero_subnormal_visible_polarity':True,'factual_U_independent_of_rawsign':True,'two_corpora_reset_local_indices_namespaced_globalIDs':True,'exact_full_RGB_binding_grouping_retains_everyrow':True,'4096ROI_chunk_partition_bound':True,'ProxyHead28033_and_ROI65_contract':True,'complete_invented_metadata_JSON_serializable':True},'actual_payload_parameter_model_current_V112_reads':0}
def prepare():
 parent=read(PARENT);assert parent['decoder'].startswith('visible z>=0')
 for k in ('backbone_code','spatial_math','head_class'):check(parent['assets'][k])
 proxy_ast(parent['assets']['head_class']['path'],parent['ProxyHead_AST_SHA'])
 g=generated();write(GENERATED,g)
 policies={k:bind(V/f)for k,f in [('native','ROOT_CORPUS_POLICY_BEFORE_PIXELS.json'),('public','ROOT_PUBLIC6_POLICY_BEFORE_COMPOSITES.json')]}
 p={'completed':True,'code':bind(__file__),'generated':bind(GENERATED),'parent_protocol':bind(PARENT),'assets':parent['assets'],'ProxyHead_AST_SHA':parent['ProxyHead_AST_SHA'],'runtime':parent['runtime'],'source_policies':policies,'expected_rows':EXPECTED,'total':56214,'output':str(OUT),'native_preprocessing':'Exact pinned V63 100px REFLECT_101 border once+Ultralytics640 rect affine; ignore detector outputs. Fullframe FP16 P3 once per unique binding, then exact V90 CPUF16→F32 phase/alignFalse/valid65 -> F16 ROI -> F32 CUDA head.','availability':'Historical nativecontinuous&nearest&ROIany; RGBcentral4 and sharedC16availability stored separately, no teacher gate.','storage':'4096ROI chunk buffer,2 cropchunk LRU; transient fullP3, no600MBfullROIarray inRAM. Scores in original NATIVE then PUBLIC_AUX order, both local indices and namespaced IDs.','GT_body_or_hash_read':False,'no_fit_current_V112_PnP':True}
 write(PROTOCOL,p);t={'approved_source_frozen_v61':False,'fit':False,'current223':False,'V112':False,'PnP':False,'initializer_reselection':False,'automatic_retry':False,'code':p['code'],'protocol':bind(PROTOCOL),'generated':bind(GENERATED),'corpora':[]};write(H/'ROOT_ENTRY_TEMPLATE.json',t)
 print(json.dumps({'code':bind(__file__),'protocol':bind(PROTOCOL),'generated':bind(GENERATED),'template':bind(H/'ROOT_ENTRY_TEMPLATE.json'),'command':parent['runtime']['python']+' '+str(Path(__file__).resolve())+' --entry ROOT_ACTUAL_V61_SOURCE_ENTRY.json','actual_forward':0}),flush=True)
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--prepare',action='store_true');a.add_argument('--entry',type=Path);o=a.parse_args()
 if o.prepare:prepare()
 elif o.entry:run(o.entry)
 else:raise SystemExit('Use --prepare or root --entry.')
