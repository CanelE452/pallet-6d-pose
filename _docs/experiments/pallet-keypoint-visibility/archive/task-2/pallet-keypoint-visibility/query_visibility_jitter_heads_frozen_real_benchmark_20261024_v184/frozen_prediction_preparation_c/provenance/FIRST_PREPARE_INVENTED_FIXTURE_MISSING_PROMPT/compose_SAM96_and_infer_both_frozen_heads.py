"""V184 original raw8 + frozenSAM96 -> exact raw10 -> both final frozen heads.

Preparation opens source/receipt metadata and invented arrays only. Actual needs
root entry after closed SAM268 and both final8000 heads. No labels, references,
source images, GT, old real scores, optimizer operations, fitting or crop generation.
"""
from pathlib import Path
import argparse,ast,hashlib,importlib.util,json,os,sys,traceback
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8');sys.dont_write_bytecode=True
import numpy as np
import torch
H=Path(__file__).resolve().parent;V=H.parent;P=V.parent
SCOPE=V/'ROOT_SCOPE_BEFORE_NEW_REAL_OUTPUTS_OR_V183_SOURCE_RESULTS.json';SCOPE_SHA='05c628ad960a5715fedd96fb4e3cde7b2a67ee3ce188e49eb644867f4e3e0b25'
CLARIFY=V/'ROOT_REFERENCE_AUTHORITY_CLARIFICATION_BEFORE_REAL_BODIES.json';CLARIFY_SHA='3e60202c1a000b2c5837f1fe826f26d05bc1201edc019e8e3ecf37f709f33fe6'
RAW_COMPLETE=V/'original_real_raw8_inputs_v1/COMPLETE.json';RAW_COMPLETE_SHA='2d90a7863fe80cfce95e3d54ee1ce529b2b800bcc85373cd7b1f0551c5e68a37'
OLD=P/'query_visibility_local_support_raw_head_20261024_v181'
SAMPLER=OLD/'raw_cache_preparation/build_source_raw96_cache.py';SAMPLER_SHA='01b6be15ad7941384ef5a209b397ef1a0b240dec8b403564e5e54c40a702225f'
HEAD=OLD/'head_preparation/local_support_head.py';HEAD_SHA='14a40c4dd14d416c1648e62b408fd7acdf74f06d4113c69785bd737b45aa848b'
F=P/'query_visibility_query_jitter_paired_source_training_20261024_v183'
FIT=F/'runs/C52_C53_JITTER_PAIRED_SOURCE_FIXED8000/COMPLETE.json';FIT_SHA='7e6c231a7daabb69c0b545e06bb04edd0a3da8f2eb5b3ed313e9faa85435ec8c'
FIT_REVIEW=F/'independent_actual_fit_review_a/REVIEW.json';FIT_REVIEW_SHA='421543bc3bac6353e1251707a13b148fbcce7bff8cfc5708ef99db9e3caabe14'
SAM_PREP=V/'sam_preparation_a/PREPARATION_COMPLETE.json'
SAM_PREP_SHA='f72745b9137eac1a365baac4c6ba4651bba3264dcf8fde13b57c1c137d8fd89e'
PROTOCOL=H/'PROTOCOL_BEFORE_REAL_RAW_SAM_OR_HEAD_WEIGHT_BODIES.json';CHECKS=H/'GENERATED_INVENTED_CHECKS.json'
CACHE_OUT=V/'real_raw10_SAM96_inputs_v1';OUT=V/'frozen_C52_C53_real_predictions_v1'
PYTHON='/home/minjae/anaconda3/envs/pallet-yolo26/bin/python'
ARMS=[('C52_RAW96_JITTER_PAIRED_SOURCE',False),('C53_RAW96_SAM_JITTER_PAIRED_SOURCE',True)]
DOMAINS={'CURRENT223_1784':1784,'PUBLIC_HELD45_360':360};U={'CURRENT223_1784':45,'PUBLIC_HELD45_360':108}
META=['row_id','original_global_index','native_slot','query_xy','L','original_query_available']
RAW8={'rgb96','valid96'}|set(META)
SAM12={'image_logits','lowres_logits','predicted_quality','mask_logits_gt0','box_xyxy','original_query_xy','original_L','native_slot','original_global_index','original_occurrence_id','original_query_available','prediction_available'}


def require(ok,msg):
 if not bool(ok):raise AssertionError(msg)
def sha(path):
 path=Path(path);require('_v112' not in str(path).lower(),'Forbidden path before filesystem read');h=hashlib.sha256()
 with path.open('rb') as f:
  for a in iter(lambda:f.read(1<<20),b''):h.update(a)
 return h.hexdigest()
def bind(path):return dict(path=str(Path(path).absolute()),sha256=sha(path))
def checked(b):require(bind(b['path'])=={k:b[k] for k in ['path','sha256']},'Exact binding '+b['path']);return Path(b['path'])
def read(path):return json.loads(Path(path).read_text())
def lines(path):return [json.loads(s) for s in Path(path).read_text().splitlines() if s]
def write(path,v):
 with Path(path).open('x') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def same(a,b):
 a=np.asarray(a);b=np.asarray(b);return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
def values_same(a,b):
 a=np.asarray(a);b=np.asarray(b);return a.shape==b.shape and (np.array_equal(a,b,equal_nan=True) if a.dtype.kind in 'fc' else np.array_equal(a,b))
def ash(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def fingerprints(x):return {k:dict(dtype=str(v.dtype),shape=list(v.shape),sha256=ash(v)) for k,v in x.items()}
def load_npz(b):
 with np.load(checked(b),allow_pickle=False) as z:return {k:z[k] for k in z.files}
def source_helpers():
 require(sha(SAMPLER)==SAMPLER_SHA,'Frozen same-grid sampler source')
 ns=dict(np=np,require=require);names={'same','qarray','validq','grid96','sample_sam'}
 nodes=[n for n in ast.parse(SAMPLER.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in names];require({n.name for n in nodes}==names,'Exact whitelisted SAM sampling math')
 exec(compile(ast.Module(body=nodes,type_ignores=[]),str(SAMPLER),'exec'),ns);return ns

def head_module():
 require(sha(HEAD)==HEAD_SHA,'Frozen115157 head/assembly source');spec=importlib.util.spec_from_file_location('v184_frozen_source_head',HEAD);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m);return m

def object_sha(value):
 h=hashlib.sha256()
 def visit(x):
  if torch.is_tensor(x):
   a=x.detach().cpu().contiguous();h.update(str((str(a.dtype),tuple(a.shape))).encode());h.update(a.numpy().tobytes())
  elif isinstance(x,dict):
   for k in sorted(x,key=str):h.update(str(k).encode());visit(x[k])
  elif isinstance(x,(list,tuple)):
   for a in x:visit(a)
  else:h.update(repr(x).encode())
 visit(value);return h.hexdigest()

def validate_raw8(x,rows,n,domain):
 require(set(x)==RAW8 and len(rows)==n,'Exact original target-free raw8 and all rows')
 for k,dt,shape in [('rgb96',np.uint8,(n,96,96,3)),('valid96',np.bool_,(n,96,96)),('query_xy',np.float64,(n,2)),('L',np.float64,(n,)),('native_slot',np.int8,(n,)),('original_global_index',np.int64,(n,)),('original_query_available',np.bool_,(n,))]:require(x[k].dtype==dt and x[k].shape==shape,'Raw original schema '+k)
 require(x['row_id'].dtype.kind=='U' and x['row_id'].shape==(n,) and len(set(x['row_id'].tolist()))==n,'Raw exact unique row IDs')
 av=x['original_query_available'];require(np.isfinite(x['query_xy'][av]).all() and np.isfinite(x['L'][av]).all() and (x['L'][av]>0).all(),'Available original q/L finitepositive')
 if domain in DOMAINS:
  start=0 if domain=='CURRENT223_1784' else 584;require(np.array_equal(x['original_global_index'],np.arange(start,start+n,dtype=np.int64)) and int((~x['original_query_available']).sum())==U[domain],'Domain-original IDs/U, no concatenation alias')
 for i,r in enumerate(rows):
  require(r['domain']==domain and r['domain_index']==i and str(x['row_id'][i])==r['row_id'] and int(x['original_global_index'][i])==r['original_global_index'] and int(x['native_slot'][i])==r['native_slot'] and bool(x['original_query_available'][i])==r['original_query_available'],'Exact row metadata join')
  q=np.array([np.nan,np.nan] if r['query_xy'] is None else r['query_xy'],np.float64);L=np.nan if r['L'] is None else r['L']
  require(values_same(q,x['query_xy'][i]) and values_same(np.asarray(L),np.asarray(x['L'][i])),'Original F64 q/L identity')
  require(r['grid_precision']=='ORIGINAL_FLOAT64' and r['actual_crop_grid_query_xy']==r['query_xy'] and r['actual_crop_grid_L']==r['L'],'IdentityF64 equals actualcropgrid, FP32alias not sampled')

def validate_sam_record(g,s):
 for key,value in g.items():require(key in s and s[key]==value,'SAM retained exact original group metadata '+key)
 require(isinstance(s['runtime_available'],bool),'SAM runtime flag')
 require(s['RGB_verified'] is True and s['observed_decoded_RGB_sha256']==g['decoded_RGB_sha256'],'All original fullRGB verified by SAM producer, including missingprompt')
 if s['runtime_available']:
  require(g['prompt_available'] and s['outputs'] is not None,'Successful SAM original prompt/output')
 else:require(s['outputs'] is None,'No fake successful outputs on unavailable SAM')

def validate_sam_packet(a,g):
 require(set(a)==SAM12,'Exact frozen fullimage SAM12 fields, no GT/targets')
 hw=tuple(g['raw_hw']);require(a['image_logits'].dtype==np.float32 and a['image_logits'].shape==hw and np.isfinite(a['image_logits']).all(),'ArbitraryHW fullimage logitsFP32')
 require(a['lowres_logits'].dtype==np.float32 and a['lowres_logits'].shape==(256,256) and np.isfinite(a['lowres_logits']).all() and a['predicted_quality'].dtype==np.float32 and a['predicted_quality'].shape==(1,) and np.isfinite(a['predicted_quality']).all(),'Frozen lowres/quality no selection')
 require(a['mask_logits_gt0'].dtype==bool and np.array_equal(a['mask_logits_gt0'],a['image_logits']>0),'Strict original mask sign')
 rr=g['original_input_metadata8'];expected=dict(original_query_xy=np.array(g['original_query_xy8'],np.float64),original_L=np.array(g['original_L8'],np.float64),native_slot=np.arange(8,dtype=np.int8),original_global_index=np.array([r['original_global_index'] for r in rr],np.int64),original_occurrence_id=np.array([r['row_id'] for r in rr]),original_query_available=np.array(g['original_query_available8'],bool))
 for key,value in expected.items():
  require(values_same(a[key],value),'SAM exact original q/L/native/ID/availability '+key)
  if key!='original_occurrence_id':require(a[key].dtype==value.dtype,'SAM numeric metadata dtype '+key)
 require(a['original_occurrence_id'].dtype.kind=='U' and a['box_xyxy'].dtype==np.float64 and a['box_xyxy'].shape==(4,) and values_same(a['box_xyxy'],np.array(g['box_xyxy'],np.float64)),'Original promptbox/Unicode ID')
 require(a['prediction_available'].dtype==bool and np.array_equal(a['prediction_available'],expected['original_query_available']),'SAM originalavailability with successful prompt/runtime')

def original_raw_domain(q,L,hw):
 # Original crop construction accepts finite sentinel coordinates even for
 # factual inputU. SAM validq deliberately does not resurrect those rows.
 require(np.asarray(q).shape==(2,) and np.isfinite(q).all() and np.isfinite(L) and L>0,'Constructed raw finite original F64 q/L')
 u=(np.arange(96,dtype=np.float64)+.5)/96-.5;xx,yy=np.meshgrid(q[0]+u*L,q[1]+u*L);h,w=hw
 return (xx>=-.5)&(xx<w-.5)&(yy>=-.5)&(yy<h-.5)

def compose_domain(x,rows,groupmap,sammap,helpers):
 before=fingerprints(x);n=len(rows);mask=np.zeros((n,96,96),bool);avail=np.zeros_like(mask);proof=[];packet=None;packet_key=None;used={}
 for i,r in enumerate(rows):
  key=(r['domain'],r['group_id']);g=groupmap[key];s=sammap[key];k=r['native_slot'];require(g['original_input_metadata8'][k]==r,'Exact original pergroup row metadata')
  q=x['query_xy'][i];L=x['L'][i];constructed=r['crop_constructed']
  if constructed:
   domain=original_raw_domain(q,L,g['raw_hw'])
   require(np.array_equal(domain,x['valid96'][i]),'Rawvalid all constructed rows equals original F64 grid, including factualU')
  else:require(not x['valid96'][i].any() and not x['rgb96'][i].any(),'Original nonconstructed placeholder pixels preserved')
  if s['runtime_available']:
   if key!=packet_key:
    packet=load_npz(s['outputs']);validate_sam_packet(packet,g);packet_key=key;used[key]=s['outputs']
   m,v=helpers['sample_sam'](packet['image_logits'],q,L,bool(x['original_query_available'][i]),g['raw_hw'])
   mask[i]=m;avail[i]=v
   if x['original_query_available'][i]:require(np.array_equal(v,x['valid96'][i]),'Successful SAM exact same rawgrid domain')
  require(x['original_query_available'][i] or not avail[i].any(),'OriginalinputU cannot be resurrected by SAM')
  proof.append(dict(domain=r['domain'],domain_index=i,row_id=r['row_id'],original_global_index=r['original_global_index'],native_slot=k,original_query_available=r['original_query_available'],prompt_available=g['prompt_available'],runtime_available=s['runtime_available'],sam_mask96_sha256=ash(mask[i]),sam_available96_sha256=ash(avail[i]),original_valid96_sha256=ash(x['valid96'][i]),SAM_output=s['outputs']))
 result=dict(x,sam_mask96=mask,sam_available96=avail);require(set(result)==RAW8|{'sam_mask96','sam_available96'} and before==fingerprints({k:result[k] for k in RAW8}),'Original raw8 every array byte unchanged in target-free raw10')
 return result,proof,used,before


def decode(support,owner,available):
 n=len(available);require(support.dtype==owner.dtype==np.float32 and support.shape==(n,2) and owner.shape==(n,3) and available.dtype==bool,'Frozen output schema')
 require(np.isfinite(support[available]).all() and np.isfinite(owner[available]).all() and np.isnan(support[~available]).all() and np.isnan(owner[~available]).all(),'OriginalinputU rawNaN first')
 sp=np.full((n,2),-1,np.int8);op=np.full(n,-1,np.int8);proxy=np.full(n,-1,np.int8)
 sp[available]=(support[available]>0).astype(np.int8);op[available]=owner[available].argmax(1).astype(np.int8)
 proxy[available]=np.where(sp[available,1]==1,1,np.where(sp[available,0]==1,0,-1)).astype(np.int8);return sp,op,proxy

def infer_model(model,x,use_sam,hm,device):
 n=len(x['row_id']);available=x['original_query_available'];ids=np.flatnonzero(available);support=np.full((n,2),np.nan,np.float32);owner=np.full((n,3),np.nan,np.float32);before=object_sha(model.state_dict());inputsha=fingerprints(x);batches=0
 with torch.inference_mode():
  for start in range(0,len(ids),192):
   ii=ids[start:start+192];rgb=torch.as_tensor(x['rgb96'][ii],device=device);valid=torch.as_tensor(x['valid96'][ii],device=device)
   extra=dict(sam_mask96=torch.as_tensor(x['sam_mask96'][ii],device=device),sam_available=torch.as_tensor(x['sam_available96'][ii],device=device)) if use_sam else {}
   inp=hm.assemble_local_input(rgb,valid,use_sam=use_sam,**extra);a=model(inp);b=model(inp)
   require(all(same(a[k].detach().cpu().numpy(),b[k].detach().cpu().numpy()) for k in a),'All fixed192 batches repeat2 byteexact')
   support[ii]=a['support_logits'].float().cpu().numpy();owner[ii]=a['owner_logits'].float().cpu().numpy();batches+=1
 require(object_sha(model.state_dict())==before and fingerprints(x)==inputsha,'Frozen model and raw10 every byte unchanged')
 sp,op,proxy=decode(support,owner,available)
 return dict(raw_support_logits=support,raw_owner_logits=owner,support_prediction=sp,owner_prediction=op,regional_support_proxy_D0_E1_Uminus1=proxy,support_order=np.array(['visible_end_support','foreign_front_support']),owner_order=np.array(['TARGET','FOREIGN','BACKGROUND']),**{k:x[k] for k in META}),dict(batches=batches,head_calls=batches*2,all_batch_repeat2_exact=True,available_rows_forwarded=len(ids),original_U_not_forwarded=int((~available).sum()),state_before=before,state_after=before,input_fingerprints=inputsha)


def invented():
 checks=[]
 def passed(name,ok):require(ok,name);checks.append(dict(name=name,PASS=True))
 def rejects(fn):
  try:fn()
  except (AssertionError,ValueError):return True
  return False
 helpers=source_helpers();q=np.array([1.25,.75],np.float64);L=2.;logits=np.arange(12,dtype=np.float32).reshape(3,4)-5
 mask,av=helpers['sample_sam'](logits,q,L,True,[3,4]);oracle=np.zeros((96,96),bool);domain=np.zeros_like(oracle)
 for y in range(96):
  for x in range(96):
   px=q[0]+((x+.5)/96-.5)*L;py=q[1]+((y+.5)/96-.5)*L;inside=-.5<=px<3.5 and -.5<=py<2.5;domain[y,x]=inside
   ix=int(np.floor(px));iy=int(np.floor(py));dx=px-ix;dy=py-iy;v=0.
   for xx,yy,w in [(ix,iy,(1-dx)*(1-dy)),(ix+1,iy,dx*(1-dy)),(ix,iy+1,(1-dx)*dy),(ix+1,iy+1,dx*dy)]:
    if 0<=xx<4 and 0<=yy<3:v+=float(logits[yy,xx])*w
   oracle[y,x]=inside and v>0
 passed('ExactV181FP64 SAMsample equals independent scalar fourtap including strict>0/domain',np.array_equal(mask,oracle) and np.array_equal(av,domain))
 m,u=helpers['sample_sam'](logits,q,L,False,[3,4]);passed('SAM unavailable stayszero without erasing original rawvalid',not m.any() and not u.any() and domain.any())
 sentinel=np.array([-1.,-1.],np.float64);sentinel_domain=original_raw_domain(sentinel,4.,[3,4]);sm,sv=helpers['sample_sam'](logits,sentinel,4.,False,[3,4])
 passed('Finite sentinel constructed rawdomain retained while factualU SAM channels stayzero',sentinel_domain.any() and not sm.any() and not sv.any())
 sub=np.nextafter(np.float32(0),np.float32(1));s=np.array([[0.,-0.],[sub,sub],[np.nan,np.nan]],np.float32);o=np.array([[0,0,0],[0,1,1],[np.nan]*3],np.float32);sp,op,proxy=decode(s,o,np.array([1,1,0],bool))
 passed('Strictsign BOTH allowed firsttieargmax foreignpriority Ufirst rawNaN',sp.tolist()==[[0,0],[1,1],[-1,-1]] and op.tolist()==[0,1,-1] and proxy.tolist()==[-1,1,-1])
 passed('AvailableNaN rejected before saving',rejects(lambda:decode(s,o,np.ones(3,bool))))
 torch.set_num_threads(1);torch.manual_seed(1840);hm=head_module();model=hm.Raw96SupportHead().float().eval().requires_grad_(False);passed('Unchanged115157 fixedpool head with everyparameter frozen',sum(p.numel() for p in model.parameters())==115157 and all(not p.requires_grad for p in model.parameters()))
 rgb=torch.arange(2*96*96*3).remainder(256).to(torch.uint8).reshape(2,96,96,3);valid=torch.ones((2,96,96),dtype=torch.bool);zero=torch.zeros_like(valid)
 raw=hm.assemble_local_input(rgb,valid,use_sam=False);missing=hm.assemble_local_input(rgb,valid,use_sam=True,sam_mask96=zero,sam_available=zero);state=object_sha(model.state_dict())
 with torch.inference_mode():a=model(raw);b=model(missing)
 passed('RAW versus missingSAM exact8channels and repeat output; stateunchanged',torch.equal(raw,missing) and all(torch.equal(a[k],b[k]) for k in a) and object_sha(model.state_dict())==state)
 g=dict(raw_hw=[3,4],original_input_metadata8=[dict(row_id='invented:native'+str(k),original_global_index=k,native_slot=k) for k in range(8)],original_query_xy8=[[1.,1.]]*8,original_L8=[2.]*8,original_query_available8=[True]*8,box_xyxy=[0.,0.,2.,2.])
 packet=dict(image_logits=logits,lowres_logits=np.zeros((256,256),np.float32),predicted_quality=np.ones(1,np.float32),mask_logits_gt0=logits>0,box_xyxy=np.array(g['box_xyxy'],np.float64),original_query_xy=np.array(g['original_query_xy8'],np.float64),original_L=np.full(8,2.,np.float64),native_slot=np.arange(8,dtype=np.int8),original_global_index=np.arange(8,dtype=np.int64),original_occurrence_id=np.array([r['row_id'] for r in g['original_input_metadata8']]),original_query_available=np.ones(8,bool),prediction_available=np.ones(8,bool));validate_sam_packet(packet,g)
 bad=dict(packet);bad['original_query_xy']=packet['original_query_xy'].copy();bad['original_query_xy'][0,0]+=.001
 passed('ArbitraryHW SAM12 admitted and originalq drift rejected',rejects(lambda:validate_sam_packet(bad,g)))
 wrong=dict(packet,owner_target=np.zeros(8,np.int8));passed('SAM packet with extra target field rejected',rejects(lambda:validate_sam_packet(wrong,g)))
 r=dict(g,prompt_available=True,decoded_RGB_sha256='invented',domain='A',group_id='G');rec=dict(r,runtime_available=True,outputs=dict(path='invented',sha256='invented'),RGB_verified=True,observed_decoded_RGB_sha256='invented');validate_sam_record(r,rec)
 passed('SAM source/decodedRGB metadata drift rejected',rejects(lambda:validate_sam_record(r,dict(rec,observed_decoded_RGB_sha256='different'))))
 # Execute actual compositor and inference adapters, not only their helpers,
 # against virtual invented packets including constructed-U and missing rows.
 xx=dict(rgb96=np.zeros((8,96,96,3),np.uint8),valid96=np.zeros((8,96,96),bool),row_id=np.array(['invented:native'+str(k) for k in range(8)]),original_global_index=np.arange(8,dtype=np.int64),native_slot=np.arange(8,dtype=np.int8),query_xy=np.ones((8,2),np.float64),L=np.full(8,2.,np.float64),original_query_available=np.ones(8,bool))
 xx['query_xy'][1]=[-1.,-1.];xx['query_xy'][2]=np.nan;xx['L'][2]=np.nan;xx['original_query_available'][1:3]=False;rr=[]
 for k in range(8):
  qq=None if k==2 else xx['query_xy'][k].tolist();ll=None if k==2 else float(xx['L'][k]);rr.append(dict(domain='INVENTED',domain_index=k,row_id=str(xx['row_id'][k]),original_global_index=k,native_slot=k,query_xy=qq,L=ll,original_query_available=bool(xx['original_query_available'][k]),actual_crop_grid_query_xy=qq,actual_crop_grid_L=ll,grid_precision='ORIGINAL_FLOAT64',crop_constructed=k!=2,group_id='INVENTED_GROUP'))
  if k!=2:xx['valid96'][k]=original_raw_domain(xx['query_xy'][k],xx['L'][k],[3,4])
 gg=dict(g,domain='INVENTED',group_id='INVENTED_GROUP',original_input_metadata8=rr,original_query_xy8=xx['query_xy'].tolist(),original_L8=xx['L'].tolist(),original_query_available8=xx['original_query_available'].tolist());pp=dict(packet,original_query_xy=xx['query_xy'],original_L=xx['L'],original_query_available=xx['original_query_available'],prediction_available=xx['original_query_available']);ss=dict(runtime_available=True,outputs=dict(path='<invented>',sha256='invented'));key=('INVENTED','INVENTED_GROUP');validate_raw8(xx,rr,8,'INVENTED')
 global load_npz
 oldload=load_npz;calls=[]
 try:
  load_npz=lambda binding:(calls.append(binding) or pp)
  merged,proof,used,before=compose_domain(xx,rr,{key:gg},{key:ss},helpers)
 finally:load_npz=oldload
 passed('Actual compositor virtualpacket all8 byteparity constructedU/missing retention and singlepacket load',len(calls)==1 and len(proof)==8 and len(used)==1 and before==fingerprints(xx) and merged['valid96'][1].any() and not merged['sam_available96'][1:3].any())
 data,receipt=infer_model(model,merged,False,hm,'cpu')
 passed('Actual inference adapter forwards onlyavailable repeat2 all13fields rawNaN and frozenstate',len(data)==13 and receipt['available_rows_forwarded']==6 and receipt['original_U_not_forwarded']==2 and np.isnan(data['raw_support_logits'][1:3]).all() and (data['owner_prediction'][1:3]==-1).all() and receipt['state_before']==receipt['state_after'])
 passed('Preparation never initializes CUDA',not torch.cuda.is_initialized())
 return checks


def prepare():
 require(sha(SCOPE)==SCOPE_SHA and sha(CLARIFY)==CLARIFY_SHA and sha(RAW_COMPLETE)==RAW_COMPLETE_SHA and sha(FIT)==FIT_SHA and sha(FIT_REVIEW)==FIT_REVIEW_SHA,'Frozen source/input/completedfit/review metadata')
 fit=read(FIT);review=read(FIT_REVIEW);require(fit['completed'] and fit['PASS'] and review['completed'] and review['PASS'],'Completed source fit/integrity review')
 # This file is read only AFTER A explicitly announces its FINAL READY.
 require(sha(SAM_PREP)==SAM_PREP_SHA,'A explicitly FINALREADY preparation binding')
 a=read(SAM_PREP);require(a['completed'] and a['PASS'] and a['prepared_only'] and a['input_complete']==bind(RAW_COMPLETE),'Final A SAM preparation metadata')
 checked(a['code']);checked(a['protocol']);checks=invented()
 write(CHECKS,dict(completed=True,PASS=True,code=bind(__file__),checks=checks,actual_real_raw_SAM_or_source_weight_bodies_or_hash_reads=0,CUDA_initialized=torch.cuda.is_initialized()))
 models={arm:{k:fit['arms'][arm][k] for k in ['checkpoint','final_state_sha','updates','use_sam']} for arm,_ in ARMS}
 p=dict(completed=True,prepared_only=True,scope=bind(SCOPE),authority_clarification=bind(CLARIFY),code=bind(__file__),checks=bind(CHECKS),original_raw_complete=bind(RAW_COMPLETE),source_fit_complete=bind(FIT),source_fit_integrity_review=bind(FIT_REVIEW),head=bind(HEAD),sampling_source=bind(SAMPLER),SAM_preparation=bind(SAM_PREP),SAM_code=a['code'],SAM_protocol=a['protocol'],models=models,
 domains=DOMAINS,originalU=U,cache_output=str(CACHE_OUT),prediction_output=str(OUT),
 composition='ExactV181 whitelist AST grid96/sample_sam FP64 fourtap strict>0. Original raw8everyarray byteparity; original finite-coordinate grid-domain verified for allconstructedrows even sentinelinputU; nonconstructedzero placeholders. SAMmissing zero2channels separately, neverchanges factualinputavailability. Identityq64==actualcropgridq64; FP32aliases not sampled. Original V72 crop96 source lines89-107/132-157 and V153 encode_full230.py139-167 pinned by originalinputprotocol establish F64actualsampling before FP32cachealias casts.',
 SAM_guard='All268 domain/group records retain every originalgroupmetadata field exactly; successful observeddecodedSHA expected; pergroup12fields/hw/prompt/IDs/native/q64/L64/flags strict; fullSAMmask is rawlogits>0, quality unused.',
 inference='Both fixed115157final8000head checkpoints, strictload source metadata/stateSHA, FP32CUDA/TF32false/strictdet/no_grad/eval, fixed192available-row batches everybatchrepeat2/state/inputbytes unchanged. No Urow forwarded; rawNaN/pred-1 first.',
 decoder='supportlogit>0 independently/BOTH allowed; ownerfirsttieargmax T/F/BG; regional foreignYES E elsevisibleYES D elseU. No threshold/model/arm/label choices.',
 output_schema='Raw10 per-domain cache and twoheads x2144=4288 originalrows in sourceV18313key packets META6 +support/owner logits,pred,proxy,orders; allpredictionsseal before annotations/baselines.',
 runtime=dict(executable=PYTHON,torch='2.1.1+cu118',numpy='1.26.4',FP32=True,TF32=False,AMP=False,strict_deterministic=True,batch192=True),checkpoint_loading='Source checkpoint payload includes stored optimizer state; loaded only after actualentry, no optimizer instantiated or used, no fit.',annotation_reference_GT_oldrealprediction_or_fullRGB_reads=0,new_model_fit_or_crop_sampler=False,selected_model=None,adopted=False,fully_solved=False)
 write(PROTOCOL,p);template=H/'ROOT_ACTUAL_COMPOSITE_AND_FROZEN_PREDICTION_ENTRY_TEMPLATE_APPROVED_FALSE.json';write(template,dict(approved_real_SAM96_and_both_frozen_prediction=False,labels_references_old_scores_or_fit_allowed=False,all_two_models_frozen=False,code=bind(__file__),protocol=bind(PROTOCOL),scope=bind(SCOPE),authority_clarification=bind(CLARIFY),sam_complete=None))
 write(H/'PREPARATION_COMPLETE.json',dict(completed=True,PASS=True,prepared_only=True,code=bind(__file__),protocol=bind(PROTOCOL),checks=bind(CHECKS),entry_template=bind(template),actual_raw_SAM_checkpoint_or_annotation_body_reads=0,actual_forward=0));print(json.dumps(dict(completed=True,PASS=True,code=bind(__file__),protocol=bind(PROTOCOL),preparation=bind(H/'PREPARATION_COMPLETE.json'))))


def run(entry):
 e=read(entry);require(e.get('approved_real_SAM96_and_both_frozen_prediction') is True and e.get('labels_references_old_scores_or_fit_allowed') is False and e.get('all_two_models_frozen') is True,'Separate frozenprediction root entry without labels')
 for key,path in [('code',Path(__file__)),('protocol',PROTOCOL),('scope',SCOPE),('authority_clarification',CLARIFY)]:require(e[key]==bind(path),'Frozen entry '+key)
 p=read(PROTOCOL)
 for key,path in [('checks',CHECKS),('original_raw_complete',RAW_COMPLETE),('source_fit_complete',FIT),('source_fit_integrity_review',FIT_REVIEW),('head',HEAD),('sampling_source',SAMPLER),('SAM_preparation',SAM_PREP)]:require(p[key]==bind(path),'Frozen dependency '+key)
 checked(p['SAM_code']);checked(p['SAM_protocol']);require(not CACHE_OUT.exists() and not OUT.exists(),'Preserve all attempts/no automatic overwrite');CACHE_OUT.mkdir();OUT.mkdir()
 write(OUT/'STARTED_BEFORE_RAW_SAM_WEIGHT_BODIES.json',dict(entry=bind(entry),code=bind(__file__),labels_references_or_old_scores_read=False))
 try:
  source=read(RAW_COMPLETE);fit=read(FIT);review=read(FIT_REVIEW);require(source['completed'] and source['PASS'] and fit['completed'] and fit['PASS'] and review['PASS'] and fit['all2_frozen_before_outputs_or_evaluation_access'],'Closed input/source models/integrity')
  require(fit['updates_each']==8000 and fit['total_updates']==16000 and fit['head']==p['head'] and set(fit['arms'])=={a for a,_ in ARMS},'Alltwo fixed finalheads')
  sam=read(checked(e['sam_complete']));require(sam['completed'] and sam['PASS'] and sam['code']==p['SAM_code'] and sam['protocol']==p['SAM_protocol'] and sam['input_complete']==p['original_raw_complete'] and sam['frozen_fit']==p['source_fit_complete'] and sam['independent_fit_audit']==p['source_fit_integrity_review'],'Closed exact frozenSAM268 input/model runtime')
  require(sam['model_state_byte_unchanged'] and sam['input_checkpoint_source_hashes_unchanged'] and sam['first_prompt_group_repeat']['PASS'] and sam['groups_manifest']==source['groups'],'SAM originalgroup closure/frozenstate/repeat exact')
  originalgroups=lines(checked(source['groups']));require(len(originalgroups)==268 and len(sam['groups'])==268,'All original268groups')
  groupmap={(g['domain'],g['group_id']):g for g in originalgroups};sammap={(g['domain'],g['group_id']):g for g in sam['groups']};require(len(groupmap)==len(sammap)==268 and set(groupmap)==set(sammap),'Domain/group identity, overlappingdomain globalIDs never collapsed')
  for key,g in groupmap.items():validate_sam_record(g,sammap[key])
  helpers=source_helpers();caches={};records={};proofs=[];allused={}
  for domain,n in DOMAINS.items():
   record=source['domains'][domain];x=load_npz(record['inputs']);rows=lines(checked(record['metadata']));validate_raw8(x,rows,n,domain)
   merged,proof,used,originalfingerprints=compose_domain(x,rows,groupmap,sammap,helpers);proofs.extend(proof);allused.update(used)
   path=CACHE_OUT/(domain+'_INPUTS.npz');np.savez_compressed(path,**merged);records[domain]=dict(rows=n,inputs=bind(path),metadata=record['metadata'],original_raw8=record['inputs'],original_array_fingerprints=originalfingerprints,original_inputU=U[domain],SAM_allpixel_unavailable_rows=int((~merged['sam_available96'].any((1,2))).sum()))
   caches[domain]=merged
  proofpath=CACHE_OUT/'ALL2144_SAM96_INPUT_PROOFS.jsonl'
  with proofpath.open('x') as f:
   for r in proofs:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
  write(CACHE_OUT/'COMPLETE.json',dict(completed=True,PASS=True,evaluation_inputs_only=True,code=bind(__file__),protocol=bind(PROTOCOL),entry=bind(entry),original_raw_complete=bind(RAW_COMPLETE),SAM_complete=e['sam_complete'],domains=records,proofs=bind(proofpath),all_original_raw8_byteexact=True,rawvalid_all_rows_original_F64_grid_parity=True,SAM_arrays_present=True,annotations_or_targets_read=False,new_crops_or_models=0))
  require(Path(sys.executable).resolve()==Path(PYTHON).resolve() and torch.__version__=='2.1.1+cu118' and np.__version__=='1.26.4' and torch.cuda.is_available(),'Pinned CUDA inference runtime')
  require(os.environ['CUBLAS_WORKSPACE_CONFIG']==':4096:8','Pinned CUBLAS config');torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;require(not torch.is_autocast_enabled(),'No AMP')
  hm=head_module();arms={};total=0
  for arm,use_sam in ARMS:
   fixed=p['models'][arm];require(fixed['checkpoint']==fit['arms'][arm]['checkpoint'] and fixed['updates']==8000 and fixed['use_sam']==use_sam,'Exact fixedarm metadata')
   payload=torch.load(checked(fixed['checkpoint']),map_location='cpu',weights_only=True);require(payload['update']==8000 and payload['seed']==1810 and payload['arm']==arm and payload['use_sam']==use_sam and payload['protocol']==fit['protocol'],'Source finalcheckpoint metadata')
   model=hm.Raw96SupportHead().cuda().float().eval().requires_grad_(False);model.load_state_dict(payload['state_dict'],strict=True);del payload
   require(sum(t.numel() for t in model.parameters())==115157 and object_sha(model.state_dict())==fixed['final_state_sha'],'Exact source final115157 state')
   directory=OUT/arm;directory.mkdir();outputs={};receipts={}
   for domain,x in caches.items():
    data,proof=infer_model(model,x,use_sam,hm,'cuda');path=directory/('FINAL_'+domain+'_OUTPUTS.npz');np.savez_compressed(path,**data);outputs[domain]=bind(path);receipts[domain]=proof;total+=len(x['row_id'])
   require(object_sha(model.state_dict())==fixed['final_state_sha'] and all(not t.requires_grad for t in model.parameters()),'Final model unchanged')
   arms[arm]=dict(checkpoint=fixed['checkpoint'],updates=8000,source_final_state_sha=fixed['final_state_sha'],use_sam=use_sam,outputs=outputs,inference=receipts);del model;checked(fixed['checkpoint'])
  require(total==4288,'Twoheads times all2144 original slots')
  for binding in allused.values():checked(binding)
  for record in source['domains'].values():checked(record['inputs']);checked(record['metadata'])
  write(OUT/'PREDICTION_SEAL.json',dict(completed=True,PASS=True,code=bind(__file__),protocol=bind(PROTOCOL),entry=bind(entry),scope=bind(SCOPE),authority_clarification=bind(CLARIFY),source_fit_complete=bind(FIT),source_fit_integrity_review=bind(FIT_REVIEW),original_raw_complete=bind(RAW_COMPLETE),SAM_complete=e['sam_complete'],raw10_cache_complete=bind(CACHE_OUT/'COMPLETE.json'),arms=arms,original_rows2144=2144,model_domain_rows4288=4288,all_two_models_completed_before_any_annotation_reference_or_baseline_score_reads=True,all_batch_repeat2_state_and_input_bytes_exact=True,original_currentU45_publicU108=True,annotations_reference_GT_oldreal_scores_or_fullRGB_read=False,new_encoder_initializer_SAM_forward_crop_fit_threshold_or_qL_changes=0,selected_model=None,adopted=False,fully_solved=False,requires_separate_saved_evaluation_entry=True,automatic_retry=False))
  print(json.dumps(dict(completed=True,PASS=True,prediction_seal=bind(OUT/'PREDICTION_SEAL.json'))))
 except BaseException:
  write(OUT/'FAILED_NO_AUTOMATIC_RETRY.json',dict(completed=False,traceback=traceback.format_exc(),automatic_retry=False));raise


if __name__=='__main__':
 ap=argparse.ArgumentParser();g=ap.add_mutually_exclusive_group(required=True);g.add_argument('--prepare',action='store_true');g.add_argument('--entry',type=Path);args=ap.parse_args();prepare() if args.prepare else run(args.entry)
