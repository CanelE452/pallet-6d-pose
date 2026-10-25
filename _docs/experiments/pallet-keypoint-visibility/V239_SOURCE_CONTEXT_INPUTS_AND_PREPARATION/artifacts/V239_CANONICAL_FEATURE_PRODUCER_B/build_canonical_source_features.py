"""One frozen full96 feature route; no classifier, optimizer, old prefix or head.

Import is inert. All payloads require a separate exact approved Root entry.
"""
import argparse,ast,hashlib,json,os,sys,time,traceback,types
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
R=Path(__file__).absolute().parent.parent
PROTOCOL=(str(R/'V239_ROOT_CANONICAL_CONTEXT/PROTOCOL_BEFORE_CANONICAL_FEATURES_OR_PAIRED_FIT.json'),'4cbbd1ede0420e1cc6dfe83446b2204141bd99ac29781836b9663c49c77fa021')
PINS={
 'model_loader_code':(str(R/'V233_DETERMINISTIC_CROP_FIX_A/train_crop_adaptation.py'),'ae6e8ceb66ef29a52633ab98511e23a659d7f5b7e8268d300a53b8f4f12d71b1'),
 'crop_module':(str(R/'V233_DETERMINISTIC_CROP_FIX_A/crop_modules.py'),'44f54dfcbf1671a401e748e66ec30463096e6c24b0779f07dda85d7c9f38ae88'),
 'preprocess_CODE':(str(R.parent/'query_visibility_frozen_rgb_head_20261023_v105/frozen_rgb_feature_preparation/prepare_frozen_rgb_features.py'),'cc1e2c331d41aaea5b3d42d144c9abbc40231dd3784d24f52dcc58f41d607578'),
 'context_preprocess_CODE':(str(R.parent/'query_visibility_target_relative_rgb_head_20261023_v108/context_feature_preparation/prepare_target_context_features.py'),'8deae91931c12bcc1692c98ce74fa5e703cfb0d57da980b126d5ca45ca74873e'),
}
INPUT_PINS={'input_complete':'a9d8d8088f4cd61dac9384f2f25f49de1e48cef20787e3638e3f9c0aa5bb1718','input_manifest':'68827e79feb8115294f147675130091792b6c3be26b28640a07bdcd402cd7f36','sample_map':'82d4ca430e038e6567538e7dc373069f621913a480fc3a69c17d32634a1e12c6','samples_npz':'837bbe7ff406e27c6ed2981493a83c2adab830cd5b2de498e4a48de456456e99'}
GEOMETRY_KEYS=('node_available','node_fraction','context_fraction','q8','L8','original_available8','box_xyxy')
SAMPLE_KEYS=('full_view','central_view','fractions','central4','condition','target','family','fit','available','row_id')

def lexical(value):
 s=str(value)
 if any(t in s.casefold() for t in ('v112','__pycache__','.pyc','current223','publicheld','oldval','v177','v178','source_holdout','v182','private_evaluation')):raise ValueError('Forbidden lexical path before filesystem access')
 p=Path(s)
 if not p.is_absolute() or '..' in p.parts:raise ValueError('Absolute path without traversal required')
 return p

def safe(value):
 p=lexical(value)
 for q in reversed((p,*p.parents)):
  if q.is_symlink():raise ValueError('Symlink path forbidden')
 return p

def read(path):
 with os.fdopen(os.open(safe(path),os.O_RDONLY|os.O_NOFOLLOW),'rb') as f:return f.read()

def identity(b):return str(lexical(b['path'])),b['sha256']

def verify(b,white,pin=None):
 if not {'path','sha256'}<=set(b)<= {'path','sha256','size_bytes'}:raise ValueError('Exact binding schema')
 if identity(b) not in {identity(x) for x in white} or (pin is not None and identity(b)!=pin):raise ValueError('Explicit exact whitelist and code pin before FS')
 p=safe(b['path']);h=hashlib.sha256();size=0
 with os.fdopen(os.open(p,os.O_RDONLY|os.O_NOFOLLOW),'rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block);size+=len(block)
 if h.hexdigest()!=b['sha256'] or ('size_bytes' in b and size!=b['size_bytes']):raise ValueError('Bound bytes changed')
 return p

def binding(path):
 p=safe(path);h=hashlib.sha256();n=0
 with os.fdopen(os.open(p,os.O_RDONLY|os.O_NOFOLLOW),'rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block);n+=len(block)
 return {'path':str(p),'sha256':h.hexdigest(),'size_bytes':n}

def write_json(path,obj):
 with os.fdopen(os.open(safe(path),os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600),'w') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
 return binding(path)

def load_code(b,white,name,pin):
 p=verify(b,white,pin);m=types.ModuleType(name);m.__file__=str(p);sys.modules[name]=m
 exec(compile(read(p),str(p),'exec'),m.__dict__);return m

def load_preprocessor(b,white,name,pin):
 p=verify(b,white,pin);nodes=[n for n in ast.parse(read(p)).body if isinstance(n,ast.FunctionDef) and n.name==name]
 if len(nodes)!=1:raise ValueError('One pinned pure preprocessing function')
 ns={'torch':torch,'F':F,'MEAN':(.485,.456,.406),'STD':(.229,.224,.225)}
 exec(compile(ast.Module(body=nodes,type_ignores=[]),str(p),'exec'),ns);return ns[name]

def state_sha(state):
 h=hashlib.sha256()
 for name,value in sorted(state.items()):
  a=value.detach().cpu().contiguous().numpy();h.update(name.encode());h.update(str(a.dtype).encode());h.update(json.dumps(list(a.shape)).encode());h.update(a.tobytes())
 return h.hexdigest()

def validate_geometry(g):
 if set(g)!=set(GEOMETRY_KEYS):raise ValueError('Exact original geometry keys')
 count=len(g['q8'])
 for k in ('original_available8','node_available'):
  if g[k].dtype!=bool or g[k].shape!=(count,8):raise ValueError('Separate original/support bool8')
 for k,shape in [('q8',(count,8,2)),('L8',(count,8)),('box_xyxy',(count,4))]:
  if g[k].dtype!=np.float64 or g[k].shape!=shape:raise ValueError('Original F64 geometry')
 for k,shape in [('node_fraction',(count,8)),('context_fraction',(count,))]:
  if g[k].dtype!=np.float32 or g[k].shape!=shape or not np.isfinite(g[k]).all() or not ((g[k]>=0)&(g[k]<=1)).all():raise ValueError('Original FP32 support fractions')
 if (g['node_available']&~g['original_available8']).any():raise ValueError('Support cannot promote input U')
 return g

def validate_samples(d):
 if set(d)!=set(SAMPLE_KEYS):raise ValueError('Exact original nonpixel samples schema')
 n=len(d['target'])
 for k in ('full_view','central_view','target','family','fit','available','row_id','central4'):
  if d[k].shape!=(n,):raise ValueError('Sample vector shape')
 for k in ('full_view','central_view'):
  if d[k].dtype.kind not in 'iu' or (d[k]<0).any():raise ValueError('Original view indices inert metadata only')
 if d['target'].dtype!=np.int8 or not np.isin(d['target'],[0,1]).all() or d['family'].dtype!=np.int8 or not np.isin(d['family'],[0,1]).all():raise ValueError('Known MAIN/AUX binary metadata unchanged')
 for k in ('fit','available','central4'):
  if d[k].dtype!=bool:raise ValueError('Original boolean metadata')
 if d['row_id'].dtype.kind!='U' or len(set(d['row_id'].tolist()))!=n:raise ValueError('Unique original row IDs')
 if d['fractions'].dtype!=np.float32 or d['fractions'].shape!=(n,2) or not np.isfinite(d['fractions']).all() or not ((d['fractions']>=0)&(d['fractions']<=1)).all():raise ValueError('Original fractions2')
 c=d['condition']
 if c.dtype!=np.float32 or c.shape!=(n,32) or not np.isfinite(c).all() or np.count_nonzero(c[d['family']==1]):raise ValueError('Original condition32/YCB allzero nativeUNKNOWN')
 s=c[d['family']==0]
 if not np.isin(s[:,:8],[0,1]).all() or not (s[:,:8].sum(1)==1).all() or not np.isin(s[:,8:16],[0,1]).all() or not (np.abs(s[:,16:])<=8).all():raise ValueError('Original SYN onehot/mask/relative')
 if len(s):
  native=s[:,:8].argmax(1);xy=s[:,16:].reshape(-1,8,2);mask=s[:,8:16].astype(bool)
  if mask[np.arange(len(s)),native].any() or np.count_nonzero(xy[~mask]):raise ValueError('Self/unavailable zero condition')
 return d

def join_samples(rows,data,g):
 n=len(data['target']);count=len(g['q8']);group=np.full(n,-1,np.int64);slot=np.full(n,-1,np.int64);roles={};ids=set();masked=0
 for row in rows:
  if row['row_id'] in ids:raise ValueError('Duplicate full original row ID')
  ids.add(row['row_id']);gi,si=int(row['group_index']),int(row['active_slot'])
  if not (0<=gi<count and 0<=si<8):raise ValueError('Original group/logical slot only')
  role=bool(row['fit'])
  if gi in roles and roles[gi]!=role:raise ValueError('Group crosses original FIT/DEV')
  roles[gi]=role
  q=np.asarray([np.nan,np.nan] if row['q'] is None else row['q'],np.float64);L=np.float64(np.nan if row['L'] is None else row['L'])
  if q.shape!=(2,) or not np.array_equal(q,g['q8'][gi,si],equal_nan=True) or not np.array_equal(L,g['L8'][gi,si],equal_nan=True):raise ValueError('Original q/L must not change')
  if bool(row['original_query_available'])!=bool(g['original_available8'][gi,si]) or bool(row['crop_available'])!=bool(g['node_available'][gi,si]):raise ValueError('Original input/support availability identity')
  i=row['sample_index']
  if i is None:
   if int(row['target'])!=-1:raise ValueError('Masked S/U remain target-1')
   masked+=1;continue
  i=int(i)
  if not 0<=i<n or group[i]>=0:raise ValueError('Known sample mapped exactly once')
  if row['row_id']!=str(data['row_id'][i]) or int(row['family_code'])!=int(data['family'][i]) or int(row['target'])!=int(data['target'][i]) or role!=bool(data['fit'][i]) or bool(row['crop_available'])!=bool(data['available'][i]):raise ValueError('Original known row/target/role identity')
  group[i],slot[i]=gi,si
 if (group<0).any():raise ValueError('Missing known sample; no fallback')
 return group,slot,masked

def relative_xy(g,group,slot):
 q=g['q8'][group];L=g['L8'][group,slot];a=q[np.arange(len(group)),slot];avail=g['original_available8'][group]
 if not np.isfinite(a).all() or not np.isfinite(L).all() or not (L>0).all() or not np.isfinite(q[avail]).all():raise ValueError('Available original q/L finite')
 out=np.zeros((len(group),8,2),np.float64);np.divide(q-a[:,None,:],L[:,None,None],out=out,where=avail[:,:,None])
 out=np.clip(out,-8,8).astype(np.float32);out[~avail]=0;out[np.arange(len(group)),slot]=0;return out

def node_pixel_batch(model,rgb,valid,preprocess,crop,device):
 """One full+centre batch forward. No classifier or condition-dependent extraction."""
 if rgb.dtype!=torch.uint8 or valid.dtype!=torch.bool:raise ValueError('Original raw pixel/valid dtypes')
 n=len(rgb);views,fractions=preprocess(rgb,valid)
 central4=valid[:,:,47:49,47:49].reshape(n,-1).all(dim=1)
 output=model.forward_features(views.to(device))
 full={k:v[:n] for k,v in output.items() if isinstance(v,torch.Tensor)}
 centre={k:v[n:] for k,v in output.items() if isinstance(v,torch.Tensor)}
 zero=torch.zeros((n,32),dtype=torch.float32,device=device)
 pixel=crop.crop_features(full,centre,fractions.to(device),central4.to(device),zero,None)[:,:1154]
 if pixel.shape!=(n,1154) or pixel.dtype!=torch.float32 or not torch.isfinite(pixel).all():raise ValueError('Finite canonical pixel1154')
 return pixel.detach().cpu().numpy(),fractions.numpy(),central4.numpy()

def extract_canonical(model,node_rgb,node_valid,context_rgb,context_valid,g,preprocess,preprocess_context,crop,device,batch_size=32):
 count=len(g['q8']);pixel=np.zeros((count,8,1154),np.float32);context=np.zeros((count,384),np.float32)
 supported=np.argwhere(g['node_available']);node4=np.zeros((count,8),bool)
 for start in range(0,count,32):
  vv=np.array(node_valid[start:start+32],copy=True);cv=np.array(context_valid[start:start+32],copy=True)
  exact=g['original_available8'][start:start+32]&vv[:,:,47:49,47:49].all((2,3))&vv.any((2,3))
  if not np.array_equal(exact,g['node_available'][start:start+32]) or not np.array_equal(vv.mean((2,3)).astype(np.float32),g['node_fraction'][start:start+32]) or not np.array_equal(cv.mean((1,2)).astype(np.float32),g['context_fraction'][start:start+32]):raise ValueError('Input-only geometry/validity exact')
 repeat=False
 with torch.no_grad():
  for first in range(0,len(supported),batch_size):
   gi,si=supported[first:first+batch_size].T
   rgb=torch.from_numpy(np.array(node_rgb[gi,si],copy=True));valid=torch.from_numpy(np.array(node_valid[gi,si,None],copy=True))
   values,fractions,central4=node_pixel_batch(model,rgb,valid,preprocess,crop,device)
   if not central4.all() or not np.array_equal(fractions[:,0],g['node_fraction'][gi,si]):raise ValueError('Supported original nodes/fraction exact')
   pixel[gi,si]=values;node4[gi,si]=central4
   if first==0:
    repeated,_,_=node_pixel_batch(model,rgb,valid,preprocess,crop,device)
    if values.tobytes()!=repeated.tobytes():raise ValueError('Same-batch canonical feature repeat not byteexact')
    repeat=True
   if first%1024==0:print(json.dumps({'stage':'canonical_node_pixel1154','nodes':min(first+batch_size,len(supported)),'total':len(supported)}),flush=True)
  contexts=np.flatnonzero(g['context_fraction']>0)
  for first in range(0,len(contexts),batch_size):
   gi=contexts[first:first+batch_size];pix=torch.from_numpy(np.array(context_rgb[gi],copy=True));valid=torch.from_numpy(np.array(context_valid[gi,None],copy=True))
   out=model.forward_features(preprocess_context(pix,valid).to(device))['x_norm_clstoken']
   context[gi]=F.normalize(out,p=2,dim=1,eps=1e-12).cpu().numpy()
 if not np.isfinite(pixel).all() or not np.isfinite(context).all() or not repeat:raise ValueError('Finite canonical outputs/repeat')
 return pixel,context,node4,len(supported),len(contexts)

def assemble_known(pixel,data,group,slot,g,node4):
 selected=np.ascontiguousarray(pixel[group,slot]);base=np.concatenate((selected,data['condition']),1).astype(np.float32,copy=False)
 node_cls=np.ascontiguousarray(pixel[:,:,:384])
 if base.shape!=(len(group),1186) or base[:,:384].tobytes()!=node_cls[group,slot].tobytes():raise ValueError('Canonical active/node sharing must be byteexact')
 if base[:,1154:].tobytes()!=data['condition'].tobytes() or selected[:,1152:1154].tobytes()!=data['fractions'].tobytes():raise ValueError('Original condition32/fractions exact')
 if not np.array_equal(node4[group,slot],data['central4']) or not g['node_available'][group,slot].all() or not data['available'].all():raise ValueError('Known original support/central4')
 if np.count_nonzero(pixel[~g['node_available']]) or np.count_nonzero(node_cls[~g['node_available']]):raise ValueError('Unsupported nodes remain zero')
 return base,node_cls

def run(entry_path):
 e=json.loads(read(entry_path))
 if e.get('approved_canonical_SOURCE_feature_build') is not True or e.get('source_only') is not True or e.get('automatic_retry') is not False:raise ValueError('Separate exact Root feature entry required')
 if e.get('batch_size')!=32:raise ValueError('Fixed canonical batch32')
 code,meta,payload=e['code_whitelist'],e['metadata_whitelist'],e['payload_whitelist']
 verify(e['code'],code,(str(safe(__file__)),binding(__file__)['sha256']));verify(e['implementation_protocol'],code)
 protocol=json.loads(read(verify(e['root_protocol'],meta,PROTOCOL)));verify(e['root_scope'],meta)
 m={k:json.loads(read(verify(e[k],meta))) for k in ('input_complete','input_manifest')}
 if not all(x.get('completed') is True and x.get('PASS') is True for x in m.values()):raise ValueError('Sealed original input metadata PASS')
 for key,sha in INPUT_PINS.items():
  if e[key]['sha256']!=sha:raise ValueError('Fixed original source binding')
 if identity(e['sample_map'])!=identity(m['input_manifest']['all_rows']):raise ValueError('Original all15040 join binding')
 inp=m['input_complete'];outputs={Path(b['path']).name:b for b in inp['outputs']}
 if inp['known7617_original_RGB96_and_valid_SHA_exact']!=7617 or inp['new_V235_counterfactuals']!=0 or inp['new_current_or_held_payload']!=0:raise ValueError('Original actual scene crop provenance')
 for key,name in [('node_rgb','NODE_RGB96.npy'),('node_valid','NODE_VALID96.npy'),('context_rgb','CONTEXT_RGB224_F32.npy'),('context_valid','CONTEXT_VALID224.npy'),('geometry_npz','INPUT_GEOMETRY.npz')]:
  if identity(e[key])!=identity(outputs[name]):raise ValueError('Original SOURCE input closed binding')
 cfg=e['model_loader_config'];official=cfg['official_state']
 if not {identity(b) for b in cfg['code_whitelist']}<={identity(b) for b in code}:raise ValueError('Official loader code closure must be in outer Root code whitelist')
 expected_payload={identity(e[k]) for k in ('node_rgb','node_valid','context_rgb','context_valid','geometry_npz','samples_npz')}|{identity(official)}
 if {identity(b) for b in payload}!=expected_payload or {identity(b) for b in cfg['payload_whitelist']}!={identity(official)}:raise ValueError('Only original source packet plus official state, no old prefix/control')
 if official['sha256']!='f433177089a681826f849f194ece3bb48f4d63fb38d32fc837e3dc7a4e5641fb' or cfg['asset_identity']['vendor_revision']!='7764ea0f912e53c92e82eb78a2a1631e92725fc8':raise ValueError('Same frozen official asset')
 out=safe(e['output_directory'])
 if not out.is_relative_to(R/'V239_ROOT_CANONICAL_CONTEXT') or out.exists():raise ValueError('Fresh V239 owned output only')
 out.mkdir();started=time.monotonic();write_json(out/'STARTED.json',{'entry':binding(entry_path),'classifier_or_optimizer':0,'oldprefix_or_oldfinalCONTROL':0})
 try:
  paths={k:verify(e[k],payload) for k in ('node_rgb','node_valid','context_rgb','context_valid','geometry_npz','samples_npz')}
  rows=[json.loads(s) for s in read(verify(e['sample_map'],meta)).decode().splitlines()]
  with np.load(paths['samples_npz'],allow_pickle=False) as z:data=validate_samples({k:z[k] for k in z.files})
  with np.load(paths['geometry_npz'],allow_pickle=False) as z:g=validate_geometry({k:z[k] for k in z.files})
  group,slot,masked=join_samples(rows,data,g);relative=relative_xy(g,group,slot);syn=data['family']==0
  if not np.array_equal(relative[syn],data['condition'][syn,16:].reshape(-1,8,2)):raise ValueError('Original SYN relative geometry unchanged')
  actual_counts={'groups':len(g['q8']),'supported_nodes':int(g['node_available'].sum()),'known':len(data['target']),'FIT':int(data['fit'].sum()),'DEV':int((~data['fit']).sum()),'all_original_rows':len(rows),'masked_target_minus1':masked,'context_supported_groups':int((g['context_fraction']>0).sum())}
  expected_counts={'groups':1767,'supported_nodes':13801,'known':7617,'FIT':5876,'DEV':1741,'all_original_rows':15040,'masked_target_minus1':7423,'context_supported_groups':1760}
  if actual_counts!=expected_counts:raise ValueError('Fixed original SOURCE counts; no fallback')
  arrays={k:np.load(paths[k],mmap_mode='r',allow_pickle=False) for k in ('node_rgb','node_valid','context_rgb','context_valid')}
  for k,shape,dtype in [('node_rgb',(1767,8,3,96,96),np.uint8),('node_valid',(1767,8,96,96),bool),('context_rgb',(1767,3,224,224),np.float32),('context_valid',(1767,224,224),bool)]:
   if arrays[k].shape!=shape or arrays[k].dtype!=dtype:raise ValueError('Exact SOURCE packet shape/dtype')
  crop=load_code(e['crop_module'],code,'crop_modules',PINS['crop_module'])
  loader=load_code(e['model_loader_code'],code,'v239_inert_official_loader',PINS['model_loader_code'])
  preprocess=load_preprocessor(e['preprocess_CODE'],code,'preprocess_views',PINS['preprocess_CODE']);precontext=load_preprocessor(e['context_preprocess_CODE'],code,'preprocess_target',PINS['context_preprocess_CODE'])
  loader.configure_runtime();model=loader.load_backbone(cfg,e['device'],treatment=False);model.eval().requires_grad_(False)
  if any(p.requires_grad for p in model.parameters()) or any(m.training for m in model.modules()):raise ValueError('Fully frozen eval encoder only')
  before=state_sha(model.state_dict())
  pixel,context,node4,supported,contexts=extract_canonical(model,arrays['node_rgb'],arrays['node_valid'],arrays['context_rgb'],arrays['context_valid'],g,preprocess,precontext,crop,e['device'],32)
  base,node=assemble_known(pixel,data,group,slot,g,node4)
  if np.count_nonzero(context[g['context_fraction']==0]):raise ValueError('Unsupported context remains zero')
  features={}
  for name,value in [('ORIGINAL_ALL_FEATURES.npy',base),('NODE_CLS.npy',node),('CONTEXT_CLS.npy',context),('SAMPLE_GROUP_INDEX.npy',group),('SAMPLE_NODE_INDEX.npy',slot),('RELATIVE_XY8.npy',relative),('NODE_PIXEL_FEATURES.npy',pixel)]:
   np.save(safe(out/name),value,allow_pickle=False);features[name]=binding(out/name)
  after=state_sha(model.state_dict())
  if before!=after:raise ValueError('Frozen encoder state changed')
  sharing={'completed':True,'PASS':True,'known_rows':len(group),'active_first384_byte_exact':base[:,:384].tobytes()==node[group,slot].tobytes(),'original_condition32_byte_exact':base[:,1154:].tobytes()==data['condition'].tobytes(),'known_pixel_fractions_byte_exact':base[:,1152:1154].tobytes()==data['fractions'].tobytes(),'original_central4_exact':bool(np.array_equal(node4[group,slot],data['central4'])),'first_node_batch_repeat_byte_exact':True,'active_first384_bytes_sha256':hashlib.sha256(base[:,:384].tobytes()).hexdigest(),'node_selected_bytes_sha256':hashlib.sha256(node[group,slot].tobytes()).hexdigest(),'fresh_construction_from_one_common_output':True,'mismatching_features_overwritten':0,'tolerance_relaxed':False}
  qa=write_json(out/'CANONICAL_ACTIVE_NODE_BYTE_SHARING_QA.json',sharing)
  complete={'completed':True,'PASS':True,'entry':binding(entry_path),'code':e['code'],'implementation_protocol':e['implementation_protocol'],'root_protocol':e['root_protocol'],'root_scope':e['root_scope'],'input_complete':e['input_complete'],'input_manifest':e['input_manifest'],'sample_map':e['sample_map'],'samples_npz':e['samples_npz'],'geometry_npz':e['geometry_npz'],'features':features,'outputs':features,'counts':actual_counts,'canonical_sharing_QA':sharing,'canonical_sharing_QA_binding':qa,'official_base':official,'asset_identity':cfg['asset_identity'],'base_state_sha256_before':before,'base_state_sha256_after':after,'base_state_sha256_before_after':before,'code_provenance':{k:e[k] for k in PINS},'supported_nodes_processed':supported,'context_groups_processed':contexts,'old_prefix_payload_reads':0,'old_final_CONTROL_payload_reads':0,'classifier_optimizer_or_training':0,'new_current_or_held_payload':0,'new_CF':0,'SOURCE_DEV_features_are_fixed_pretrained_only':True,'all_original_rows_retained_by_sample_map':True,'masked7423_binary_loss_excluded':True,'source_referents':{'0':'SYN intended-native witness0 MAIN','1':'YCB associated-r AUX/nativeUNKNOWN; q8 GT-jitter emulation'},'no_forward_GT_refs_pose_owner_target_identity':True,'automatic_retry':False,'actual_seconds':time.monotonic()-started}
  write_json(out/'COMPLETE.json',complete);print(json.dumps({'completed':True,'PASS':True,'COMPLETE':binding(out/'COMPLETE.json')}),flush=True)
 except BaseException:
  write_json(out/'FAILED_EXECUTION.json',{'completed':False,'PASS':False,'automatic_retry':False,'traceback':traceback.format_exc()});raise
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--entry',required=True);run(p.parse_args().entry)
