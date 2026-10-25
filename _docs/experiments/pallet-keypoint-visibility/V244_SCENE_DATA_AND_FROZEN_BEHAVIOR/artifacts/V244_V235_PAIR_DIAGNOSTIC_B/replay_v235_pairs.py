"""Read-only frozen V235 old-post96 pair behavior reconstruction, Root entry only.

No optimizer, training, V242 features, native recomposition, original RGB or GT.
The original CF features were not saved: this reconstructs the same old pipeline
and never claims byte comparison with an absent historical CF feature artifact.
"""
import argparse,csv,hashlib,json,os,stat,sys,time,traceback,types
from pathlib import Path
import numpy as np
import torch

R=Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility/publication_root_preparation')
H=R/'V244_V235_PAIR_DIAGNOSTIC_B'
OLD_TRAINER={'path':str(R/'V235_DATA_ONLY_TRAINER_A/train_data_only_counterfactual.py'),'sha256':'7a118b4f650e0521edc57bb23dfc0f55cdd5a137bde8b8e0133e5968fa99cacb'}
FIT_COMPLETE={'path':str(R/'V235_ROOT_PUBLIC_COUNTERFACTUAL/data_only_source_fit_v1/COMPLETE.json'),'sha256':'597ddc17d69ee8718c106181e8292e71b230aa768451bfef1f35c546bbf92e48'}
PAIRS_COMPLETE={'path':str(R/'V235_ROOT_PUBLIC_COUNTERFACTUAL/pairs_v1/COMPLETE.json'),'sha256':'8958c47fc5cf65d592346d5b9671c592ad470e799cfdcdb926e6e2788ca2fc13'}
OLD_ENTRY={'path':str(R/'V235_ROOT_PUBLIC_COUNTERFACTUAL/ACTUAL_DATA_ONLY_HEAD_FIT_ENTRY.json'),'sha256':'cba9789738e36327e9bbae1fe84e6e1d26ea04511ae0900d1e585e7c4a870a2c'}
ROOT_PROTOCOL={'path':str(R/'V244_ROOT_FROZEN_BEHAVIOR/PROTOCOL_BEFORE_FROZEN_SOURCE_DIAGNOSTICS.json'),'sha256':'fb283db25a4c8d73d0a90b8e7cd9ee4c7ba53ca1665bd10a9e1c691974cc4de4'}
BASE_SHA='bb4c7a80aa873bf3d34c3c1d44a28cee331503d7ccf52aba85f2d23edb65c3c3'
HEAD_SHA='e5cdd7f4f43c01218fa87a7be787c7a8b290426fc70a3a07470f8deea8e822e3'
SAMPLE_KEYS=('full_view','central_view','fractions','central4','condition','target','family','fit','available','row_id')
ROLES=('COVER','SHAM')


def lexical(value):
 s=str(value)
 if any(t in s.casefold() for t in ('v112','__pycache__','.pyc','current223','publicheld','oldval','v177','v178','source_holdout','v182','private_evaluation')):raise ValueError('Forbidden lexical path before FS')
 p=Path(s)
 if not p.is_absolute() or '..' in p.parts:raise ValueError('Absolute path without traversal')
 return p


def safe(value):
 p=lexical(value)
 for q in reversed((p,*p.parents)):
  try:mode=os.lstat(q).st_mode
  except FileNotFoundError:continue
  if stat.S_ISLNK(mode):raise ValueError('No symlink ancestors')
 return p


def ident(b):
 if not isinstance(b,dict) or not {'path','sha256'}<=set(b)<={'path','sha256','size_bytes'}:raise ValueError('Closed binding schema')
 p=str(lexical(b['path']));s=b['sha256']
 if not isinstance(s,str) or len(s)!=64 or any(c not in '0123456789abcdef' for c in s):raise ValueError('SHA256 format')
 if 'size_bytes' in b and (type(b['size_bytes']) is not int or b['size_bytes']<0):raise ValueError('Size metadata')
 return p,s


def read(path):
 with os.fdopen(os.open(safe(path),os.O_RDONLY|os.O_NOFOLLOW),'rb') as f:return f.read()


def binding(path):
 p=safe(path);h=hashlib.sha256();n=0
 with os.fdopen(os.open(p,os.O_RDONLY|os.O_NOFOLLOW),'rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block);n+=len(block)
 return {'path':str(p),'sha256':h.hexdigest(),'size_bytes':n}


def verify(b,white,pin=None):
 key=ident(b)
 if key not in {ident(x) for x in white} or (pin is not None and key!=ident(pin)):raise ValueError('Exact whitelist before FS')
 actual=binding(b['path'])
 if ident(actual)!=key or ('size_bytes'in b and actual['size_bytes']!=b['size_bytes']):raise ValueError('Bound bytes changed')
 return safe(b['path'])


def exact(a,b):
 if ident(a)!=ident(b):raise ValueError('Historical binding differs')


def write_json(path,obj):
 with os.fdopen(os.open(safe(path),os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600),'w') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
 return binding(path)


def load_utility(b,white):
 p=verify(b,white,OLD_TRAINER);m=types.ModuleType('v244_inert_v235_utility');m.__file__=str(p)
 exec(compile(read(p),str(p),'exec'),m.__dict__);return m


def cf_batch_features(model,crop,preprocess,pixels,support,fractions,central4,condition,device):
 """Same equations/schedule as V235 old trainer lines238-246, not V242 math."""
 views,actual_fractions=preprocess(pixels,support)
 if not torch.equal(actual_fractions,torch.from_numpy(fractions)) or not np.array_equal(support.numpy()[:,0,47:49,47:49].all((1,2)),central4):raise ValueError('Original valid/fractions/centre32 support unchanged')
 output=model.forward_features(views.to(device));n=len(pixels)
 full={k:v[:n] for k,v in output.items() if isinstance(v,torch.Tensor)}
 centre={k:v[n:] for k,v in output.items() if isinstance(v,torch.Tensor)}
 x=crop.crop_features(full,centre,torch.as_tensor(fractions,device=device),torch.as_tensor(central4,device=device),torch.as_tensor(condition,device=device),None)
 if x.shape!=(n,1186) or x.dtype!=torch.float32 or not torch.isfinite(x).all():raise ValueError('Finite old CF feature1186')
 return x


def original_D_rows(path,parents,data):
 """Read saved CSV bytes, numerically interpret successful FIT-D parents only."""
 wanted=set(int(x) for x in parents);out={};traversed=0
 with safe(path).open(newline='') as f:
  for row in csv.DictReader(f):
   traversed+=1;i=int(row['sample_index'])
   if i not in wanted:continue
   if row['role']!='FIT' or row['row_id']!=str(data['row_id'][i]) or int(row['family'])!=data['family'][i] or int(row['target'])!=0 or row['available']!='True':raise ValueError('Saved natural D row identity')
   z=float(row['selected_family_logit'])
   if not np.isfinite(z) or row['prediction']!=('E' if z>=0 else 'D'):raise ValueError('Saved original D scalar decode')
   out[i]={'logit':z,'prediction':row['prediction']}
 if traversed!=7617 or set(out)!=wanted:raise ValueError('Saved original parent coverage')
 return out,traversed


def summarize(rows):
 out={}
 for name,family in [('ALL',None),('SYN_MAIN',0),('YCB_ASSOCIATED_R_AUX',1)]:
  take=[r for r in rows if r['success'] and (family is None or r['family']==family)]
  n=len(take)
  def ratio(v):return v/n if n else None
  direction=np.array([r['COVER_minus_SHAM'] for r in take],np.float64)
  originallyD=[r for r in take if r['original_D_prediction']=='D']
  out[name]={'successful_pairs':n,'COVER_E':sum(r['COVER_prediction']=='E' for r in take),'SHAM_D':sum(r['SHAM_prediction']=='D' for r in take),'COVER_E_and_SHAM_D':sum(r['COVER_prediction']=='E' and r['SHAM_prediction']=='D' for r in take),'COVER_gt_SHAM':int((direction>0).sum()),'COVER_eq_SHAM':int((direction==0).sum()),'COVER_lt_SHAM':int((direction<0).sum()),'COVER_minus_SHAM_mean':float(direction.mean()) if n else None,'COVER_minus_SHAM_median':float(np.median(direction)) if n else None,'COVER_E_rate':ratio(sum(r['COVER_prediction']=='E' for r in take)),'SHAM_D_rate':ratio(sum(r['SHAM_prediction']=='D' for r in take)),'paired_crossing_rate':ratio(sum(r['COVER_prediction']=='E' and r['SHAM_prediction']=='D' for r in take)),'original_saved_D_correct':len(originallyD),'original_D_to_COVER_E':sum(r['COVER_prediction']=='E' for r in originallyD),'SHAM_preserves_original_correct_D':sum(r['SHAM_prediction']=='D' for r in originallyD),'SHAM_preservation_denominator':len(originallyD)}
 return out


def run(entry_path):
 e=json.loads(read(entry_path))
 if e.get('approved_frozen_V235_pair_replay') is not True or e.get('source_only') is not True or e.get('training_authorized') is not False or e.get('automatic_retry') is not False:raise PermissionError('Separate exact Root replay entry required')
 if e['batch_size']!=32 or e['threshold_logit']!=0 or e['save_reconstructed_features'] is not True:raise ValueError('Fixed old batch32/threshold0/cache preservation')
 code,meta,payload=e['code_whitelist'],e['metadata_whitelist'],e['payload_whitelist']
 verify(e['code'],code,binding(__file__));verify(e['implementation_protocol'],code)
 json.loads(read(verify(e['root_protocol'],meta,ROOT_PROTOCOL)))
 fit=json.loads(read(verify(e['fit_complete'],meta,FIT_COMPLETE)));pairs=json.loads(read(verify(e['pairs_complete'],meta,PAIRS_COMPLETE)));old=json.loads(read(verify(e['old_fit_entry'],meta,OLD_ENTRY)))
 if not all(d.get('completed') is True and d.get('PASS') is True for d in (fit,pairs)):raise ValueError('Historical completions must PASS')
 exact(fit['entry'],e['old_fit_entry']);exact(fit['pairs_complete'],e['pairs_complete']);exact(e['checkpoint'],fit['checkpoints']['TREAT_CF_DATA_ONLY']);exact(e['original_D_CSV'],fit['outputs']['TREAT_CF_DATA_ONLY'])
 if e['checkpoint']['sha256']!=HEAD_SHA:raise ValueError('Exact frozen V235 final TREAT only')
 for key,name in [('pair_rgb','PAIR_RGB96.npy'),('parent_indices','parent_indices.npy'),('success','success.npy')]:exact(e[key],pairs['outputs'][name])
 exact(e['valid96'],pairs['source_bindings']['valid96']);exact(e['samples_npz'],pairs['source_bindings']['samples'])
 cfg=e['model_loader_config']
 if cfg!=old['model_loader_config']:raise ValueError('Unchanged historical official loader configuration')
 exact(cfg['official_state'],fit['official_base'])
 if cfg['asset_identity']!=fit['asset_identity'] or fit['base_state_sha256_before_after']!=BASE_SHA:raise ValueError('Same frozen official encoder identity')
 expected=[e[k] for k in ('pair_rgb','parent_indices','success','valid96','samples_npz','checkpoint')]+[cfg['official_state']]
 if {ident(x) for x in payload}!={ident(x) for x in expected} or {ident(x) for x in cfg['payload_whitelist']}!={ident(cfg['official_state'])}:raise ValueError('Exactly7 payloads; no prefix/raw RGB/V242/GT')
 for key in ('source_trainer','crop_module','prefix_runtime','preprocess_CODE'):exact(e[key],old[key])
 if not {ident(x) for x in cfg['code_whitelist']}<={ident(x) for x in code}:raise ValueError('Exact old vendor/loader code closure')
 out=safe(e['output_directory'])
 if out.parent!=R/'V244_ROOT_FROZEN_BEHAVIOR' or out.name!='v235_pair_replay_v1' or out.exists():raise ValueError('Fresh immutable diagnostic directory')
 utility=load_utility(e['old_trainer_code'],code)
 source,crop,_=utility.load_sources(e);preprocess=utility.load_preprocess(e)
 out.mkdir();started=time.monotonic()
 try:
  paths={k:verify(e[k],payload) for k in ('pair_rgb','parent_indices','success','valid96','samples_npz','checkpoint')}
  rgb=np.load(paths['pair_rgb'],mmap_mode='r',allow_pickle=False);valid=np.load(paths['valid96'],mmap_mode='r',allow_pickle=False)
  parents=np.load(paths['parent_indices'],allow_pickle=False);success=np.load(paths['success'],allow_pickle=False)
  with np.load(paths['samples_npz'],allow_pickle=False) as z:data=source.validate_samples({k:z[k] for k in SAMPLE_KEYS},15234)
  if len(data['target'])!=7617 or rgb.shape!=(5406,2,3,96,96) or rgb.dtype!=np.uint8 or valid.shape!=(7617,1,96,96) or valid.dtype!=bool or parents.shape!=(5406,) or parents.dtype!=np.int64 or success.shape!=(5406,) or success.dtype!=bool:raise ValueError('Exact old source/pair shape/dtype')
  eligible=data['fit']&data['available']&(data['target']==0)
  if not np.array_equal(np.sort(parents),np.flatnonzero(eligible)):raise ValueError('All original FIT D5406 once')
  attempts=np.flatnonzero(success)
  if len(attempts)!=5367 or attempts.tolist()!=fit['CF_feature_mapping']['compact_successful_attempt_order']:raise ValueError('Exact old successful compact order')
  if int((data['family'][parents[attempts]]==0).sum())!=4537 or int((data['family'][parents[attempts]]==1).sum())!=830:raise ValueError('Same original FIT success family counts')
  old_D,traversed=original_D_rows(verify(e['original_D_CSV'],meta),parents[attempts],data)
  checkpoint=torch.load(paths['checkpoint'],map_location='cpu',weights_only=True)
  if checkpoint['arm']!='TREAT_CF_DATA_ONLY' or checkpoint['updates']!=1000 or checkpoint['head_parameters']!=76098 or checkpoint['use_lora'] is not False or checkpoint['adapter_parameters']!=0 or checkpoint['base_state_sha256_before_after']!=BASE_SHA:raise ValueError('Frozen V235 finalhead schema')
  source.configure_runtime();model=source.load_backbone(cfg,e['device'],treatment=False);model.eval().requires_grad_(False)
  head=crop.CropFamilyHead().float().to(e['device']);head.load_state_dict(checkpoint['head_state_dict'],strict=True);head.eval().requires_grad_(False)
  if sum(p.numel() for p in head.parameters())!=76098 or any(p.requires_grad for p in (*model.parameters(),*head.parameters())):raise ValueError('Both fixed encoder/head eval only')
  before=source.tensor_state_sha(model.state_dict(),base=True);head_before=source.tensor_state_sha(head.state_dict())
  if before!=BASE_SHA:raise ValueError('Exact old encoder state before replay')
  feature=np.lib.format.open_memmap(out/'SUCCESSFUL_PAIR_FEATURES.npy',mode='w+',dtype=np.float32,shape=(5367,2,1186));logits=np.full((5406,2,2),np.nan,np.float32)
  image_attempt=np.repeat(attempts,2);image_variant=np.tile(np.array([0,1],np.int64),len(attempts))
  featureflat=feature.reshape(-1,1186)
  with torch.no_grad():
   for first in range(0,len(image_attempt),32):
    aa=image_attempt[first:first+32];vv=image_variant[first:first+32];pp=parents[aa]
    if not eligible[pp].all():raise ValueError('Only successful FIT D pixel rows')
    pixels=torch.from_numpy(np.array(rgb[aa,vv],copy=True));support=torch.from_numpy(np.array(valid[pp],copy=True))
    x=cf_batch_features(model,crop,preprocess,pixels,support,data['fractions'][pp],data['central4'][pp],data['condition'][pp],e['device'])
    z=head(x).cpu().numpy()
    if z.shape!=(len(aa),2) or not np.isfinite(z).all():raise ValueError('Finite twofamily logits')
    featureflat[first:first+len(aa)]=x.cpu().numpy();logits[aa,vv]=z
    if first%1024==0 or first+32>=len(image_attempt):print(json.dumps({'stage':'V235_old_CF_reconstruction','images':min(first+32,len(image_attempt)),'total':len(image_attempt)}),flush=True)
  after=source.tensor_state_sha(model.state_dict(),base=True);head_after=source.tensor_state_sha(head.state_dict())
  if before!=after or head_before!=head_after:raise ValueError('Frozen states changed')
  feature.flush();np.save(out/'SUCCESSFUL_ATTEMPT_INDICES.npy',attempts,allow_pickle=False);np.save(out/'SUCCESSFUL_PARENT_SAMPLE_INDICES.npy',parents[attempts],allow_pickle=False)
  rows=[]
  for i,p in enumerate(parents):
   family=int(data['family'][p]);ok=bool(success[i]);cc=None if not ok else float(logits[i,0,family]);ss=None if not ok else float(logits[i,1,family]);saved=old_D.get(int(p))
   row={'pair_index':i,'parent_sample_index':int(p),'row_id':str(data['row_id'][p]),'family':family,'source_role':'FIT','success':ok,'original_teacher_D':True,'original_D_prediction':None if saved is None else saved['prediction'],'original_D_logit':None if saved is None else saved['logit'],
    'COVER_target':1 if ok else -1,'SHAM_target':0 if ok else -1,'COVER_logit0':float(logits[i,0,0]) if ok else None,'COVER_logit1':float(logits[i,0,1]) if ok else None,'SHAM_logit0':float(logits[i,1,0]) if ok else None,'SHAM_logit1':float(logits[i,1,1]) if ok else None,'COVER_selected_logit':cc,'SHAM_selected_logit':ss,'COVER_prediction':'U' if cc is None else 'E' if cc>=0 else 'D','SHAM_prediction':'U' if ss is None else 'E' if ss>=0 else 'D','COVER_minus_SHAM':None if not ok else cc-ss,'COVER_delta_from_original':None if saved is None else cc-saved['logit'],'SHAM_delta_from_original':None if saved is None else ss-saved['logit'],'object_or_logical_aux':'UNKNOWN_PENDING_SAVED_METADATA_JOIN','native_gold':False}
   rows.append(row)
  csvpath=out/'ALL5406_OLD_PAIR_BEHAVIOR.csv'
  with csvpath.open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
  summary=write_json(out/'SUMMARY.json',{'completed':True,'PASS':True,'status':'FROZEN_SOURCE_FIT_REPLAY_NOT_TRAINING','attempts':5406,'successful_pairs':5367,'successful_role_images':10734,'failed_pairs_U':39,'family_metrics':summarize(rows),'object_aux_breakdown':'PENDING existing-metadata-only join; no guessed mapping','denominator':'Preexposed SOURCE FIT constructed weak pairs; not heldout/generalization/native physical gold','original_D_delta_limit':'Original saved natural CSV used the V235 old-prefix original-input route; pairs use old CF fullforward route. Reconstructed same old CF pipeline, not bytecompared to absent saved pair features.'})
  outputs={name:binding(out/name) for name in ('SUCCESSFUL_PAIR_FEATURES.npy','SUCCESSFUL_ATTEMPT_INDICES.npy','SUCCESSFUL_PARENT_SAMPLE_INDICES.npy','ALL5406_OLD_PAIR_BEHAVIOR.csv','SUMMARY.json')}
  complete={'completed':True,'PASS':True,'schema':'V244_FROZEN_OLD_V235_PAIR_BEHAVIOR_RECONSTRUCTION','entry':binding(entry_path),'code':e['code'],'implementation_protocol':e['implementation_protocol'],'root_protocol':e['root_protocol'],'historical_fit_complete':e['fit_complete'],'historical_pairs_complete':e['pairs_complete'],'old_fit_entry':e['old_fit_entry'],'checkpoint':e['checkpoint'],'official_base':cfg['official_state'],'asset_identity':cfg['asset_identity'],'payload_bindings':{k:e[k] for k in ('pair_rgb','parent_indices','success','valid96','samples_npz')},'original_D_CSV':e['original_D_CSV'],'outputs':outputs,'base_state_sha_before':before,'base_state_sha_after':after,'head_state_sha_before':head_before,'head_state_sha_after':head_after,'head_forward_device':e['device'],'encoder_forward_device':e['device'],'batch_size32_images_preprocess64_views':True,'threshold_logit':0,'successful_order':'np.flatnonzero(oldsuccess), interleave COVER then SHAM; same original CF train schedule','saved_features_shape':[5367,2,1186],'saved_features_dtype':'float32','absent_historical_pair_features_or_logits_byte_comparison':False,'SOURCE_DEV_nonpixel_metadata_in_samples_loaded':True,'SOURCE_DEV_nonpixel_metadata_used_for_fit_or_selection':False,'SOURCE_DEV_valid_pixel_rows_numerically_accessed':0,'SOURCE_DEV_new_head_outputs':0,'original_saved_CSV_raw_rows_traversed':traversed,'original_saved_CSV_numeric_rows_interpreted':5367,'failed39_pair_RGB_rows_numerically_accessed':0,'original_RGB_GT_rawdonor_prefix_V242_current_held_payloads':0,'new_optimizer_or_training':0,'model_threshold_or_data_selection':False,'automatic_retry':False,'seconds':time.monotonic()-started}
  write_json(out/'COMPLETE.json',complete);print(json.dumps({'completed':True,'PASS':True,'COMPLETE':binding(out/'COMPLETE.json')}),flush=True)
 except BaseException:
  write_json(out/'FAILED_EXECUTION.json',{'completed':False,'PASS':False,'automatic_retry':False,'traceback':traceback.format_exc()});raise


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--entry',required=True);run(p.parse_args().entry)
