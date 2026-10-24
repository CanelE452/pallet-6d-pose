"""C14 input-only T/Q refresh; inert until an exact root source entry."""
import os,sys
os.environ['XFORMERS_DISABLED']='1';os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
sys.dont_write_bytecode=True
from pathlib import Path
import argparse,hashlib,importlib.util,json,time,traceback
import numpy as np
import torch
from torch.nn import functional as F
H=Path(__file__).resolve().parent;V=H.parent;P=V.parent
DESIGN=V/'ROOT_C14_DESIGN_BEFORE_SOURCE_FORWARD.json';DSHA='01e2bf2ae6f19d554dbd01cca2894aa2ab87219c47d81711d011a1c8f7b9c7f6'
PROTOCOL=H/'FEATURE_PROTOCOL_BEFORE_SOURCE_FORWARD.json';OUT=V/'consistent_context_features';BATCH=8
C11=P/'query_visibility_spatial_target_query_features_20261023_v111/feature_preparation/prepare_spatial_query_features.py'
REPLAY=V/'feature_feasibility/native_context_replay.py'
C13FIT=P/'query_visibility_same_material_query_head_20261023_v118/runs/C13_SAME_MATERIAL_FOREGROUND_SPATIAL_QUERY/FIT_COMPLETE.json'
N,U=46108,15004
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def check(b):assert sha(b['path'])==b['sha256'],b['path'];return Path(b['path'])
def read(p):return json.loads(Path(p).read_text())
def rows(p):return[json.loads(s)for s in Path(p).read_text().splitlines()if s]
def write(p,x):
 with Path(p).open('x')as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def module(p,n):
 spec=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def update_TQ(old,rawT,rawpatch,queries,boxes,fractions,producer):
 """Exact source Q grid/L2/cosines; F/C/P and local fractions are copied literally."""
 if old.dtype!=torch.float32 or old.ndim!=2 or old.shape[1]!=1928:raise ValueError('FP32 old1928')
 n=len(old);frac=np.asarray(fractions,np.float32)
 if frac.shape!=(n,)or not np.isfinite(frac).all()or not((frac>=0)&(frac<=1)).all():raise ValueError('Fraction')
 if rawT.shape!=(n,384)or rawpatch.shape!=(n,256,384):raise ValueError('Context outputs')
 q,inside=producer.sample_Q_raw(rawpatch,queries,boxes,frac>0)
 normalize=lambda x,g:torch.where(torch.as_tensor(g,device=x.device)[:,None],F.normalize(x.float(),p=2,dim=1,eps=1e-12),torch.zeros_like(x))
 T=normalize(rawT,frac>0);Q=normalize(q,inside);x=old.clone();x[:,:384]=T;x[:,384:768]=Q
 for k,(a,b)in enumerate(((0,3),(0,2),(1,3),(1,4))):x[:,1920+k]=F.cosine_similarity(x[:,384*a:384*(a+1)],x[:,384*b:384*(b+1)],dim=1,eps=1e-12)
 x[:,1924]=torch.as_tensor(frac,device=x.device);x[:,1927]=torch.as_tensor(inside,dtype=torch.float32,device=x.device)
 assert torch.equal(x[:,768:1920],old[:,768:1920])and torch.equal(x[:,1925:1927],old[:,1925:1927])and torch.isfinite(x).all()
 return x,inside

def copy_updated_aliases(x,unitrows):
 aliases=[]
 for r in unitrows:
  u=r['source_feature_unit_index'];c=r['canonical_context_feature_unit_index']
  if c!=u:
   assert unitrows[c]['canonical_context_feature_unit_index']==c
   x[u]=x[c];aliases.append([u,c])
 return aliases

def extraction_plan(unitrows,groups,occ,expected=None):
 assert [r['source_feature_unit_index']for r in unitrows]==list(range(len(unitrows)))
 assert [r['global_occurrence_index']for r in occ]==list(range(len(occ)))
 fresh=[r for r in unitrows if r['canonical_context_feature_unit_index']==r['source_feature_unit_index']and r['box_changed']and r['proposed_context_available_geometry']]
 byid={g['context_group_id']:g for g in groups};seen=set();jobs=[]
 for r in fresh:
  key=r['context_group_id']
  if key in seen:continue
  seen.add(key);g=byid[key];us=[u for u in g['canonical_units']if unitrows[u]['box_changed']and unitrows[u]['proposed_context_available_geometry']]
  assert us and g['context_available_geometry']and g['any_canonical_unit_requires_new_context']
  for u in us:
   rr=unitrows[u]
   for k,a in(('full_RGB','full_RGB'),('decoded_RGB_sha256','decoded_RGB_sha256'),('raw_hw','raw_hw'),('proposed_q8_box_xyxy','proposed_q8_box_xyxy')):assert rr[k]==g[a]
  jobs.append(dict(context_group_id=key,units=us,representative_unit=us[0]))
 assert sorted(u for j in jobs for u in j['units'])==sorted(r['source_feature_unit_index']for r in fresh)
 mapping=np.array([r['source_feature_unit_index']for r in occ],np.int32)
 for o in occ:
  r=unitrows[o['source_feature_unit_index']]
  assert o['canonical_context_feature_unit_index']==r['canonical_context_feature_unit_index']and o['query_xy']==r['query_xy']and o['L']==r['L']
 counts=dict(units=len(unitrows),occurrences=len(occ),fresh_units=len(fresh),fresh_views=len(jobs),same_box=sum(r['box_exact_equal']for r in unitrows),aliases=sum(r['canonical_context_feature_unit_index']!=r['source_feature_unit_index']for r in unitrows),unavailable=sum(not r['proposed_context_available_geometry']for r in unitrows))
 if expected is not None:assert counts==expected,counts
 return jobs,mapping,counts

EXPECTED=dict(units=U,occurrences=N,fresh_units=1400,fresh_views=623,same_box=13560,aliases=28,unavailable=72)
def prepare():
 assert sha(DESIGN)==DSHA;d=read(DESIGN);g=H/'GENERATED_FEATURE_TESTS.json';assert read(g)['PASS']and read(g)['code']==bind(__file__)
 for b in d['metadata_inputs'].values():check(b)
 ur=rows(d['metadata_inputs']['units']['path']);gr=rows(d['metadata_inputs']['native_context_groups']['path']);oc=rows(d['metadata_inputs']['occurrences']['path']);jobs,_,counts=extraction_plan(ur,gr,oc,EXPECTED)
 source=read(check(d['feature_extraction']['old_C11_frozen_DINO_operator']));assert bind(C11)==d['features']['exact_C11_producer']
 # Only JSON/code read at prepare. Feature/weight/image bindings inherited; not hashed/opened now.
 sources={r['source_feature_ref']['file']['path']:r['source_feature_ref']['file']for r in ur};assert len(sources)==3
 fit=read(C13FIT);replay_binding=bind(REPLAY)
 p=dict(completed=True,design=bind(DESIGN),code=bind(__file__),generated=bind(g),metadata=d['metadata_inputs'],metadata_plan=d['metadata_plan'],native_replay=replay_binding,C11_producer=bind(C11),C11_protocol=d['feature_extraction']['old_C11_frozen_DINO_operator'],target_math=source['target_crop_code'],target_preprocess=source['C9_target_preprocess'],asset_runtime=source['asset_runtime'],asset_plan=source['asset_plan'],weight=source['weight'],official_repo=source['official_repo'],official_revision=source['official_revision'],official_source_bindings=source['official_source_bindings'],runtime=source['runtime'],source_feature_packets=list(sources.values()),C13_fit=bind(C13FIT),C13_input_metadata_scores=fit['scores'],C13_score_members=['input_unit_index','query_available','factual_query_available','context_available','global_occurrence_index'],counts=counts,encoder_batch_contexts=8,primary_calls=78,first_repeat_calls=1,expected_total_calls=79,output=str(OUT),feature_shape=[U,1928],occurrences=N,allowed_changes=['T0:384','Q384:768','cosines1920:1924','targetfraction1924','Qinside1927'],forbidden_change=['F/C/P768:1920','local/centralfractions1925:1927','originalq/L/IDs/folds/occurrence_unit/factual/availability'],unsupported='T/Q/cosines/targetfraction/Qinside zero; retain cached F/C/P/localfractions and allrows.',alias='Copy full UPDATED canonical vector into28 explicit BEFORE aliases after all canonical updates.',fresh_context_pixels='Native fullRGB matching each changed/original variant; no localRGB96 regeneration or local encoder call.',root_entry_required_before_source_payload_or_weight_hash_read=True,preparation_source_pixels_feature_parameters_encoder_head_fit_current_V112_PnP=0)
 write(PROTOCOL,p);write(H/'ROOT_FEATURE_ENTRY_TEMPLATE.json',dict(approved_actual_source_feature_extraction=False,source_only=True,current_allowed=False,fit_allowed=False,PnP_allowed=False,code=bind(__file__),protocol=bind(PROTOCOL),design=bind(DESIGN),generated=bind(g),output=str(OUT)))
 print(json.dumps(dict(code=bind(__file__),protocol=bind(PROTOCOL),template=bind(H/'ROOT_FEATURE_ENTRY_TEMPLATE.json'),counts=counts,command=f'{sys.executable} {Path(__file__).resolve()} --entry ROOT_APPROVED_ENTRY.json')),flush=True)

def run(entry):
 e=read(entry);assert e['approved_actual_source_feature_extraction']is True and e['source_only']is True and all(e[k]is False for k in('current_allowed','fit_allowed','PnP_allowed'))
 for k in('code','protocol','design','generated'):check(e[k])
 assert e['code']==bind(__file__)and e['protocol']==bind(PROTOCOL)and e['design']==bind(DESIGN)and sha(DESIGN)==DSHA
 p=read(PROTOCOL);assert p['code']==e['code']and p['generated']==e['generated']and read(check(e['generated']))['PASS']and Path(e['output']).resolve()==OUT and not OUT.exists()
 for k in('metadata_plan','native_replay','C11_producer','C11_protocol','target_math','target_preprocess','asset_runtime','asset_plan','weight','C13_fit','C13_input_metadata_scores'):check(p[k])
 for b in[*p['metadata'].values(),*p['source_feature_packets'],*p['official_source_bindings']]:check(b)
 assert torch.__version__==p['runtime']['torch']=='2.1.1+cu118'and torch.cuda.is_available()and Path(sys.executable).resolve()==Path(p['runtime']['executable']).resolve()
 ur=rows(p['metadata']['units']['path']);gr=rows(p['metadata']['native_context_groups']['path']);oc=rows(p['metadata']['occurrences']['path']);jobs,mapping,counts=extraction_plan(ur,gr,oc,EXPECTED)
 with np.load(p['C13_input_metadata_scores']['path'],allow_pickle=False)as f:meta={k:np.array(f[k],copy=True)for k in p['C13_score_members']}
 assert np.array_equal(mapping,meta['input_unit_index'])and np.array_equal(meta['global_occurrence_index'],np.arange(N))
 assert np.array_equal(meta['query_available'],meta['factual_query_available']&meta['context_available'])
 context=np.array([r['proposed_context_available_geometry']for r in ur],bool);assert np.array_equal(context[mapping],meta['context_available'])
 arrays={}
 for b in p['source_feature_packets']:
  with np.load(b['path'],allow_pickle=False)as f:arrays[b['path']]=np.array(f['features'],copy=True)
 x=np.stack([arrays[r['source_feature_ref']['file']['path']][r['source_feature_ref']['index']]for r in ur]);del arrays
 assert x.dtype==np.float32 and x.shape==(U,1928)and np.isfinite(x).all();original=x.copy()
 for r in ur:
  if r['canonical_context_feature_unit_index']!=r['source_feature_unit_index']:assert np.array_equal(original[r['source_feature_unit_index']],original[r['canonical_context_feature_unit_index']])
 producer=module(p['C11_producer']['path'],'C14_exact_C11');pre=module(p['target_preprocess']['path'],'C14_exact_C9_preprocess');replay=module(p['native_replay']['path'],'C14_public_native_replay');math=module(p['target_math']['path'],'C14_target_math')
 freshrows=[r for r in ur if r['canonical_context_feature_unit_index']==r['source_feature_unit_index']and r['box_changed']and r['proposed_context_available_geometry']];reader=replay.NativeRGBReplay(freshrows,allow_public_payload_reads=True)
 OUT.mkdir();write(OUT/'STARTED.json',dict(entry=bind(entry),source_only=True,encoder_calls=0,no_automatic_restart=True));started=time.monotonic()
 try:
  torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
  sys.path.insert(0,p['official_repo']);from dinov2.hub.backbones import dinov2_vits14_reg
  model=dinov2_vits14_reg(pretrained=False);state=torch.load(p['weight']['path'],map_location='cpu',weights_only=True);loaded=model.load_state_dict(state,strict=True);assert not loaded.missing_keys and not loaded.unexpected_keys and tuple(state['register_tokens'].shape)==(1,4,384);del state
  model=model.cuda().float().eval().requires_grad_(False);before=producer.state_sha(model);assert before==read(p['asset_runtime']['path'])['state_before_SHA256'];torch.cuda.reset_peak_memory_stats();calls=0;repeat=False;facts=[]
  for start in range(0,len(jobs),BATCH):
   batch=jobs[start:start+BATCH];images=[];valid=[]
   for job in batch:
    u=job['representative_unit'];rgb,vm=reader.target224(u);fraction=np.float32(vm.mean());assert fraction==np.float32(ur[u]['proposed_context_valid224_fraction_geometry'])and vm.any()
    images.append(rgb.transpose(2,0,1));valid.append(vm[None]);facts.append(dict(context_group_id=job['context_group_id'],representative_unit=u,units=job['units'],native_RGB_sha256=ur[u]['decoded_RGB_sha256'],target_RGB224_F32_sha256=hashlib.sha256(np.ascontiguousarray(rgb).tobytes()).hexdigest(),target_valid_fraction=float(fraction),native_box=ur[u]['proposed_q8_box_xyxy']))
   views=pre.preprocess_target(torch.as_tensor(np.stack(images),device='cuda'),torch.as_tensor(np.stack(valid),device='cuda'))
   with torch.inference_mode():
    out=model.forward_features(views);calls+=1
    if not repeat:
     again=model.forward_features(views);calls+=1
     for key in('x_norm_clstoken','x_norm_patchtokens'):assert torch.equal(out[key],again[key])
     repeat=True
    allunits=[];groupix=[]
    for i,job in enumerate(batch):allunits.extend(job['units']);groupix.extend([i]*len(job['units']))
    ix=torch.as_tensor(groupix,device='cuda');qs=[ur[u]['query_xy']for u in allunits];boxes=[ur[u]['proposed_q8_box_xyxy']for u in allunits];fra=[np.float32(ur[u]['proposed_context_valid224_fraction_geometry'])for u in allunits]
    changed,inside=update_TQ(torch.as_tensor(x[allunits],device='cuda'),out['x_norm_clstoken'][ix],out['x_norm_patchtokens'][ix],qs,boxes,fra,producer)
   x[allunits]=changed.cpu().numpy()
   if(start//BATCH+1)%20==0:print(json.dumps(dict(context_views=start+len(batch),total=623,calls=calls,seconds=time.monotonic()-started)),flush=True)
  changed_unsupported=[r['source_feature_unit_index']for r in ur if r['box_changed']and not r['proposed_context_available_geometry']and r['canonical_context_feature_unit_index']==r['source_feature_unit_index']]
  x[changed_unsupported,:768]=0;x[changed_unsupported,1920:1925]=0;x[changed_unsupported,1927]=0
  aliases=copy_updated_aliases(x,ur);same=np.array([r['box_exact_equal']for r in ur],bool);unsupported=~context
  assert len(aliases)==28 and calls==79 and repeat and len(facts)==623 and len(changed_unsupported)==16
  assert np.array_equal(x[same],original[same])and np.array_equal(x[:,768:1920],original[:,768:1920])and np.array_equal(x[:,1925:1927],original[:,1925:1927])
  assert not x[unsupported,:768].any()and not x[unsupported,1920:1925].any()and not x[unsupported,1927].any()
  assert np.array_equal(x[:,1924],np.array([r['proposed_context_valid224_fraction_geometry']for r in ur],np.float32))and np.isfinite(x).all()
  after=producer.state_sha(model);assert before==after and all(not m.training for m in model.modules())and all(not v.requires_grad for v in model.parameters())and sha(p['weight']['path'])==p['weight']['sha256']
  inside=x[:,1927].astype(bool);assert np.array_equal(x[:,1927],inside.astype(np.float32))and not x[~inside,384:768].any()
  packet=OUT/'SOURCE_CONSISTENT_CONTEXT_FEATURES.npz';np.savez_compressed(packet,features=x,occurrence_unit=mapping,query_available=meta['query_available'],factual_query_available=meta['factual_query_available'],context_available=meta['context_available'],global_occurrence_index=meta['global_occurrence_index'],unit_Q_inside=inside)
  factfile=OUT/'FRESH_CONTEXT623_INPUT_FACTS.jsonl'
  with factfile.open('x')as f:
   for r in facts:f.write(json.dumps(r,separators=(',',':'))+'\n')
  write(OUT/'FEATURE_COMPLETE.json',dict(completed=True,PASS=True,output=bind(packet),code=e['code'],protocol=e['protocol'],entry=bind(entry),design=e['design'],features_shape=[U,1928],occurrences=N,metadata_inputs=p['metadata'],new_query_lineage=p['metadata']['occurrences'],input_unit_lineage=p['metadata']['units'],source_feature_packets=p['source_feature_packets'],C13_input_metadata_scores=p['C13_input_metadata_scores'],fresh_context_facts=bind(factfile),fresh_supported_canonical_units=1400,fresh_unique_context_views=623,unchanged_box_units_exactly_reused=13560,updated_canonical_BEFORE_alias_copies=28,changed_unsupported_zero_context_units=16,all_context_unavailable_units=72,F_C_P_and_local_fractions_bitexact=True,all_mapping_and_availability_exact_C13=True,original_q_L_ID_fold_order_preserved=True,model_state_unchanged=True,frozen_encoder_state_before=before,frozen_encoder_state_after=after,first_batch_repeat_exact=True,encoder_calls=calls,primary_encoder_calls=78,repeat_calls=1,max_views=8,Q_outside_never_adds_U=True,Q_inside_units=int(inside.sum()),new_context_input_family='unclipped_unexpanded_initialq8_minmax',source_GT_labels_or_masks_or_depth_pose_read=0,new_head_fit_private_current_V112_PnP=0,seconds=time.monotonic()-started,peak_GPU_bytes=torch.cuda.max_memory_allocated()))
  print(json.dumps(dict(completed=True,receipt=bind(OUT/'FEATURE_COMPLETE.json'),calls=calls,seconds=time.monotonic()-started)),flush=True)
 except BaseException:
  write(OUT/'FAILED_EXECUTION.json',dict(completed=False,traceback=traceback.format_exc(),no_automatic_retry=True));raise

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');ap.add_argument('--entry',type=Path);a=ap.parse_args()
 if a.prepare:prepare()
 elif a.entry:run(a.entry)
 else:raise SystemExit('Use --prepare or --entry ROOT_APPROVED_ENTRY')
