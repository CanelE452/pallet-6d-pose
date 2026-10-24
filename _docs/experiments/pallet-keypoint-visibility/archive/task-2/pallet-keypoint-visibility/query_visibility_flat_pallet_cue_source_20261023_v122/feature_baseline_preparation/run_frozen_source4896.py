"""One PUBLIC4896 feature + frozen-C14 baseline run; saved weak evaluation is separate.
No import-time payload/model I/O, no fit/current/V112/PnP, no automatic restart.
"""
import os,sys
os.environ['XFORMERS_DISABLED']='1';os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8');sys.dont_write_bytecode=True
from pathlib import Path
import argparse,ast,csv,hashlib,importlib.util,json,time,traceback,collections
import numpy as np
H=Path(__file__).resolve().parent;V=H.parent;P=V.parent
PROTOCOL=H/'FEATURE_BASELINE_PROTOCOL_BEFORE_FORWARD.json';OUT=V/'frozen_C14_source_baseline'
POLICY=V/'ROOT_CONCRETE_CORPUS_POLICY_BEFORE_COMPOSITES.json'
V121=P/'query_visibility_natural_cue_query_baseline_20261023_v121/feature_baseline_preparation'
QUERIES=V/'input_preparation/QUERIES288_INPUT_ONLY.jsonl'
CORPUS_CODE=V/'composite_preparation/build_flat_cue_corpus.py'
N=4896;BATCH=8;VARIANTS=('BEFORE','OPAQUE_COVER','CUE_PRESERVING_SHAM')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for z in iter(lambda:f.read(1<<20),b''):h.update(z)
 return h.hexdigest()
def bind(p):return {'path':str(Path(p).resolve()),'sha256':sha(p)}
def check(b):assert set(b)>={'path','sha256'} and sha(b['path'])==b['sha256'],b;return Path(b['path'])
def read(p):return json.loads(Path(p).read_text())
def rows(p):return[json.loads(s)for s in Path(p).read_text().splitlines()if s]
def write(p,x):
 with Path(p).open('x')as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def jsonl(p,rs):
 with Path(p).open('x')as f:
  for r in rs:f.write(json.dumps(r,separators=(',',':'),allow_nan=False)+'\n')
def writecsv(p,rs):
 with Path(p).open('x',newline='')as f:
  w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
def ash(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def mod(p,n):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def pure(p,names,extra=None):
 tree=ast.parse(Path(p).read_text());nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name in names];assert {n.name for n in nodes}==set(names)
 ns={'np':np};ns.update(extra or{});exec(compile(ast.Module(body=nodes,type_ignores=[]),str(p)+'::inert_AST_functions','exec'),ns);return ns

def validate_queries(qs):
 assert len(qs)==288 and len({r['query_id']for r in qs})==288
 for i,r in enumerate(qs):
  assert r['query_index']==i and r['cue_index']==i//9 and r['jitter_index']==i%9
  assert r['offset_xy']==[(-3,0,3)[i%3],(-3,0,3)[(i%9)//3]]
  q=np.asarray(r['initial_points'],np.float64);assert q.shape==(8,2) and np.isfinite(q).all() and np.array_equal(q[0],r['query_xy'])
  assert r['L']==float(np.clip(.25*np.max(np.ptp(q,axis=0)),16,max(r['raw_hw'])))
  good=~(q==[-1,-1]).all(1);assert r['context_box_xyxy']==np.r_[q[good].min(0),q[good].max(0)].tolist()
  assert r['fixed_carrier_slot']==0 and r['source_emulated_not_initializer'] and not r['original_initializer_touched']
 return True

def validate_occurrences(rs,qs):
 assert len(rs)==N and len({r['occurrence_id']for r in rs})==N
 for i,r in enumerate(rs):
  qi=i//17;role=i%17;variant='BEFORE'if role==0 else VARIANTS[1+(role-1)%2];rank=None if role==0 else(role-1)//2
  assert r['occurrence_index']==r['rgb96_row']==i and r['query_index']==qi and r['cue_index']==qi//9 and r['jitter_index']==qi%9 and r['variant']==variant and r['donor_rank']==rank
  for k,v in qs[qi].items():assert r[k]==v,(i,k)
  assert not any(k in r for k in('candidate_aux_hidden_label_GT_only','geometry_pair_ready','teacher_label','reference_xy','protected_union','uncertainty_bbox','pair_ready'))
  z=r['full_RGB_recipe'];assert z['kind']in('ORIGINAL','OPAQUE_PATCH') and z['original_full_RGB']==r['full_RGB'] and z['original_format']==r['full_RGB_format'] and z['raw_hw']==r['raw_hw']
  if z['kind']=='ORIGINAL':assert 'patch'not in z and z['decoded_RGB_sha256']==r['decoded_RGB_sha256']
  else:assert len(z['placement_xy'])==2 and all(int(v)==v for v in z['placement_xy'])
 return True

def unit_key(r):
 # Exact model-input equality only. Variant, teacher, donor and cue identity are absent.
 return json.dumps([r['full_RGB_recipe']['decoded_RGB_sha256'],r['raw_hw'],r['query_xy'],r['L'],r['context_box_xyxy'],r['rgb96_sha256'],r['valid96_sha256']],separators=(',',':'))
def factual_availability(q,hw,valid96,contextvalid):
 x,y=map(float,q);h,w=hw;finite=bool(np.isfinite([x,y]).all() and [x,y]!=[-1.,-1.])
 continuous=bool(finite and 0<=x<w and 0<=y<h);nearest=bool(finite and 0<=np.floor(x+.5)<w and 0<=np.floor(y+.5)<h)
 c4=bool(valid96[:,47:49,47:49].all());factual=continuous and nearest and c4;context=bool(contextvalid.any())
 return factual,context,factual and context,c4

def authorize(entry):
 e=read(entry);assert e['approved_PUBLIC_source_feature_and_frozen_baseline']is True
 assert all(e[k]is False for k in('fit_allowed','current_allowed','V112_allowed','initializer_allowed','PnP_allowed'))
 for k in('code','protocol','policy','queries288','corpus_seal','input_manifest','stored_crops'):check(e[k])
 p=read(e['protocol']['path']);assert e['code']==p['code']==bind(__file__) and e['protocol']==bind(PROTOCOL) and e['policy']==p['policy']==bind(POLICY) and e['queries288']==p['queries288']==bind(QUERIES)
 assert p['V121_exact_operator']==bind(V121/'run_frozen_public666.py') and p['V121_operator_protocol']==bind(V121/'BASELINE_PROTOCOL_BEFORE_FORWARD.json')
 for k in('C11_producer','C11_protocol','asset_runtime','asset_plan','weight','C8_preprocess','C9_target_preprocess','local_crop_code','target_crop_code','source_model_freeze','checkpoint','head_class_code'):check(p[k])
 for b in p['official_source_bindings']:check(b)
 cs=read(e['corpus_seal']['path']);assert cs['completed']and cs['PASS']and cs['all4896_retained'] and cs['code']==p['corpus_reconstruction_code'];check(cs['code'])
 assert cs['outputs']['OCCURRENCES4896_INPUT_ONLY']==e['input_manifest'] and cs['outputs']['INPUT_CROPS4896_INPUT_ONLY']==e['stored_crops']
 # The weak-sidecar descriptor is compared only. Its bytes/hash are untouched until evaluate_saved.
 assert cs['outputs']['ORIGINAL_WEAK_AND_COMPOSITE_LABELS4896_GT_ONLY']==e['weak_evidence']
 freeze=read(p['source_model_freeze']['path']);assert freeze['completed']and freeze['selected_update']==1200 and freeze['threshold_logit']==0 and freeze['checkpoint']==p['checkpoint']and freeze['head_class_code']==p['head_class_code']and not freeze['current_used_for_fit_or_selection']
 qs=rows(e['queries288']['path']);rs=rows(e['input_manifest']['path']);validate_queries(qs);validate_occurrences(rs,qs)
 return e,p,rs

def run(entry):
 e,p,rs=authorize(entry);assert not OUT.exists(),'Preserve previous attempt, no automatic restart'
 import torch
 assert torch.__version__==p['runtime']['torch']=='2.1.1+cu118' and torch.cuda.is_available() and Path(sys.executable).resolve()==Path(p['runtime']['executable']).resolve()
 OUT.mkdir();write(OUT/'STARTED.json',{'entry':bind(entry),'source_only':True,'weak_evidence_body_or_hash_read':False,'no_automatic_restart':True});started=time.monotonic()
 try:
  torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
  producer=mod(p['C11_producer']['path'],'v122_C11_exact');c8=mod(p['C8_preprocess']['path'],'v122_C8_exact');c9=mod(p['C9_target_preprocess']['path'],'v122_C9_exact');tm=mod(p['target_crop_code']['path'],'v122_target_exact');crop=producer.pure_crop()
  recon=pure(p['corpus_reconstruction_code']['path'],('require','patch_slices','reconstruct_full_RGB'))['reconstruct_full_RGB']
  with np.load(e['stored_crops']['path'],allow_pickle=False)as f:
   rgb96=f['rgb96'];valid96=f['valid96'];indices=f['occurrence_index']
  assert rgb96.dtype==np.uint8 and rgb96.shape==(N,3,96,96) and valid96.dtype==bool and valid96.shape==(N,1,96,96) and np.array_equal(indices,np.arange(N))
  unitmap=[];representatives=[];lookup={}
  for i,r in enumerate(rs):
   assert ash(rgb96[i])==r['rgb96_sha256'] and ash(valid96[i])==r['valid96_sha256']
   key=unit_key(r)
   if key not in lookup:lookup[key]=len(representatives);representatives.append(i)
   else:
    j=representatives[lookup[key]];assert np.array_equal(rgb96[i],rgb96[j])and np.array_equal(valid96[i],valid96[j])
   unitmap.append(lookup[key])
  unitmap=np.asarray(unitmap,np.int32);U=len(representatives);assert np.array_equal(np.unique(unitmap),np.arange(U))
  originalcache={};patchcache={};composed=collections.OrderedDict();readfiles=set()
  def full_image(r):
   z=r['full_RGB_recipe'];ob=z['original_full_RGB'];key=(ob['path'],ob['sha256'])
   if key not in originalcache:
    check(ob);assert z['original_format']=='NPY'
    x=np.load(ob['path'],allow_pickle=False);assert x.dtype==np.uint8 and x.shape==tuple(z['raw_hw'])+(3,) and ash(x)==r['decoded_RGB_sha256'];originalcache[key]=x;readfiles.add(key)
   ck=z['decoded_RGB_sha256']
   if ck in composed:composed.move_to_end(ck);return composed[ck]
   native=originalcache[key]
   if z['kind']=='ORIGINAL':x=native
   else:
    pb=z['patch'];pk=(pb['path'],pb['sha256']);assert Path(pb['path']).resolve().is_relative_to(Path(e['corpus_seal']['path']).parent.resolve())
    if pk not in patchcache:
     check(pb)
     with np.load(pb['path'],allow_pickle=False)as f:patchcache[pk]=(f['rgb'],f['alpha'])
    pr,pa=patchcache[pk];x=recon(native,pr,pa,z['placement_xy'])
   assert x.dtype==np.uint8 and x.shape==tuple(r['raw_hw'])+(3,)and ash(x)==ck
   composed[ck]=x
   while len(composed)>8:composed.popitem(last=False)
   return x
  sys.path.insert(0,p['official_repo']);from dinov2.hub.backbones import dinov2_vits14_reg
  encoder=dinov2_vits14_reg(pretrained=False);state=torch.load(p['weight']['path'],map_location='cpu',weights_only=True);encoder.load_state_dict(state,strict=True);del state;encoder=encoder.cuda().float().eval().requires_grad_(False)
  encstate=producer.state_sha(encoder);assert encstate==read(p['asset_runtime']['path'])['state_before_SHA256'];torch.cuda.reset_peak_memory_stats()
  features=[];facts=[];fa=[];ca=[];qi=[];calls=0;repeated=False
  for start in range(0,U,BATCH):
   reps=representatives[start:start+BATCH];n=len(reps);lr=[];lv=[];tr=[];tv=[];fra=[];central4=[]
   for ix in reps:
    r=rs[ix];native=full_image(r);a,v=crop(native,np.asarray(r['query_xy'],np.float64),r['L'],96)
    assert np.array_equal(a,rgb96[ix]) and np.array_equal(v,valid96[ix]),('Stored RGB96 must match same native composition',ix)
    t,vt=tm.sample_target_context(native,r['context_box_xyxy']);factual,ctx,available,c4=factual_availability(r['query_xy'],r['raw_hw'],v,vt)
    assert factual==r['factual_query_available'] and ctx==r['context_available'] and available==r['query_available']
    fractions=[np.float32(vt.mean()),np.float32(v.mean()),np.float32(v[:,32:64,32:64].mean())]
    lr.append(rgb96[ix]);lv.append(valid96[ix]);tr.append(t.transpose(2,0,1));tv.append(vt[None]);fra.append(fractions);central4.append(c4);fa.append(factual);ca.append(ctx)
    facts.append({'input_unit_index':len(facts),'representative_occurrence_index':ix,'recipe':r['full_RGB_recipe'],'query_xy':r['query_xy'],'L':r['L'],'context_box_xyxy':r['context_box_xyxy'],'RGB96_sha256':ash(rgb96[ix]),'valid96_sha256':ash(valid96[ix]),'central32_sha256':ash(rgb96[ix,:,32:64,32:64]),'central_valid32_sha256':ash(valid96[ix,:,32:64,32:64]),'target_RGB224_F32_sha256':ash(t),'target_valid_sha256':ash(vt),'target_fraction':float(fractions[0]),'local_fraction':float(fractions[1]),'central_fraction':float(fractions[2]),'factual_query_available':factual,'context_available':ctx,'query_available':available})
   local,_=c8.preprocess_views(torch.as_tensor(np.stack(lr),device='cuda'),torch.as_tensor(np.stack(lv),device='cuda'));target=c9.preprocess_target(torch.as_tensor(np.stack(tr),device='cuda'),torch.as_tensor(np.stack(tv),device='cuda'));views=torch.cat((target,local),0);assert views.shape==(3*n,3,224,224)and len(views)<=24
   with torch.inference_mode():
    z=encoder.forward_features(views);calls+=1
    if not repeated:
     again=encoder.forward_features(views);calls+=1;assert all(torch.equal(z[k],again[k])for k in('x_norm_clstoken','x_norm_patchtokens'));repeated=True
    cl=z['x_norm_clstoken'];pt=z['x_norm_patchtokens'];x,inside=producer.assemble(cl[:n],pt[:n],cl[n:2*n],pt[n:2*n],cl[2*n:],[rs[ix]['query_xy']for ix in reps],[rs[ix]['context_box_xyxy']for ix in reps],fra,central4)
   features.append(x.cpu().numpy());qi.extend(inside.tolist())
   if(start//BATCH+1)%40==0:print(json.dumps({'unique_units_done':start+n,'total_units':U,'encoder_calls':calls,'seconds':time.monotonic()-started}),flush=True)
  x=np.concatenate(features);fa=np.asarray(fa,bool);ca=np.asarray(ca,bool);qi=np.asarray(qi,bool);available=(fa&ca)[unitmap]
  assert x.dtype==np.float32 and x.shape==(U,1928)and np.isfinite(x).all()and np.array_equal(x[:,1927],qi.astype(np.float32))and not x[~qi,384:768].any()and calls==(U+7)//8+1
  for i,r in enumerate(rs):
   assert bool(fa[unitmap[i]])==r['factual_query_available']and bool(ca[unitmap[i]])==r['context_available']and bool(available[i])==r['query_available']
  assert producer.state_sha(encoder)==encstate and all(not z.training for z in encoder.modules())and all(not z.requires_grad for z in encoder.parameters());del encoder
  ff=OUT/'SOURCE_FEATURES4896_INPUT_ONLY.npz';np.savez_compressed(ff,features=x,occurrence_unit=unitmap,query_available=available,factual_query_available=fa[unitmap],context_available=ca[unitmap],unit_Q_inside=qi,global_occurrence_index=np.arange(N,dtype=np.int32))
  factfile=OUT/'UNIQUE_INPUT_UNIT_FACTS_INPUT_ONLY.jsonl';jsonl(factfile,facts)
  lineage=OUT/'ALL4896_FEATURE_INPUT_LINEAGE.jsonl';jsonl(lineage,[dict(r,input_unit_index=int(unitmap[i]),feature_available=bool(available[i]))for i,r in enumerate(rs)])
  write(OUT/'FEATURE_COMPLETE.json',{'completed':True,'PASS':True,'entry':bind(entry),'protocol':e['protocol'],'code':e['code'],'corpus_seal':e['corpus_seal'],'input_manifest':e['input_manifest'],'stored_crops':e['stored_crops'],'output':bind(ff),'inputfacts':bind(factfile),'inputlineage':bind(lineage),'occurrences':N,'unique_input_units':U,'features_shape':[U,1928],'canonical_RGB96_replay_all_unique_units_exact':True,'encoder_calls':calls,'primary_calls':(U+7)//8,'repeat_calls':1,'encoder_state_before':encstate,'encoder_state_after':encstate,'model_state_unchanged':True,'first_batch_repeat_exact':repeated,'weak_evidence_body_or_hash_read':False,'fit_current_V112_PnP':0})
  headmod=mod(p['head_class_code']['path'],'v122_C14_head');payload=torch.load(p['checkpoint']['path'],map_location='cpu',weights_only=True);assert payload['update']==1200 and payload['head_class_code']==p['head_class_code'];head=headmod.SpatialTargetQueryHead().cuda().float().eval().requires_grad_(False);head.load_state_dict(payload['state_dict'],strict=True);del payload;assert sum(t.numel()for t in head.parameters())==221825;hs=producer.state_sha(head)
  with torch.inference_mode():raw=head(torch.as_tensor(x,device='cuda'));again=head(torch.as_tensor(x,device='cuda'));assert torch.equal(raw,again)
  raw=raw.cpu().numpy().astype(np.float32);assert raw.shape==(U,)and np.isfinite(raw).all()and producer.state_sha(head)==hs
  beforeU=raw[unitmap];logit=np.where(available,beforeU,0).astype(np.float32);pred=np.where(available,(logit>=0).astype(np.int8),-1).astype(np.int8)
  pf=OUT/'RAW_C14_PREDICTIONS4896_INPUT_ONLY.npz';np.savez_compressed(pf,query_logit=logit,raw_head_logit_before_U=beforeU,prediction=pred,query_available=available,unit_raw_logit=raw,occurrence_unit=unitmap,global_occurrence_index=np.arange(N,dtype=np.int32),occurrence_id=np.array([r['occurrence_id']for r in rs]))
  sc=OUT/'C14_PREDICTIONS4896_INPUT_ONLY.csv';writecsv(sc,[{'occurrence_index':i,'occurrence_id':r['occurrence_id'],'query_index':r['query_index'],'cue_index':r['cue_index'],'jitter_index':r['jitter_index'],'variant':r['variant'],'donor_rank':r['donor_rank'],'source':r['source'],'split':r['split'],'raw_split':r['raw_split'],'source_frame_group':r['source_frame_group'],'query_x':r['query_xy'][0],'query_y':r['query_xy'][1],'L':r['L'],'input_unit_index':int(unitmap[i]),'query_available':bool(available[i]),'raw_aux_external_logit':float(logit[i]),'prediction':int(pred[i])}for i,r in enumerate(rs)])
  write(OUT/'PREDICTION_SEAL.json',{'completed':True,'PASS':True,'entry':bind(entry),'protocol':e['protocol'],'code':e['code'],'source_model_freeze':p['source_model_freeze'],'checkpoint':p['checkpoint'],'input_manifest':e['input_manifest'],'corpus_seal':e['corpus_seal'],'feature_receipt':bind(OUT/'FEATURE_COMPLETE.json'),'features':bind(ff),'predictions':bind(pf),'scoreCSV':bind(sc),'rows':N,'unique_units':U,'available':int(available.sum()),'factual_U':int((~fa[unitmap]).sum()),'context_U':int((~ca[unitmap]).sum()),'final_U':int((~available).sum()),'threshold_logit':0,'decoder':'D z<0; E z>=0, unavailableU first','head_calls':2,'head_repeat_exact':True,'head_state_before':hs,'head_state_after':hs,'model_state_unchanged':True,'weak_evidence_body_or_hash_read':False,'seconds':time.monotonic()-started,'peak_GPU_bytes':torch.cuda.max_memory_allocated(),'fit_current_V112_PnP':0})
  print(json.dumps({'completed':True,'prediction_seal':bind(OUT/'PREDICTION_SEAL.json')}),flush=True)
 except BaseException:
  write(OUT/'FAILED_NO_RESTART.json',{'completed':False,'traceback':traceback.format_exc(),'preserve_all_outputs':True});raise

def metrics(y,p):
 known=np.isin(y,[0,1]);cm=np.zeros((2,3),np.int64)
 for a,b in zip(y[known],p[known]):cm[int(a),2 if b<0 else int(b)]+=1
 div=lambda a,b:float(a/b)if b else None
 return {'all_rows':len(y),'teacher_D':int((y==0).sum()),'teacher_E':int((y==1).sum()),'teacher_U':int((y<0).sum()),'teacher_U_model_available':int(((y<0)&(p>=0)).sum()),'CM_D_E_rows_D_E_U_columns':cm.tolist(),'D_recall':div(cm[0,0],cm[0].sum()),'E_recall':div(cm[1,1],cm[1].sum()),'D_precision':div(cm[0,0],cm[:,0].sum()),'E_precision':div(cm[1,1],cm[:,1].sum()),'model_U_on_known':int(cm[:,2].sum()),'all_model_decisions':{str(c):int((p==c).sum())for c in[-1,0,1]}}

def evaluate_saved(entry):
 e,p,rs=authorize(entry);assert e['allow_weak_evaluation_after_prediction_seal']is True
 seal=read(OUT/'PREDICTION_SEAL.json');assert seal['completed']and seal['PASS']and seal['checkpoint']==p['checkpoint']and not seal['weak_evidence_body_or_hash_read']
 for k in('features','predictions','scoreCSV'):check(seal[k])
 # First weak-sidecar body/hash read is after the completed prediction seal.
 gt=rows(check(e['weak_evidence']));assert len(gt)==N
 with np.load(seal['predictions']['path'],allow_pickle=False)as f:z=f['query_logit'];pr=f['prediction'];a=f['query_available'];indices=f['global_occurrence_index'];ids=f['occurrence_id']
 assert np.array_equal(indices,np.arange(N))and np.array_equal(pr,np.where(a,(z>=0).astype(np.int8),-1))
 for i,(r,g)in enumerate(zip(rs,gt)):
  for k in('occurrence_index','occurrence_id','cue_index','jitter_index','variant','donor_rank'):assert r[k]==g[k]
  assert ids[i]==r['occurrence_id']and g['candidate_aux_hidden_label_GT_only']in(-1,0,1)and not g['human_gold']and g['native_mapping']=='UNKNOWN'
 y=np.array([g['candidate_aux_hidden_label_GT_only']for g in gt],np.int8);summary={};groups=sorted({(r['source'],r['raw_split'])for r in rs})
 for source,fold in [('ALL','ALL')]+groups:
  keep=np.array([(source=='ALL'or r['source']==source)and(fold=='ALL'or r['raw_split']==fold)for r in rs]);centre=np.array([r['jitter_index']==4 for r in rs]);key=source+'::'+fold
  summary[key]={'all9_correlated':metrics(y[keep],pr[keep]),'centre_offset0':metrics(y[keep&centre],pr[keep&centre]),'by_variant':{v:metrics(y[keep&np.array([r['variant']==v for r in rs])],pr[keep&np.array([r['variant']==v for r in rs])])for v in VARIANTS}}
 pair=[]
 for qi in range(288):
  base=17*qi
  for rank in range(8):
   ix=[base,base+1+2*rank,base+2+2*rank];r=rs[base];validteacher=bool(np.array_equal(y[ix],[0,1,0]))
   pair.append({'query_index':qi,'cue_index':r['cue_index'],'jitter_index':r['jitter_index'],'source':r['source'],'raw_split':r['raw_split'],'source_frame_group':r['source_frame_group'],'donor_rank':rank,'teacher_triplet_known_D_E_D':validteacher,'before_label':int(y[ix[0]]),'cover_label':int(y[ix[1]]),'sham_label':int(y[ix[2]]),'before_z':float(z[ix[0]]),'cover_z':float(z[ix[1]]),'sham_z':float(z[ix[2]]),'before_prediction':int(pr[ix[0]]),'cover_prediction':int(pr[ix[1]]),'sham_prediction':int(pr[ix[2]]),'all3_correct':bool(validteacher and np.array_equal(pr[ix],[0,1,0])),'cover_minus_sham':float(z[ix[1]]-z[ix[2]])})
 out=OUT/'saved_weak_evaluation';assert not out.exists();out.mkdir();pf=out/'PAIRED_QUERY_RANK2304_DIAGNOSTICS.csv';writecsv(pf,pair)
 pranks={}
 for source,fold in [('ALL','ALL')]+groups:
  for rank in range(8):
   ss=[r for r in pair if(source=='ALL'or r['source']==source)and(fold=='ALL'or r['raw_split']==fold)and r['donor_rank']==rank];known=[r for r in ss if r['teacher_triplet_known_D_E_D']]
   pranks[f'{source}::{fold}::R{rank}']={'planned_query_rank_conditions':len(ss),'known_triplets':len(known),'all3_correct':sum(r['all3_correct']for r in known),'centre_known_triplets':sum(r['jitter_index']==4 for r in known),'centre_all3_correct':sum(r['jitter_index']==4 and r['all3_correct']for r in known)}
 rf=out/'SOURCE_RESULTS.json';write(rf,{'completed':True,'prediction_seal':bind(OUT/'PREDICTION_SEAL.json'),'weak_evidence':e['weak_evidence'],'metrics':summary,'paired_by_rank':pranks,'paired_table':bind(pf),'all4896_retained':True,'failed_condition_teacherU_does_not_gate_model_available':True,'source_model_freeze':p['source_model_freeze'],'checkpoint':p['checkpoint'],'limits':['Conditional AI weak structural-cue visibility; no native/human/pixel gold.','Nine offsets/ranks/variants are correlated derivatives. LOCO original folds retained; allTOMIE one development episode, noTRAIN admission.','Frozen C14 descriptive baseline, no headfit/threshold/checkpoint/arm choice.','All teacherU and factual-input/modelU are separate; known unavailable remains in recall denominators.'],'new_model_fit_current_V112_PnP':0})
 write(out/'EVALUATION_SEAL.json',{'completed':True,'PASS':True,'prediction_seal':bind(OUT/'PREDICTION_SEAL.json'),'weak_evidence':e['weak_evidence'],'outputs':[bind(rf),bind(pf)],'new_model_forward':0});print(json.dumps({'completed':True,'evaluation_seal':bind(out/'EVALUATION_SEAL.json')}),flush=True)

def prepare():
 assert not PROTOCOL.exists();old=read(V121/'BASELINE_PROTOCOL_BEFORE_FORWARD.json');qs=rows(QUERIES);validate_queries(qs)
 # Three bounded invented checks; source pixels, weights and model modules untouched.
 recon=pure(CORPUS_CODE,('require','patch_slices','reconstruct_full_RGB'))['reconstruct_full_RGB'];im=np.zeros((5,6,3),np.uint8);pat=np.full((3,4,3),77,np.uint8);al=np.ones((3,4),bool);a=recon(im,pat,al,[-1,3]);expected=im.copy();expected[3:5,0:3]=77;assert np.array_equal(a,expected)and not im.any()
 f,c,v,c4=factual_availability([2.,2.],[5,6],np.ones((1,96,96),bool),np.ones((224,224),bool));assert f and c and v and c4 # no teacher input in signature
 assert metrics(np.array([0,1,-1]),np.array([0,-1,1]))['CM_D_E_rows_D_E_U_columns']==[[1,0,0],[0,0,1]]
 keys=('asset_runtime','asset_plan','weight','official_repo','official_revision','official_source_bindings','runtime','C8_preprocess','C9_target_preprocess','local_crop_code','target_crop_code','C11_producer','C11_protocol','source_model_freeze','checkpoint','head_class_code')
 p={k:old[k]for k in keys};p.update(completed=True,code=bind(__file__),policy=bind(POLICY),queries288=bind(QUERIES),V121_exact_operator=bind(V121/'run_frozen_public666.py'),V121_operator_protocol=bind(V121/'BASELINE_PROTOCOL_BEFORE_FORWARD.json'),corpus_reconstruction_code=bind(CORPUS_CODE),output=str(OUT),planned_occurrences=N,source_cues=32,metadata_query_count=288,batch_unique_units=8,maximum_encoder_views=24,feature_operator='Exact C11 assemble T/Q/F/C/P384 each +4cos/3fractions/Qinside=1928; exact C8/C9 preprocessing unchanged',availability='finite non-sentinel native continuous AND nearest query support AND stored/canonical RGB96 central4valid AND context any; Qinside feature only; no weak label/construction status input',stored_RGB96='Use actual corpus U8 RGB96/boolvalid verbatim; replay every unique input crop from the same reconstructed native fullRGB and assert exact equality',dedup='Only exact composition decodedSHA+HW/q/L/context+localRGB/validSHA; all4896 map/order/teacher records retained',head='Frozen C14selected1200 checkpoint only, raw0 D<0/E>=0/Ufirst, no fit',weak_evaluation_boundary='weak-sidecar descriptor only before completedPREDICTION_SEAL; first body/hash read in separately invoked --evaluate-saved',code_checks={'queries288_order_q8_L_context':True,'invented_native_clipping_reconstruction':True,'teacher_not_availability_argument':True,'invented_CM_knownU_denominator':True,'real_pixels_weights_model_forward_fit_current_V112':0})
 write(PROTOCOL,p)
 t={'approved_PUBLIC_source_feature_and_frozen_baseline':False,'fit_allowed':False,'current_allowed':False,'V112_allowed':False,'initializer_allowed':False,'PnP_allowed':False,'allow_weak_evaluation_after_prediction_seal':True,'code':bind(__file__),'protocol':bind(PROTOCOL),'policy':bind(POLICY),'queries288':bind(QUERIES),'corpus_seal':{'path':str(V/'actual_corpus/CORPUS_SEAL.json'),'sha256':'ROOT_FILL_AFTER_COMPLETION'},'input_manifest':{'path':str(V/'actual_corpus/OCCURRENCES4896_INPUT_ONLY.jsonl'),'sha256':'ROOT_FILL_AFTER_COMPLETION'},'stored_crops':{'path':str(V/'actual_corpus/INPUT_CROPS4896_INPUT_ONLY.npz'),'sha256':'ROOT_FILL_AFTER_COMPLETION'},'weak_evidence':{'path':str(V/'actual_corpus/ORIGINAL_WEAK_AND_COMPOSITE_LABELS4896_GT_ONLY.jsonl'),'sha256':'ROOT_FILL_DESCRIPTOR_ONLY_NO_BODY_READ'}}
 write(H/'ROOT_ENTRY_TEMPLATE.json',t)
 print(json.dumps({'completed':True,'protocol':bind(PROTOCOL),'code':bind(__file__),'template':bind(H/'ROOT_ENTRY_TEMPLATE.json'),'command':p['runtime']['executable']+' '+str(Path(__file__).resolve())+' --entry /ROOT_EXACT_ENTRY.json','evaluation_command':'Same command plus --evaluate-saved AFTER completedPREDICTION_SEAL','actual_forward':0}),flush=True)

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--prepare',action='store_true');a.add_argument('--entry',type=Path);a.add_argument('--evaluate-saved',action='store_true');args=a.parse_args()
 if args.prepare:prepare()
 else:
  assert args.entry is not None
  if args.evaluate_saved:evaluate_saved(args.entry)
  else:run(args.entry)
