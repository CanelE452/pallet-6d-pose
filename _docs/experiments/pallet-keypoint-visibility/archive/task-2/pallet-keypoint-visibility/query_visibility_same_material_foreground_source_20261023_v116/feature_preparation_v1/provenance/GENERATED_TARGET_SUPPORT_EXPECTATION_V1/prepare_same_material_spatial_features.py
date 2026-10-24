"""C11-exact V116 source input features. No pixels/model/checkpoint I/O on import."""
import os,sys
os.environ['XFORMERS_DISABLED']='1';os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8');sys.dont_write_bytecode=True
from pathlib import Path
import argparse,ast,hashlib,importlib.util,json,time,traceback
import numpy as np

H=Path(__file__).resolve().parent;V=H.parent;P=V.parent;OUT=V/'spatial_features_v1'
C11=P/'query_visibility_spatial_target_query_features_20261023_v111'
SOURCE_PROTOCOL=C11/'feature_preparation/FEATURE_PROTOCOL_BEFORE_SOURCE_FORWARD.json'
SOURCE_SHA='e7f12ad8bf61994ec22fd510839d213564f7dd97933e6a89587f6f1063080645'
OPERATOR=C11/'feature_preparation/prepare_spatial_query_features.py'
OPERATOR_SHA='cbf2f1cad553673b727c39fbe5c9c4263dfbdac35dba0aeea086f956e74ff19e'
PROTOCOL=H/'FEATURE_PROTOCOL_BEFORE_ENCODER.json'
CORPUS=V/'corpus_build_v1/CORPUS_OUTPUT_SEAL.json';CORPUS_SHA='0ef41a9fd9d35471acb286efc044aa4831fb6d209d2d916cae3bc343aec1689b'
ADMISSION=V/'ROOT_ACTUAL_EXECUTION_AND_FIXED_VISUAL_DECISION.json'
CONTEXTS=V/'metadata_preparation/BEFORE28_C11_FEATURE_CONTEXT_READINESS_INPUT_ONLY.jsonl'


def require(ok,why):
 if not ok:raise ValueError(why)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def check(b):require(sha(b['path'])==b['sha256'],'Changed bound artifact '+b['path']);return Path(b['path'])
def read(p):return json.loads(Path(p).read_text())
def rows(p):return[json.loads(x)for x in Path(p).read_text().splitlines()if x]
def pixel(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def write(p,v):
 with Path(p).open('x')as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')
def jsonl(p,rs):
 with Path(p).open('x')as f:
  for r in rs:f.write(json.dumps(r,allow_nan=False,separators=(',',':'))+'\n')
def module(p,name):
 spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def pure_crop(path):
 fn=next(n for n in ast.parse(Path(path).read_text()).body if isinstance(n,ast.FunctionDef)and n.name=='crop');ns={'np':np}
 exec(compile(ast.Module(body=[fn],type_ignores=[]),str(path)+'::crop_AST_only','exec'),ns);return ns['crop']


def unit_plan(occ,contexts,expected_changed=124):
 """Base units0..27 plus one distinct unit per changed occurrence; no GT join."""
 require(len(occ)==924 and len(contexts)==28,'All924/28 retained')
 units=[];mapping=np.empty(924,np.int32)
 for i,ctx in enumerate(contexts):
  o=occ[33*i]
  require(o['occurrence_index']==33*i and o['base_index']==ctx['base_index']==i and o['variant']=='BEFORE'and not o['full_rgb_changed'],'Original BEFORE order')
  require(o['query_xy']==ctx['query_xy']and o['L']==ctx['L']and o['frozen_context_box_xyxy']==ctx['frozen_context_box_xyxy'],'Original q/L/context')
  units.append(dict(input_unit_index=i,kind='EXACT_C11_BEFORE_REUSE',occurrence_index=33*i,base_index=i,old_C11_unit_index=ctx['C11_spatial_feature_input_unit_index'],old_C11_occurrence_index=ctx['original_V96_C11_global_occurrence_index'],input_id=o['input_id'],native_slot_logical_only=o['native_slot_logical_only'],fullRGB=o['fullRGB'],format=o['format'],decoded_RGB_sha256=o['decoded_RGB_sha256'],raw_hw=o['raw_hw'],query_xy=o['query_xy'],L=o['L'],box_xyxy=o['frozen_context_box_xyxy'],rgb96_row=o['rgb96_row'],rgb96_sha256=o['rgb96_sha256'],valid96_sha256=o['crop_valid96_sha256'],raw_split=o['raw_split'],source=o['source']))
 for n,o in enumerate(occ):
  base=o['base_index'];b=units[base]
  require(o['occurrence_index']==n and base==n//33 and o['query_xy']==b['query_xy']and o['L']==b['L']and o['frozen_context_box_xyxy']==b['box_xyxy']and o['raw_split']==b['raw_split'],'All924 original input joins')
  if o['full_rgb_changed']:
   require(o['variant']in('OPAQUE_COVER','CUE_PRESERVING_SHAM'),'Changed after variant only')
   u=dict(b,input_unit_index=len(units),kind='FRESH_CHANGED_FULLRGB',occurrence_index=n,fullRGB=o['fullRGB'],format=o['format'],decoded_RGB_sha256=o['decoded_RGB_sha256'],rgb96_row=o['rgb96_row'],rgb96_sha256=o['rgb96_sha256'],valid96_sha256=o['crop_valid96_sha256']);units.append(u);mapping[n]=u['input_unit_index']
  else:
   require(o['fullRGB']==b['fullRGB']and o['decoded_RGB_sha256']==b['decoded_RGB_sha256']and o['rgb96_sha256']==b['rgb96_sha256']and o['crop_valid96_sha256']==b['valid96_sha256'],'Unchanged complete input must match BEFORE exactly')
   mapping[n]=base
 require(len(units)==28+expected_changed,'Exact124 changed units,28 reused')
 return units,mapping


def make_batch(units,images,rgbs,valids,producer,c8,c9,tm,encode,torch,device):
 """Changed RGB always supplies its OWN newly sampled target context."""
 N=len(units);require(0<N<=8,'At most8units/24views')
 target=[];target_valid=[];fra=[];c4=[];facts=[]
 for u,image,rgb,valid in zip(units,images,rgbs,valids):
  require(image.dtype==np.uint8 and image.shape==tuple(u['raw_hw'])+(3,)and pixel(image)==u['decoded_RGB_sha256'],'Changed native fullRGB identity')
  require(rgb.dtype==np.uint8 and rgb.shape==(3,96,96)and valid.dtype==bool and valid.shape==(1,96,96),'Exact saved local input types')
  require(pixel(rgb)==u['rgb96_sha256']and pixel(valid)==u['valid96_sha256'],'Exact stored RGB96/valid96 bytes')
  t,tv=tm.sample_target_context(image,u['box_xyxy']);target.append(t.transpose(2,0,1));target_valid.append(tv[None]);c4.append(bool(valid[:,47:49,47:49].all()))
  fra.append([np.float32(tv.mean()),np.float32(valid.mean()),np.float32(valid[:,32:64,32:64].mean())])
  facts.append(dict(input_unit_index=u['input_unit_index'],occurrence_index=u['occurrence_index'],base_index=u['base_index'],RGB96_sha256=pixel(rgb),valid96_sha256=pixel(valid),native_RGB_sha256=pixel(image),target_RGB224_F32_sha256=pixel(t),context_valid_fraction=float(np.float32(tv.mean())),local_valid_fraction=float(np.float32(valid.mean())),central_valid_fraction=float(np.float32(valid[:,32:64,32:64].mean())),RGB_central4_valid=c4[-1],fresh_context_from_changed_fullRGB=True))
 local,_=c8.preprocess_views(torch.as_tensor(np.stack(rgbs),device=device),torch.as_tensor(np.stack(valids),device=device))
 ctx=c9.preprocess_target(torch.as_tensor(np.stack(target),device=device),torch.as_tensor(np.stack(target_valid),device=device))
 views=torch.cat((ctx,local),0);require(tuple(views.shape)==(3*N,3,224,224),'Exact C11 target/full/central view ordering')
 raw=encode(views);cls=raw['x_norm_clstoken'];patch=raw['x_norm_patchtokens']
 with torch.inference_mode():x,inside=producer.assemble(cls[:N],patch[:N],cls[N:2*N],patch[N:2*N],cls[2*N:], [u['query_xy']for u in units],[u['box_xyxy']for u in units],fra,c4)
 return x.cpu().numpy(),np.array(inside,bool),np.asarray(fra,np.float32),np.array(c4,bool),facts


def prepare():
 require(sha(SOURCE_PROTOCOL)==SOURCE_SHA and sha(OPERATOR)==OPERATOR_SHA and sha(CORPUS)==CORPUS_SHA,'Pinned C11/source corpus')
 corpus=read(CORPUS);require(corpus['completed']and corpus['PASS']and not corpus['training_admission'],'Completed conditional image corpus')
 admission=read(ADMISSION);require(admission['conditional_AUX_admission']['admitted']is True and admission['conditional_AUX_admission']['new_fit_authorized_by_this_record']is False,'Root conditional source admission')
 occ=rows(check(corpus['occurrences']));contexts=rows(CONTEXTS);units,mapping=unit_plan(occ,contexts)
 jsonl(H/'UNIT152_INPUT_ONLY.jsonl',units)
 jsonl(H/'OCCURRENCE924_TO_UNIT_INPUT_ONLY.jsonl',[dict(occurrence_index=i,input_unit_index=int(u),occurrence_id=occ[i]['occurrence_id'],base_index=occ[i]['base_index'],query_xy=occ[i]['query_xy'],L=occ[i]['L'],raw_split=occ[i]['raw_split'])for i,u in enumerate(mapping)])
 source=read(SOURCE_PROTOCOL);old_receipt=C11/'spatial_query_features/FEATURE_COMPLETE.json';old=read(old_receipt)
 require(old['completed']and old['PASS']and old['features_shape']==[11652,1928]and old['occurrences']==41984,'Exact old C11 packet metadata')
 g=H/'GENERATED_FEATURE_TESTS.json';require(read(g)['PASS']and read(g)['code']==bind(__file__),'Exact generated checks')
 cropfile=next(b for b in corpus['files']if b['path'].endswith('/RGB96_AND_VALID924_INPUT_ONLY.npz'))
 p=dict(completed=True,code=bind(__file__),generated=bind(g),source_protocol=bind(SOURCE_PROTOCOL),source_operator=bind(OPERATOR),corpus_seal=bind(CORPUS),root_conditional_admission=bind(ADMISSION),input_occurrences=corpus['occurrences'],contexts=bind(CONTEXTS),unit_inputs=bind(H/'UNIT152_INPUT_ONLY.jsonl'),occurrence_map=bind(H/'OCCURRENCE924_TO_UNIT_INPUT_ONLY.jsonl'),RGB96_packet=cropfile,old_receipt=bind(old_receipt),old_output=old['output'],asset_runtime=source['asset_runtime'],weight=source['weight'],official_repo=source['official_repo'],official_revision=source['official_revision'],official_source_bindings=source['official_source_bindings'],C8_preprocess=source['C8_preprocess'],C9_target_preprocess=source['C9_target_preprocess'],target_crop_code=source['target_crop_code'],local_crop_code=source['local_crop_code'],runtime=source['runtime'],output=str(OUT),features_shape=[152,1928],occurrences=924,reused_BEFORE_units=28,fresh_changed_units=124,encoder_batch_units=8,max_views=24,expected_primary_encoder_batches=16,expected_first_repeat=1,feature_layout='Exact C11 T384/Q384/F384/C384/P384/+8',availability='Copy C11 factual availability for each original base. Fresh context geometry and unchanged valid96 checked; query=factual AND context. Qoutside never addsU.',training_allowed=False,current223=False,V112=False,V117=False,head_forward_fit_PnP=0)
 write(PROTOCOL,p)
 write(H/'ROOT_FEATURE_ENTRY_TEMPLATE.json',dict(approved=False,stage='V116_EXTRACT_FROZEN_C11_SOURCE_FEATURES',source_encoder_extraction_authorized=False,source_only=True,labels_GT_body_allowed=False,fit_allowed=False,current223=False,V112=False,V117=False,PnP=0,code=bind(__file__),protocol=bind(PROTOCOL),generated=bind(g),root_conditional_admission=bind(ADMISSION),corpus_seal=bind(CORPUS),output_directory=str(OUT)))
 print(json.dumps(dict(protocol=bind(PROTOCOL),template=bind(H/'ROOT_FEATURE_ENTRY_TEMPLATE.json'),units=len(units),occurrences=len(mapping),actual_encoder_calls=0)))


def authorize(path):
 e=read(path);require(e.get('approved')is True and e.get('stage')=='V116_EXTRACT_FROZEN_C11_SOURCE_FEATURES'and e.get('source_encoder_extraction_authorized')is True,'Explicit root encoder entry required')
 require(e['source_only']is True and e['labels_GT_body_allowed']is False and e['fit_allowed']is False and e['current223']is False and e['V112']is False and e['V117']is False and e['PnP']==0,'Source feature-only scope')
 for name in('code','protocol','generated','root_conditional_admission','corpus_seal'):check(e[name])
 require(e['code']==bind(__file__)and e['protocol']==bind(PROTOCOL)and e['corpus_seal']['sha256']==CORPUS_SHA,'Pinned code/protocol/corpus')
 p=read(PROTOCOL);require(p['code']==e['code']and p['generated']==e['generated']and p['root_conditional_admission']==e['root_conditional_admission']and p['corpus_seal']==e['corpus_seal'],'Entry metadata exact')
 require(read(e['generated']['path'])['PASS']and read(e['generated']['path'])['code']==e['code'],'Current generated proof')
 for name in('source_protocol','source_operator','input_occurrences','contexts','unit_inputs','occurrence_map','old_receipt','asset_runtime','C8_preprocess','C9_target_preprocess','target_crop_code','local_crop_code'):check(p[name])
 require(sha(SOURCE_PROTOCOL)==SOURCE_SHA and sha(OPERATOR)==OPERATOR_SHA,'Unchanged exact C11 operators')
 return e,p


def run(entry):
 e,p=authorize(entry);require(Path(e['output_directory']).resolve()==OUT and not OUT.exists(),'Fixed new output directory; no restart')
 units=rows(p['unit_inputs']['path']);occmap=rows(p['occurrence_map']['path']);occ=rows(p['input_occurrences']['path']);contexts=rows(p['contexts']['path'])
 expected_units,mapping=unit_plan(occ,contexts);require(expected_units==units and [r['input_unit_index']for r in occmap]==mapping.tolist(),'Prospective exact unit mapping')
 OUT.mkdir();write(OUT/'STARTED.json',dict(entry=bind(entry),no_automatic_restart=True,source_only=True,GT_labels_read=False))
 try:
  import torch
  require(torch.__version__==p['runtime']['torch']=='2.1.1+cu118'and torch.cuda.is_available()and Path(sys.executable).resolve()==Path(p['runtime']['executable']).resolve(),'Exact learned source runtime')
  torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
  for b in p['official_source_bindings']:check(b)
  check(p['weight']);check(p['old_output']);check(p['RGB96_packet'])
  producer=module(p['source_operator']['path'],'v116_exact_C11_math');c8=module(p['C8_preprocess']['path'],'v116_exact_C8_preprocess');c9=module(p['C9_target_preprocess']['path'],'v116_exact_C9_preprocess');tm=module(p['target_crop_code']['path'],'v116_exact_target_context');crop=pure_crop(p['local_crop_code']['path'])
  with np.load(p['old_output']['path'],allow_pickle=False)as z:old={name:np.array(z[name],copy=True)for name in('features','occurrence_unit','query_available','factual_query_available','context_available','unit_Q_inside')}
  with np.load(p['RGB96_packet']['path'],allow_pickle=False)as z:rgb96=z['rgb96'];valid96=z['crop_valid96']
  require(old['features'].shape==(11652,1928)and old['features'].dtype==np.float32 and rgb96.shape==(924,3,96,96)and valid96.shape==(924,1,96,96),'Bound feature/crop packet shapes')
  features=np.empty((152,1928),np.float32);inside=np.zeros(152,bool);fa=np.zeros(152,bool);ca=np.zeros(152,bool);facts=[]
  for u in units[:28]:
   i,oldunit,oldrow=u['input_unit_index'],u['old_C11_unit_index'],u['old_C11_occurrence_index']
   require(int(old['occurrence_unit'][oldrow])==oldunit,'Old row/unit identity')
   features[i]=old['features'][oldunit];inside[i]=old['unit_Q_inside'][oldunit];fa[i]=old['factual_query_available'][oldrow];ca[i]=old['context_available'][oldrow]
   require(pixel(rgb96[u['rgb96_row']])==u['rgb96_sha256']and pixel(valid96[u['rgb96_row']])==u['valid96_sha256'],'Exact BEFORE local bytes')
   facts.append(dict(input_unit_index=i,occurrence_index=u['occurrence_index'],kind=u['kind'],old_C11_unit_index=oldunit,feature_vector_sha256=pixel(features[i]),RGB96_sha256=u['rgb96_sha256'],valid96_sha256=u['valid96_sha256'],Q_inside=bool(inside[i]),factual_query_available=bool(fa[i]),context_available=bool(ca[i]),encoder_calls=0))
  del old
  sys.path.insert(0,p['official_repo']);from dinov2.hub.backbones import dinov2_vits14_reg
  encoder=dinov2_vits14_reg(pretrained=False);loaded=encoder.load_state_dict(torch.load(p['weight']['path'],map_location='cpu',weights_only=True),strict=True);require(not loaded.missing_keys and not loaded.unexpected_keys,'Strict official weight state')
  encoder=encoder.cuda().float().eval().requires_grad_(False);before=producer.state_sha(encoder);require(before==read(p['asset_runtime']['path'])['state_before_SHA256'],'Exact frozen DINO state')
  calls=0;repeated=False;start=time.monotonic()
  def encode(views):
   nonlocal calls,repeated
   with torch.inference_mode():
    value=encoder.forward_features(views);calls+=1
    if not repeated:
     again=encoder.forward_features(views);calls+=1;require(all(torch.equal(value[k],again[k])for k in('x_norm_clstoken','x_norm_patchtokens')),'First changed batch repeat exact');repeated=True
   return value
  for start_index in range(28,152,8):
   batch=units[start_index:start_index+8];images=[];rs=[];vs=[]
   from PIL import Image
   for u in batch:
    rgb=np.array(Image.open(check(u['fullRGB'])).convert('RGB'),dtype=np.uint8,copy=True);images.append(rgb);r=rgb96[u['rgb96_row']];v=valid96[u['rgb96_row']]
    recrop,revalid=crop(rgb,u['query_xy'],u['L'],96);require(np.array_equal(r,recrop)and np.array_equal(v,revalid),'Exact changed nativeRGB and localRGB96 matching')
    rs.append(r);vs.append(v)
   x,ii,frac,c4,fc=make_batch(batch,images,rs,vs,producer,c8,c9,tm,encode,torch,'cuda')
   N=len(batch);features[start_index:start_index+N]=x;inside[start_index:start_index+N]=ii
   for j,u in enumerate(batch):
    i=u['input_unit_index'];base=u['base_index'];fa[i]=fa[base];ca[i]=frac[j,0]>0
    require(c4[j]==bool(valid96[units[base]['rgb96_row'],:,47:49,47:49].all()),'Unchanged original central4 validity')
    require(ca[i]==ca[base],'Unchanged original context geometry availability')
    fc[j].update(kind=u['kind'],Q_inside=bool(ii[j]),factual_query_available=bool(fa[i]),context_available=bool(ca[i]));facts.append(fc[j])
   print(json.dumps(dict(changed_units_done=start_index+N-28,total=124,encoder_calls=calls,seconds=time.monotonic()-start)),flush=True)
  require(features.shape==(152,1928)and np.isfinite(features).all()and np.array_equal(features[:,1927],inside.astype(np.float32)),'All152 C11 exact schema')
  after=producer.state_sha(encoder);require(before==after and repeated and calls==17,'Frozen state/16primary+1repeat batches')
  require(all(not t.requires_grad for t in encoder.parameters())and not encoder.training,'Frozen eval encoder')
  packet=OUT/'SOURCE_SPATIAL_QUERY_FEATURES.npz';np.savez_compressed(packet,features=features,occurrence_unit=mapping,unit_Q_inside=inside,query_available=(fa&ca)[mapping],factual_query_available=fa[mapping],context_available=ca[mapping],global_occurrence_index=np.arange(924,dtype=np.int32))
  jsonl(OUT/'INPUT_RGB_SPATIAL_FACTS152_INPUT_ONLY.jsonl',facts)
  manifest=dict(completed=True,output=bind(packet),units=p['unit_inputs'],occurrence_map=p['occurrence_map'],input_occurrences=p['input_occurrences'],old_C11_features=p['old_output'],reused_units28=28,fresh_units124=124,features_shape=[152,1928],occurrences=924,feature_layout=p['feature_layout'],GT_labels_in_input_packet=False,entry=bind(entry))
  write(OUT/'PACK_MANIFEST.json',manifest);check(p['weight']);check(p['old_output']);check(p['RGB96_packet'])
  write(OUT/'FEATURE_COMPLETE.json',dict(completed=True,PASS=True,output=bind(packet),manifest=bind(OUT/'PACK_MANIFEST.json'),protocol=e['protocol'],code=e['code'],entry=bind(entry),root_conditional_admission=p['root_conditional_admission'],corpus_seal=p['corpus_seal'],features_shape=[152,1928],occurrences=924,old_C11_feature_units_exactly_reused=28,changed_fullRGB_units_freshly_encoded=124,input_RGB_spatial_facts=bind(OUT/'INPUT_RGB_SPATIAL_FACTS152_INPUT_ONLY.jsonl'),model_state_unchanged=True,frozen_encoder_state_before=before,frozen_encoder_state_after=after,first_batch_repeat_exact=repeated,encoder_calls=calls,max_views=24,Q_outside_never_adds_U=True,available=int((fa&ca)[mapping].sum()),factual_U=int((~fa)[mapping].sum()),context_U=int((~ca)[mapping].sum()),labels_GT_mask_pose_depth_private_V112_V117_read=0,head_forward_fit_PnP=0,seconds=time.monotonic()-start))
  print(json.dumps(dict(completed=True,receipt=bind(OUT/'FEATURE_COMPLETE.json'))))
 except BaseException:
  write(OUT/'FAILED_EXECUTION.json',dict(completed=False,traceback=traceback.format_exc(),no_automatic_restart=True));raise


if __name__=='__main__':
 ap=argparse.ArgumentParser();g=ap.add_mutually_exclusive_group(required=True);g.add_argument('--prepare',action='store_true');g.add_argument('--entry');a=ap.parse_args()
 if a.prepare:prepare()
 else:run(a.entry)
