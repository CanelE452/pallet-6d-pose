"""Inert C9 target-relative feature producer; actual encoder requires root entry."""
import os,sys
sys.dont_write_bytecode=True
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import argparse,hashlib,importlib.util,json,time,traceback
import numpy as np
import torch
from torch.nn import functional as F
H=Path(__file__).resolve().parent;V=H.parent;P=V.parent
C8=P/'query_visibility_frozen_rgb_head_20261023_v105'
V107=P/'query_visibility_target_context_rgb_20261023_v107'
C8CODE=C8/'frozen_rgb_feature_preparation/prepare_frozen_rgb_features.py'
MATH=V107/'target_context_math.py'
OLD=P/'query_visibility_query_anchored_cue_20261023_v96/actual_source_pair_input_pack/PACK_MANIFEST.json'
NEW=P/'query_visibility_diverse_expected_cue_source_20261023_v99/matching_feature_pack/PACK_MANIFEST.json'
OUT=V/'target_context_features';PROTOCOL=H/'FEATURE_PROTOCOL_BEFORE_SOURCE_FORWARD.json';BATCH=32
MEAN=(.485,.456,.406);STD=(.229,.224,.225)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for x in iter(lambda:f.read(1<<20),b''):h.update(x)
 return h.hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def check(b):assert sha(b['path'])==b['sha256'],b['path'];return Path(b['path'])
def read(p):return json.loads(Path(p).read_text())
def rows(p):return [json.loads(x)for x in Path(p).read_text().splitlines()if x]
def write(p,r):
 with Path(p).open('x')as f:json.dump(r,f,indent=2,allow_nan=False);f.write('\n')
def module(p,name):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def preprocess_target(rgb224,valid224):
 if rgb224.dtype!=torch.float32 or rgb224.ndim!=4 or tuple(rgb224.shape[1:])!=(3,224,224):raise ValueError('F32 native RGB224 in0..255 required')
 if valid224.dtype!=torch.bool or tuple(valid224.shape)!=(len(rgb224),1,224,224):raise ValueError('bool targetvalid required')
 if not torch.isfinite(rgb224).all()or not((rgb224>=0)&(rgb224<=255)).all():raise ValueError('Finite RGB0..255 required')
 x=rgb224/255.;v=valid224.float();mean=x.new_tensor(MEAN)[None,:,None,None];std=x.new_tensor(STD)[None,:,None,None]
 return(x*v+mean*(1-v)-mean)/std

def assemble(target_raw512,c8_features,target_fraction,context_available):
 n=len(c8_features)
 if c8_features.shape!=(n,1026)or target_raw512.shape!=(n,512)or target_fraction.shape!=(n,)or context_available.shape!=(n,):raise ValueError('C9 schema')
 if any(not torch.isfinite(x).all()for x in(target_raw512,c8_features,target_fraction)):raise ValueError('Nonfinite')
 if not((target_fraction>=0)&(target_fraction<=1)).all():raise ValueError('Fraction range')
 present=context_available.bool()&(target_fraction>0)
 target=F.normalize(target_raw512.float(),p=2,dim=1,eps=1e-12)
 target=torch.where(present[:,None],target,torch.zeros_like(target))
 centre=c8_features[:,512:1024] # literal exact C8 values; do not normalize again
 centre_fraction=c8_features[:,1025]
 fraction=torch.where(present,target_fraction.float(),torch.zeros_like(target_fraction))
 cosine=F.cosine_similarity(target,centre,dim=1,eps=1e-12)
 out=torch.cat((target,centre,cosine[:,None],fraction[:,None],centre_fraction[:,None]),1)
 assert out.shape==(n,1027)and torch.isfinite(out).all()
 assert torch.equal(out[:,512:1024],centre)and torch.equal(out[:,1026],centre_fraction)
 return out

def prepare(design):
 import torchvision
 assert torch.__version__=='2.1.1+cu118'and torchvision.__version__=='0.16.1+cu118'
 generated=H/'GENERATED_CONTEXT_FEATURE_TESTS.json';g=read(generated);assert g['PASS']and g['code']==bind(__file__)
 math=module(MATH,'c9_native_crop_math')
 oldm=read(OLD);newm=read(NEW);oldline=rows(check(oldm['input_lineage']));newline=rows(check(newm['input_lineage']))
 units=np.array([r['input_unit_index']for r in oldline]+[1348+r['input_unit_index']for r in newline],np.int32)
 factual=np.array([r['input_facts']['operator_query_available']for r in oldline+newline],bool)
 assert units.shape==factual.shape==(33984,)and units.min()==0 and units.max()==3651
 definitions={d['context_id']:d for d in rows(V107/'TARGET_CONTEXT_DEFINITIONS216_INPUT_ONLY.jsonl')};occ=rows(V107/'OCCURRENCES33984_TARGET_CONTEXT_INPUT_ONLY.jsonl');assert len(occ)==33984
 context=[None]*3652
 for i,(r,u)in enumerate(zip(occ,units)):
  d=definitions[r['context_id']];assert r['global_occurrence_index']==i and r['operator_query_available']==bool(factual[i])
  a=dict(input_unit_index=int(u),full_RGB=r['full_RGB'],full_RGB_format=r['full_RGB_format'],decoded_RGB_sha256=r['decoded_RGB_sha256'],raw_hw=r['raw_hw'],context_id=r['context_id'],box_xyxy=d['target_box_xyxy'],query_xy=r['query_xy'],L=r['L'])
  if context[u]is None:context[u]=a
  else:assert context[u]==a,('Inconsistent context for reused unit',int(u),i)
 assert all(x is not None for x in context)
 available=[];box_present=[];fractions=[]
 for r in context:
  b=r['box_xyxy'];present=b is not None;box_present.append(present)
  if not present:available.append(False);fractions.append(0.);continue
  # Geometry-only availability: same positive bilinear-neighbor rule as V107 sampler.
  grid=math.target_context_grid(b);x=grid[...,0];y=grid[...,1];ix=np.floor(x);iy=np.floor(y);dx=x-ix;dy=y-iy;Hn,Wn=r['raw_hw'];valid=np.ones(x.shape,bool)
  for ox,oy,w in((0,0,(1-dx)*(1-dy)),(1,0,dx*(1-dy)),(0,1,(1-dx)*dy),(1,1,dx*dy)):
   valid &= ((ix+ox>=0)&(ix+ox<Wn)&(iy+oy>=0)&(iy+oy<Hn))|(w==0)
  available.append(bool(valid.any()));fractions.append(float(valid.mean()))
 available=np.array(available,bool);box_present=np.array(box_present,bool);fractions=np.array(fractions,np.float32)
 unitfile=H/'UNIT_CONTEXTS3652_INPUT_ONLY.jsonl'
 with unitfile.open('x')as f:
  for r in context:f.write(json.dumps(r,separators=(',',':'),allow_nan=False)+'\n')
 np.savez_compressed(H/'OCCURRENCE_MAPPING_INPUT_ONLY.npz',occurrence_unit=units,factual_query_available=factual,context_available=available[units],unit_context_available=available,unit_box_present=box_present,unit_target_valid_fraction=fractions,global_occurrence_index=np.arange(33984,dtype=np.int32))
 c8receipt=read(C8/'frozen_rgb_features/FEATURE_COMPLETE.json');assert c8receipt['completed']and c8receipt['PASS']
 c8protocol=read(C8/'frozen_rgb_feature_preparation/FEATURE_PROTOCOL_BEFORE_SOURCE_FORWARD.json')
 assert c8receipt['code']==bind(C8CODE)
 tv=Path(torchvision.__file__).parent
 protocol=dict(completed=True,design=bind(design),code=bind(__file__),generated=bind(generated),crop_math=bind(MATH),V107_inventory=bind(V107/'RESULT.json'),V107_occurrences=bind(V107/'OCCURRENCES33984_TARGET_CONTEXT_INPUT_ONLY.jsonl'),V107_definitions=bind(V107/'TARGET_CONTEXT_DEFINITIONS216_INPUT_ONLY.jsonl'),old_manifest=bind(OLD),new_manifest=bind(NEW),unit_contexts=bind(unitfile),occurrence_mapping=bind(H/'OCCURRENCE_MAPPING_INPUT_ONLY.npz'),C8_feature_receipt=bind(C8/'frozen_rgb_features/FEATURE_COMPLETE.json'),C8_feature_pack=c8receipt['output'],C8_feature_factory=bind(C8CODE),pretrained=c8protocol['pretrained'],weight_provenance=c8protocol['weight_download_provenance'],torchvision_resnet_code=bind(tv/'models/resnet.py'),runtime=dict(python=str(Path(sys.executable).resolve()),torch=torch.__version__,torchvision=torchvision.__version__,numpy=np.__version__,TF32=False,AMP=False,deterministic=True,threads=1),schema=dict(features='float32[3652,1027]',occurrences=33984,units=3652,layout=['target512','exact_C8_central512','cosine1','target_valid_fraction1','exact_C8_central_fraction1'],cosine='F.cosine_similarity(target_L2,C8central,dim1,eps1e-12)',target_preprocess='V107 native bilinear float32 RGB224 (no rounding), invalid sample ImageNet mean fill then mean/std; no extra resizing',availability='Immutable factual query availability AND original finite box with at least one fully supported native bilinear224 sample; partial retained. Separate box-presence retained.',missing='Zero target descriptor/cosine/targetfraction; C8central preserved; final query U.'),counts=dict(missing_context_units=int((~available).sum()),missing_box_units=int((~box_present).sum()),missing_context_occurrences=int((~available[units]).sum()),factual_U=int((~factual).sum()),final_U=int((~(factual&available[units])).sum())),output=str(OUT),batch_units=BATCH,no_GT_or_native_or_labels_body=True,model_inputs=['target_RGB224_and_valid','exact_saved_C8central_features'],no_C8full96_or_C1prior=True,preparation_encoder_forward_or_parameter_load=0)
 write(PROTOCOL,protocol);write(H/'ROOT_FEATURE_ENTRY_TEMPLATE.json',dict(approved_actual_source_feature_extraction=False,source_only=True,current_allowed=False,fit_allowed=False,PnP_allowed=False,code=bind(__file__),protocol=bind(PROTOCOL),design=bind(design),generated=bind(generated),output=str(OUT)))
 print(json.dumps(dict(protocol=bind(PROTOCOL),entry_template=bind(H/'ROOT_FEATURE_ENTRY_TEMPLATE.json'),counts=protocol['counts'],command=f'{sys.executable} {Path(__file__).resolve()} --entry ROOT_APPROVED_ENTRY.json')),flush=True)

def run(entry):
 e=read(entry);assert e['approved_actual_source_feature_extraction']is True and e['source_only']and e['current_allowed']is False and e['fit_allowed']is False and e['PnP_allowed']is False
 for k in('code','protocol','design','generated'):check(e[k])
 assert e['code']==bind(__file__)and e['protocol']==bind(PROTOCOL)
 p=read(PROTOCOL);assert p['code']==e['code']and p['design']==e['design']and p['generated']==e['generated']and read(check(e['generated']))['PASS']
 assert not OUT.exists()and Path(e['output']).resolve()==OUT.resolve()
 import torchvision
 from PIL import Image,__version__ as PILVERSION
 assert str(Path(sys.executable).resolve())==p['runtime']['python']and torch.__version__==p['runtime']['torch']and torchvision.__version__==p['runtime']['torchvision']and torch.cuda.is_available()
 for k in('crop_math','unit_contexts','occurrence_mapping','C8_feature_receipt','C8_feature_pack','C8_feature_factory','pretrained','weight_provenance','torchvision_resnet_code','V107_inventory','V107_occurrences','V107_definitions','old_manifest','new_manifest'):check(p[k])
 assert read(check(p['C8_feature_receipt']))['output']==p['C8_feature_pack']
 with np.load(p['occurrence_mapping']['path'],allow_pickle=False)as f:meta={k:np.array(f[k],copy=True)for k in f.files}
 with np.load(p['C8_feature_pack']['path'],allow_pickle=False)as f:
  c8=np.array(f['features'],copy=True);cu=np.array(f['occurrence_unit'],copy=True);ca=np.array(f['query_available'],copy=True)
 assert c8.dtype==np.float32 and c8.shape==(3652,1026)and np.array_equal(cu,meta['occurrence_unit'])and np.array_equal(ca,meta['factual_query_available'])
 contexts=rows(p['unit_contexts']['path']);math=module(p['crop_math']['path'],'c9_crop_exact');factory=module(p['C8_feature_factory']['path'],'c9_exact_C8_factory')
 OUT.mkdir(parents=True);write(OUT/'STARTED.json',dict(entry=bind(entry),source_only=True,encoder_calls_before_this_run=0,no_automatic_restart=True));started=time.monotonic()
 try:
  torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
  encoder=factory.load_imagenet_backbone().cuda().eval().requires_grad_(False);before=factory.state_sha(encoder);torch.cuda.reset_peak_memory_stats();chunks=[];decodefacts=[];repeat=False;calls=0
  for start in range(0,3652,BATCH):
   end=min(start+BATCH,3652);rgbs=[];valids=[]
   for u in range(start,end):
    r=contexts[u];assert r['input_unit_index']==u
    if r['box_xyxy']is None:
     rgbs.append(np.zeros((3,224,224),np.float32));valids.append(np.zeros((1,224,224),bool));continue
    path=check(r['full_RGB'])
    if r['full_RGB_format']=='CANONICAL_RGB_UINT8_HWC_NPY':im=np.load(path,allow_pickle=False);decoder='CANONICAL_NPY'
    else:im=np.array(Image.open(path).convert('RGB'),dtype=np.uint8,copy=True);decoder='Pillow.convertRGB'
    assert im.dtype==np.uint8 and list(im.shape)==r['raw_hw']+[3]
    decoded=hashlib.sha256(im.tobytes()).hexdigest();assert decoded==r['decoded_RGB_sha256'],(u,decoded,r['decoded_RGB_sha256'])
    crop,valid=math.sample_target_context(im,r['box_xyxy']);assert np.float32(valid.mean())==meta['unit_target_valid_fraction'][u]
    rgbs.append(crop.transpose(2,0,1));valids.append(valid[None]);decodefacts.append(dict(input_unit_index=u,file=r['full_RGB'],decoded_RGB_sha256=decoded,decoder=decoder,target_RGB224_float32_sha256=hashlib.sha256(np.ascontiguousarray(crop).tobytes()).hexdigest()))
   x=torch.as_tensor(np.stack(rgbs),device='cuda');v=torch.as_tensor(np.stack(valids),device='cuda');norm=preprocess_target(x,v)
   with torch.inference_mode():
    raw=encoder(norm);calls+=1
    if start==0:assert torch.equal(raw,encoder(norm));calls+=1;repeat=True
    out=assemble(raw,torch.as_tensor(c8[start:end],device='cuda'),v.float().mean((1,2,3)),torch.as_tensor(meta['unit_context_available'][start:end],device='cuda'))
   chunks.append(out.cpu().numpy())
   if(start//BATCH+1)%20==0:print(json.dumps(dict(units=end,total=3652,seconds=time.monotonic()-started)),flush=True)
  features=np.concatenate(chunks).astype(np.float32);assert features.shape==(3652,1027)and np.isfinite(features).all()
  assert np.array_equal(features[:,512:1024],c8[:,512:1024])and np.array_equal(features[:,1026],c8[:,1025])and np.array_equal(features[:,1025],meta['unit_target_valid_fraction'])
  assert np.all(features[~meta['unit_context_available'],:512]==0)and np.all(features[~meta['unit_context_available'],1024:1026]==0)
  after=factory.state_sha(encoder);assert before==after and repeat and not any(m.training for m in encoder.modules())and all(not v.requires_grad for v in encoder.parameters())
  available=meta['factual_query_available']&meta['context_available']
  output=OUT/'SOURCE_TARGET_CONTEXT_FEATURES.npz'
  np.savez_compressed(output,features=features,occurrence_unit=meta['occurrence_unit'],query_available=available,factual_query_available=meta['factual_query_available'],context_available=meta['context_available'],unit_context_available=meta['unit_context_available'],unit_box_present=meta['unit_box_present'],global_occurrence_index=meta['global_occurrence_index'])
  facts=OUT/'CANONICAL_RGB_AND_TARGET_CROP_CERTIFICATES_INPUT_ONLY.jsonl'
  with facts.open('x')as f:
   for r in decodefacts:f.write(json.dumps(r,separators=(',',':'))+'\n')
  write(OUT/'FEATURE_COMPLETE.json',dict(completed=True,PASS=True,output=bind(output),protocol=e['protocol'],entry=bind(entry),code=e['code'],design=e['design'],canonical_RGB_certificates=bind(facts),C8_feature_receipt=p['C8_feature_receipt'],C8_feature_pack=p['C8_feature_pack'],pretrained=p['pretrained'],features_shape=[3652,1027],features_dtype='float32',occurrences=33984,context_consistent_per_existing_unit=True,exact_C8central_and_fraction_copied=True,C8full96_used=False,C1_prior_used=False,counts=dict(**p['counts'],final_query_available=int(available.sum())),first_batch_repeat_exact=True,model_state_unchanged=True,frozen_encoder_state_before=before,frozen_encoder_state_after=after,encoder_forward_batch_calls=calls,seconds=time.monotonic()-started,peak_GPU_bytes=torch.cuda.max_memory_allocated(),decoder=dict(Pillow=PILVERSION,numpy=np.__version__),GT_or_native_or_label_body_read=False,new_head_fit_private_current_PnP=0))
  print(json.dumps(dict(completed=True,receipt=bind(OUT/'FEATURE_COMPLETE.json'),seconds=time.monotonic()-started)),flush=True)
 except BaseException:
  write(OUT/'FAILED_EXECUTION.json',dict(completed=False,traceback=traceback.format_exc(),no_automatic_restart=True));raise

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');ap.add_argument('--design',type=Path);ap.add_argument('--entry',type=Path);a=ap.parse_args()
 if a.prepare:
  assert a.design is not None;prepare(a.design)
 elif a.entry:run(a.entry)
 else:raise SystemExit('Use --prepare --design PATH or --entry ROOT_APPROVED_ENTRY')
