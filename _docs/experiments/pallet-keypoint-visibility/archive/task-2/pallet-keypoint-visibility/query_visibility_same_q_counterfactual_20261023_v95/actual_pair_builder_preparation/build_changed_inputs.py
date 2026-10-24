"""V95 sealed feasibility -> unchanged base refs + fresh changed RGB/P3 inputs.
No reconstruction/search of feasibility, detector query reselection, or head fit.
Actual execution requires root entry; generated tests inject arrays and stubs.
"""
from pathlib import Path
from collections import Counter
import argparse,hashlib,importlib.util,json,os,sys,time
import numpy as np

HERE=Path(__file__).resolve().parent
V95=HERE.parent;P=V95.parent
V94=P/'query_visibility_matched_query_scalar_20261023_v94'
V3=V94/'paired_source_preparation/v3_integrated_same_q_builder/integrated_same_q_v3.py'
V3_SHA='ce40397c0bfd695aee4ee07dde425711c92e0c0f37fb854f8ffe02fd396517fb'
MATH=P/'query_visibility_public_cue_inputs_20261023_v90/producer/spatial_input_math.py'
MATH_SHA='6a6c2543e28715e3d47e15c543c25f75ff7d3efa3b881e523bd3451817fc004b'
AUTH=P/'query_visibility_controlled_real_20261022_v66/pallox_features/FEATURE_PROTOCOL.json'
SCOPE=V95/'ROOT_SOURCE_SCOPE_BEFORE_ACTUAL_CONSTRUCTION.json'
PROTOCOL=HERE/'FEATURE_PROTOCOL_BEFORE_SOURCE_EXECUTION.json'
GENERATED=HERE/'GENERATED_DRIVER_TESTS.json'
OUT=V95/'actual_source_pair_input_pack'
FIELDS=('p3_roi','rgb96','crop_valid96')
VARIANTS=('BEFORE','OPAQUE_COVER','CUE_PRESERVING_SHAM')
BACK_SHA='296d81b2adde5ce74192f0012e554ae546c95f309a82dd808024ee34105721ab'
WEIGHT_SHA='970a0913b38ed4c9e3662837abccbf9d91b8b0858deafae854c1055e477644f7'


def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb')as f:
  for a in iter(lambda:f.read(1<<20),b''):h.update(a)
 return h.hexdigest()
def bind(p):return {'path':str(Path(p).resolve()),'sha256':sha(p)}
def check(b):
 if sha(b['path'])!=b['sha256']:raise ValueError('Changed binding '+b['path'])
 return Path(b['path'])
def contains(t,b):
 if isinstance(t,dict):return (t.get('path')==b.get('path')and t.get('sha256')==b.get('sha256'))or any(contains(x,b)for x in t.values())
 if isinstance(t,list):return any(contains(x,b)for x in t)
 return False
def read(p):return json.loads(Path(p).read_text())
def lines(p):return [json.loads(s)for s in Path(p).read_text().splitlines()if s]
def write(p,d):
 with Path(p).open('x')as f:json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def array_sha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def module(p,name):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m

def helpers():
 assert sha(V3)==V3_SHA and sha(MATH)==MATH_SHA
 return module(V3,'v95_bound_v3'),module(MATH,'v95_exact_input_math')

def input_facts(x):
 assert set(x)==set(FIELDS)
 assert x['p3_roi'].shape==(65,9,9)and x['p3_roi'].dtype.kind=='f'and np.isfinite(x['p3_roi']).all()
 assert x['rgb96'].shape==(3,96,96)and x['rgb96'].dtype==np.uint8
 assert x['crop_valid96'].shape==(1,96,96)and x['crop_valid96'].dtype==bool
 v=x['p3_roi'][64];assert np.isin(v,[0,1]).all()
 assert not x['p3_roi'][:64][:,~v.astype(bool)].any()
 return {'ROI_anyvalid':bool(v.any()),'ROI_center_valid':bool(v[4,4]),
         'RGB_central4_valid':bool(x['crop_valid96'][0,47:49,47:49].all()),
         'operator_query_available':bool(v[4,4]and x['crop_valid96'][0,47:49,47:49].all())}

def same_q(row,base):
 assert base['query_xy']==row['query_xy']and base['L']==row['L']
 assert base['image_sha256']==row['image_sha256']


def materialize_one(row,occurrence,base_tensors,base_pixel_sha,load_changed,extract_changed,math):
 """Only edited fullRGB reaches extractor; it receives RGB and old q/L only."""
 same=occurrence['full_rgb_decoded_sha256']==base_pixel_sha
 assert bool(occurrence['full_rgb_changed'])==(not same)
 if same:
  # All BEFORE/U/failure unchanged occurrences reuse the exact bound triplet.
  return base_tensors,False,dict(reused_before_tensors=True,new_feature_forward=0,**input_facts(base_tensors))
 rgb=load_changed(occurrence)
 assert rgb.dtype==np.uint8 and rgb.shape==tuple(row['raw_hw'])+(3,)
 assert array_sha(rgb)==occurrence['full_rgb_decoded_sha256']
 q=row['query_xy'];L=row['L']
 if q is None or L is None:
  # Absent q only: explicit placeholder. Never zero a finite outside-centre q.
  x={'p3_roi':np.zeros((65,9,9),np.float16),'rgb96':np.zeros((3,96,96),np.uint8),'crop_valid96':np.zeros((1,96,96),bool)}
  return x,True,dict(reused_before_tensors=False,new_feature_forward=0,noquery_placeholder=True,**input_facts(x))
 assert np.isfinite(q).all()and np.isfinite(L)and L>0
 roi,feature_provenance=extract_changed(rgb,np.asarray(q,np.float64),float(L))
 crop,valid=math.crop(rgb,np.asarray(q,np.float64),float(L),96)
 x={'p3_roi':roi,'rgb96':crop,'crop_valid96':valid}
 return x,True,dict(reused_before_tensors=False,new_feature_forward=1,feature_provenance=feature_provenance,**input_facts(x))


def validate_metadata(rows,lineage,occurrences,construction,expected=960):
 v3,_=helpers();assert len(rows)==len(lineage)==len(construction)==expected
 assert len(occurrences)==3*expected
 for i,(r,l,g)in enumerate(zip(rows,lineage,construction)):
  assert r['slot_index']==l['slot_index']==g['base_row_index']==i;v3.checked_row(r)
  assert l['input_id']==r['input_id']and l['native_slot_logical_only']==r['native_slot_logical_only']
  assert l['query_xy']==r['query_xy']and l['L']==r['L']and l['image_sha256']==r['image_sha256']
  assert g['original_natural_label_GT_only']in(-1,0,1)
  assert g['before_aux_state']in('D','U')and(g['before_aux_state']=='D')==(g['original_natural_label_GT_only']==0)
  assert set(l['before_input_refs'])==set(FIELDS)
  for j,variant in enumerate(VARIANTS):
   o=occurrences[3*i+j];assert o['occurrence_index']==3*i+j and o['slot_index']==i and o['variant']==variant
   assert o['occurrence_id']==f"{r['input_id']}::slot{r['native_slot_logical_only']}::{variant}"
   if variant=='BEFORE':assert not o['full_rgb_changed']
   assert (not o['full_rgb_changed'])==(o['full_rgb_decoded_sha256']==l['decoded_RGB_sha256'])
 # No filtering: 960x3 includes every unknown/unavailable/failed edit.
 return {'original_rows':expected,'occurrences':3*expected,'natural_before_counts':dict(Counter(g['original_natural_label_GT_only']for g in construction)),
         'controlled_D_eligible':sum(g['before_aux_state']=='D'for g in construction)}


class RefReader:
 """Small array-member cache. No labels/masks/reference fields are accepted."""
 def __init__(self):self.cache={};self.checked=set()
 def get(self,ref):
  d=ref['file'];key=(d['path'],d['sha256'],ref['key'])
  if (d['path'],d['sha256'])not in self.checked:check(d);self.checked.add((d['path'],d['sha256']))
  if key not in self.cache:
   if len(self.cache)>8:self.cache.clear()
   with np.load(d['path'],allow_pickle=False)as z:self.cache[key]=z[ref['key']]
  return np.array(self.cache[key][ref['index']],copy=True)
 def triplet(self,line):return {k:self.get(line['before_input_refs'][k])for k in FIELDS}


def state_sha(model):
 h=hashlib.sha256()
 for k,v in sorted(model.state_dict().items()):
  a=v.detach().cpu().contiguous().numpy();h.update(k.encode()+b'\0'+str(a.dtype).encode()+b'\0'+str(a.shape).encode()+b'\0'+a.tobytes())
 return h.hexdigest()


def authorize(path):
 e=read(path);assert e['approved_by_root']is True and e['source_only']is True
 assert e['current_allowed']is False and e['training_allowed']is False and e['PnP_allowed']is False
 for k in('code','protocol','generated','independent_review','intake_seal'):check(e[k])
 assert e['code']==bind(__file__)and Path(e['protocol']['path'])==PROTOCOL
 p=read(PROTOCOL);proof=read(e['generated']['path']);assert proof['PASS']and proof['code']==bind(__file__)and read(e['independent_review']['path'])['PASS']
 intake=read(e['intake_seal']['path']);assert intake['completed']and intake['PASS']
 sealed_metadata=[intake]
 if 'metadata_lock'in intake:
  md=read(check(intake['metadata_lock']));assert md['completed']and md['PASS'];sealed_metadata.append(md)
 for d in p['code_bindings'].values():check(d)
 for k in('input_rows','source_lineage','occurrences','construction_ledger'):
  check(e[k]);assert any(contains(t,e[k])for t in sealed_metadata)
 for d in p['assets'].values():check(d)
 return e,p


def run(entry_path):
 e,p=authorize(entry_path)  # no source pixels/array/model parameters before root guard
 import torch,cv2
 from PIL import Image
 assert torch.__version__=='2.1.1+cu118'and cv2.__version__=='4.9.0'
 assert Path(sys.executable).resolve()==Path('/home/minjae/anaconda3/envs/pallet-yolo26/bin/python').resolve()
 v3,math=helpers();rows=lines(e['input_rows']['path']);source=lines(e['source_lineage']['path']);occs=lines(e['occurrences']['path']);gt=lines(e['construction_ledger']['path'])
 summary=validate_metadata(rows,source,occs,gt);assert summary['controlled_D_eligible']==143
 assert summary['natural_before_counts']=={-1:797,0:143,1:20}
 out=Path(e['output_directory']);out.mkdir(exist_ok=False);(out/'chunks').mkdir()
 write(out/'STARTED.json',dict(entry=bind(entry_path),code=bind(__file__),source_rows=960,occurrences=2880,new_current_training_PnP=0))
 torch.set_num_threads(1);cv2.setNumThreads(1);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 os.environ['YOLO_CONFIG_DIR']=str(HERE/'runtime/ultralytics')
 back=None;backmod=None;before_state=None;forwards=0;repeat=0;warmup=0;first=True;cache=RefReader();buffer=[];chunks=[];output=[];counts=Counter();start=time.monotonic()
 def extract(rgb,q,L):
  nonlocal back,backmod,before_state,forwards,repeat,warmup,first
  if back is None:
   backmod=module(p['assets']['backbone_code']['path'],'v95_frozen_yolo')
   back=backmod.FrozenYoloFeatures(p['assets']['backbone_checkpoint']['path'],device='cuda:0')
   with torch.inference_mode():back.predict(np.zeros((480,640,3),np.uint8),already_padded=False,cpu_features=True)
   warmup=1;before_state=state_sha(back.yolo.model)
  with torch.inference_mode():z=back.predict(np.ascontiguousarray(rgb[...,::-1]),already_padded=False,cpu_features=True)
  forwards+=1
  affine=math.expected_affine(rgb.shape[:2]);gain,offset=backmod.canvas_affine(z['canvas_shape'],z['input_shape'])
  assert z['added_border']==100 and list(z['canvas_shape'])==affine['canvas_shape']and list(z['input_shape'])==affine['input_shape']
  assert gain==affine['gain']and np.array_equal(offset,affine['offset'])
  full=z['p3'].numpy();h,w=affine['p3_actual_hw'];assert full.dtype==np.float16 and full.shape==(64,80,80)
  assert not full[:,h:,:].any()and not full[:,:,w:].any()
  if first:
   with torch.inference_mode():zz=back.predict(np.ascontiguousarray(rgb[...,::-1]),already_padded=False,cpu_features=True)
   repeat+=1;assert torch.equal(z['p3'],zz['p3'])
   for key in('added_border','canvas_shape','input_shape'):assert z[key]==zz[key]
   first=False
  roi,_,_=math.sample_p3(full[:,:h,:w],q[None],[L],rgb.shape[:2],gain,offset,100)
  return roi[0],dict(added_border=100,canvas_shape=list(z['canvas_shape']),input_shape=list(z['input_shape']),actual_P3_hw=[h,w],gain=gain,offset=offset.tolist(),fullP3_stored=False)
 def changed(o):
  assert o['edited_fullRGB_format']in('NPY','PNG')
  path=check(o['edited_fullRGB_file']);a=np.load(path,allow_pickle=False)if o['edited_fullRGB_format']=='NPY'else np.array(Image.open(path).convert('RGB'),copy=True)
  return a
 def flush():
  if not buffer:return
  ci=len(chunks);cp=out/'chunks'/f'changed_{ci:04d}.npz';np.savez_compressed(cp,**{k:np.stack([x[k]for _,x in buffer])for k in FIELDS})
  d=bind(cp);chunks.append(dict(count=len(buffer),file=d))
  for j,(oi,x)in enumerate(buffer):output[oi]['input_ref']={'kind':'CHANGED_CHUNK','file':d,'index':j}
  buffer.clear()
 for i,(r,s)in enumerate(zip(rows,source)):
  same_q(r,s);base=cache.triplet(s);basefacts=input_facts(base)
  for o in occs[3*i:3*i+3]:
   x,isnew,facts=materialize_one(r,o,base,s['decoded_RGB_sha256'],changed,extract,math)
   oi=o['occurrence_index'];assert oi==len(output)
   rec=dict(occurrence_index=oi,occurrence_id=o['occurrence_id'],slot_index=i,variant=o['variant'],source=s['source'],input_id=r['input_id'],original_image_sha256=r['image_sha256'],image_sha256=o['edited_fullRGB_file']['sha256']if isnew else r['image_sha256'],split=r['split'],original_split=s.get('original_split',s.get('split',r['split'])),query_xy=r['query_xy'],L=r['L'],native_slot_logical_only=r['native_slot_logical_only'],original_input_valid=r['original_input_valid'],full_rgb_changed=isnew,full_rgb_decoded_sha256=o['full_rgb_decoded_sha256'],input_facts=facts,
      input_ref={'kind':'BASE_ROW','slot_index':i,'base_lineage':e['source_lineage']},tensor_value_sha256={k:array_sha(x[k])for k in FIELDS})
   output.append(rec);counts['changed'if isnew else'base_reused']+=1
   if isnew:buffer.append((oi,x))
   if len(buffer)==128:flush()
  if(i+1)%100==0:print(json.dumps(dict(rows=i+1,occurrences=len(output),source_predict_calls=forwards,seconds=time.monotonic()-start)),flush=True)
 flush();assert len(output)==2880
 after_state=None if back is None else state_sha(back.yolo.model);assert after_state==before_state
 if back is not None:
  assert not any(q.requires_grad for q in back.yolo.model.parameters())and not any(m.training for m in back.yolo.model.modules());back.close()
 for d in p['assets'].values():check(d)
 lp=out/'ALL2880_INPUT_LINEAGE.jsonl';lp.write_text(''.join(json.dumps(o,separators=(',',':'))+'\n'for o in output))
 gp=out/'ORIGINAL_NATURAL_LABELS_GT_ONLY.jsonl';gp.write_text(''.join(json.dumps({'slot_index':i,'original_natural_label_GT_only':g['original_natural_label_GT_only'],'before_aux_state_construction_only':g['before_aux_state'],'before_final_label_must_keep_natural_E':g['original_natural_label_GT_only']},separators=(',',':'))+'\n'for i,g in enumerate(gt)))
 manifest=dict(completed=True,entry=bind(entry_path),protocol=bind(PROTOCOL),base_rows=960,occurrences=2880,query_count=2880,summary=summary,counts=dict(counts),original_input_rows=e['input_rows'],base_input_lineage=e['source_lineage'],input_lineage=bind(lp),natural_before_GT_ledger=bind(gp),construction_GT_ledger=e['construction_ledger'],chunks=chunks,source_predict_calls=forwards,first_repeat_calls=repeat,blank_warmup_calls=warmup,model_state_before=before_state,model_state_after=after_state,model_unchanged=True,q_L_slot_candidate_order_unchanged=True,source_only=True,current_allowed=False,head_forward_training_PnP=0,seconds=time.monotonic()-start)
 write(out/'PACK_MANIFEST.json',manifest)
 write(out/'INPUT_PACK_QA.json',dict(completed=True,PASS=True,manifest=bind(out/'PACK_MANIFEST.json'),all2880_inputrefs_and_tensor_hashes_recorded=True,all_original_q_L_order_retained=True,unchanged_rgb_only_base_reuse=True,changed_fullRGB_freshP3_and_RGB96=True,finite_invalid_crops_not_zeroed=True,model_unchanged=True,source_only=True,no_head_fit_current_PnP=True))
 write(out/'INPUT_PACK_SEAL.json',dict(completed=True,PASS=True,manifest=bind(out/'PACK_MANIFEST.json'),QA=bind(out/'INPUT_PACK_QA.json'),lineage=bind(lp),original_natural_GT_only=bind(gp),changed_chunks=[c['file']for c in chunks],new_current_fit_PnP=0))
 print(json.dumps(dict(completed=True,seal=bind(out/'INPUT_PACK_SEAL.json'))),flush=True)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--entry',required=True);run(ap.parse_args().entry)
