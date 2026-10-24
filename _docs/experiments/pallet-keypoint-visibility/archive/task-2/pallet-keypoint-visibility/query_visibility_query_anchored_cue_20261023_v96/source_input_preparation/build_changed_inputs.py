"""V96 sparse matching source inputs. Actual execution requires a root entry.
Only native RGB and unchanged q/L reach the frozen feature extractor; no head.
"""
from pathlib import Path
from collections import Counter
import argparse,hashlib,importlib.util,json,os,sys,time
import numpy as np

HERE=Path(__file__).resolve().parent;V96=HERE.parent;P=V96.parent
V95=P/'query_visibility_same_q_counterfactual_20261023_v95'
OLD=V95/'actual_pair_builder_preparation/build_changed_inputs.py'
OLD_SHA='912c218de69f42e57fc9ce8951a255b9744d5e701993a227bc10f289171feb1d'
MATH=P/'query_visibility_public_cue_inputs_20261023_v90/producer/spatial_input_math.py'
MATH_SHA='6a6c2543e28715e3d47e15c543c25f75ff7d3efa3b881e523bd3451817fc004b'
PROTOCOL=HERE/'FEATURE_PROTOCOL_BEFORE_SOURCE_EXECUTION.json'
FIELDS=('p3_roi','rgb96','crop_valid96')
VARIANTS=('BEFORE','OPAQUE_COVER','CUE_PRESERVING_SHAM')
RAW_FIELDS={'slot_index','frame_index','input_id','native_slot_logical_only','split','image_sha256','raw_hw','query_xy','L','selected_box_xyxy','original_input_valid'}
PLAN_FIELDS=('input_id','source','raw_split','raw_hw','original_q','original_L','native_slot_logical_only','original_input_valid')
FORBIDDEN_JOB_FIELDS={'candidate_aux_hidden_label_GT_only','query_anchored_before_label','original_C1_V95_label','scalar_target_D0_E1_Uminus1','pair_ready','target_ref','protected_regions_GT_only','teacher_label','cue_reference_xy_unchanged','mask','mask_vis','mask_full','pose','depth','donor'}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def check(b):
 assert sha(b['path'])==b['sha256'],'Binding changed: '+b['path']
 return Path(b['path'])
def read(p):return json.loads(Path(p).read_text())
def lines(p):return [json.loads(s) for s in Path(p).read_text().splitlines() if s]
def write(p,x):
 with Path(p).open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def mod(p,name):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def dependencies():
 assert sha(OLD)==OLD_SHA and sha(MATH)==MATH_SHA
 return mod(OLD,'v96_pinned_v95_input_helpers'),mod(MATH,'v96_pinned_v90_math')
def arrsha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def layout(index,base):
 offset=index-33*base
 assert 0<=offset<33
 return (None,'BEFORE') if offset==0 else ((offset-1)//2,VARIANTS[1+(offset-1)%2])
def expected_id(r,rank,variant):
 prefix=f"{r['input_id']}::slot{r['native_slot_logical_only']}::V96::"
 return prefix+('BEFORE' if rank is None else f'RANK_{rank:02d}::{variant}')
def validate_metadata(rows,baseline,source,plan,occs,expected=960):
 assert len(rows)==len(source)==len(baseline)==expected and len(plan)==len(occs)==33*expected
 for i,(r,b,s) in enumerate(zip(rows,baseline,source)):
  assert set(r)==RAW_FIELDS and r['slot_index']==i
  for l in [b,s]:
   assert l['slot_index']==i and l['input_id']==r['input_id'] and l['native_slot_logical_only']==r['native_slot_logical_only']
   assert l['query_xy']==r['query_xy'] and l['L']==r['L'] and l['image_sha256']==r['image_sha256'] and l['split']==r['split']
   assert l['raw_hw']==r['raw_hw'] and set(l['before_input_refs'])==set(FIELDS)
  assert s['before_input_refs']==b['before_input_refs'] and s['decoded_RGB_sha256']==b['decoded_RGB_sha256']
  for k in ['source','selected_candidate_index','selected_candidate_score','before_decoded_sha256']:
   assert s[k]==b[k],(i,k)
  assert s['source']==('LOCO' if i<160 else 'PALLOX') if expected==960 else True
  q,L=r['query_xy'],r['L']
  assert (q is None)==(L is None)
  if q is not None:assert len(q)==2 and np.isfinite(q).all() and np.isfinite(L) and L>0
  for j in range(33):
   oi=33*i+j;o=occs[oi];a=plan[oi];rank,variant=layout(oi,i)
   assert not (set(o)&FORBIDDEN_JOB_FIELDS),'GT field in input-only image jobs'
   for z in [o,a]:
    assert z['occurrence_index']==oi and z['base_row_index']==i and z['donor_rank']==rank and z['variant']==variant
    assert z['occurrence_id']==expected_id(r,rank,variant)
   for k in PLAN_FIELDS:assert o[k]==a[k],(oi,k)
   assert a['input_id']==r['input_id'] and a['source']==s['source'] and a['raw_split']==r['split'] and a['raw_hw']==r['raw_hw']
   assert a['original_q']==q and a['original_L']==L and a['native_slot_logical_only']==r['native_slot_logical_only'] and a['original_input_valid']==r['original_input_valid']
   assert a['image_sha256_before']==r['image_sha256']
   assert o['slot_index']==i and o['before_decoded_sha256']==s['decoded_RGB_sha256']
   same=o['full_rgb_decoded_sha256']==s['decoded_RGB_sha256']
   assert bool(o['full_rgb_changed'])==(not same)
   if variant=='BEFORE':assert same
   if not same:
    assert q is not None,'Changed no-query job is invalid; retain absent-query baseline unchanged'
    assert o['edited_fullRGB_format'] in ['PNG','NPY'] and o['edited_fullRGB_file']
 return dict(base_rows=expected,occurrences=33*expected,raw_split_counts=dict(Counter(r['split'] for r in rows)),all_q_L_order_candidate_rawsplit_preserved=True)
def materialize(row,job,base,base_decoded_sha,load_changed,extract,math,old):
 same=job['full_rgb_decoded_sha256']==base_decoded_sha
 assert bool(job['full_rgb_changed'])==(not same)
 if same:return base,False,dict(reused_before_tensors=True,new_feature_forward=0,**old.input_facts(base))
 assert row['query_xy'] is not None and row['L'] is not None
 rgb=load_changed(job)
 assert rgb.dtype==np.uint8 and rgb.shape==tuple(row['raw_hw'])+(3,) and arrsha(rgb)==job['full_rgb_decoded_sha256']
 q=np.asarray(row['query_xy'],np.float64);L=float(row['L']);assert np.isfinite(q).all() and np.isfinite(L) and L>0
 # Global change always extracts fresh matching P3, irrespective of local pixels.
 roi,provenance=extract(rgb,q,L)
 crop,valid=math.crop(rgb,q,L,96)
 x=dict(p3_roi=roi,rgb96=crop,crop_valid96=valid)
 return x,True,dict(reused_before_tensors=False,new_feature_forward=1,feature_provenance=provenance,**old.input_facts(x))
def authorize(path):
 e=read(path)
 assert e['approved_by_root'] is True and e['source_only'] is True
 for k in ['current_allowed','training_allowed','head_allowed','PnP_allowed']:assert e[k] is False
 for k in ['code','protocol','generated','independent_review','intake_seal','metadata_lock']:check(e[k])
 assert e['code']==bind(__file__) and Path(e['protocol']['path'])==PROTOCOL
 p=read(PROTOCOL);proof=read(e['generated']['path'])
 assert proof['PASS'] and proof['code']==bind(__file__) and read(e['independent_review']['path'])['PASS']
 old,_=dependencies();intake=read(e['intake_seal']['path']);lock=read(e['metadata_lock']['path'])
 assert intake['completed'] and intake['PASS'] and lock['completed'] and lock['PASS']
 for k,v in p['code_bindings'].items():check(v)
 for k,v in p['assets'].items():check(v)
 for key in ['input_rows','base_source_lineage','input_plan']:
  assert e[key]==p['inputs'][key];check(e[key])
 for key in ['source_lineage','occurrences','gt_only_occurrences']:
  check(e[key]);assert old.contains(intake,e[key]),'Not in completed image seal: '+key
 assert old.contains(intake,e['metadata_lock']) or e['metadata_lock']==p['inputs']['metadata_lock']
 assert e['metadata_lock']==p['inputs']['metadata_lock']
 assert Path(e['output_directory']).resolve()==Path(p['output_directory']).resolve()
 return e,p
def run(entry_path):
 e,p=authorize(entry_path) # No source arrays/RGB/parameters before this guard.
 import torch,cv2
 from PIL import Image
 assert torch.__version__=='2.1.1+cu118' and cv2.__version__=='4.9.0'
 assert Path(sys.executable).resolve()==Path('/home/minjae/anaconda3/envs/pallet-yolo26/bin/python').resolve()
 old,math=dependencies()
 rows=lines(e['input_rows']['path']);baseline=lines(e['base_source_lineage']['path']);source=lines(e['source_lineage']['path']);plan=lines(e['input_plan']['path']);occs=lines(e['occurrences']['path'])
 summary=validate_metadata(rows,baseline,source,plan,occs)
 out=Path(e['output_directory']);out.mkdir(exist_ok=False);(out/'chunks').mkdir()
 write(out/'STARTED.json',dict(entry=bind(entry_path),code=bind(__file__),base_rows=960,occurrences=31680,no_silent_restart=True))
 torch.set_num_threads(1);cv2.setNumThreads(1);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 os.environ['YOLO_CONFIG_DIR']=str(HERE/'runtime/ultralytics')
 reader=old.RefReader();back=None;backmod=None;state_before=None;first=True;forwards=repeat=warmup=0;start=time.monotonic()
 def extract(rgb,q,L):
  nonlocal back,backmod,state_before,first,forwards,repeat,warmup
  if back is None:
   backmod=mod(p['assets']['backbone_code']['path'],'v96_frozen_yolo')
   back=backmod.FrozenYoloFeatures(p['assets']['backbone_checkpoint']['path'],device='cuda:0')
   with torch.inference_mode():back.predict(np.zeros((480,640,3),np.uint8),already_padded=False,cpu_features=True)
   warmup=1;state_before=old.state_sha(back.yolo.model)
  with torch.inference_mode():z=back.predict(np.ascontiguousarray(rgb[...,::-1]),already_padded=False,cpu_features=True)
  forwards+=1;affine=math.expected_affine(rgb.shape[:2]);gain,offset=backmod.canvas_affine(z['canvas_shape'],z['input_shape'])
  assert z['added_border']==100 and list(z['canvas_shape'])==affine['canvas_shape'] and list(z['input_shape'])==affine['input_shape']
  assert gain==affine['gain'] and np.array_equal(offset,affine['offset'])
  full=z['p3'].numpy();h,w=affine['p3_actual_hw'];assert full.dtype==np.float16 and full.shape==(64,80,80)
  assert not full[:,h:,:].any() and not full[:,:,w:].any()
  if first:
   with torch.inference_mode():zz=back.predict(np.ascontiguousarray(rgb[...,::-1]),already_padded=False,cpu_features=True)
   repeat+=1;assert torch.equal(z['p3'],zz['p3'])
   for key in ['added_border','canvas_shape','input_shape']:assert z[key]==zz[key]
   first=False
  roi,_,_=math.sample_p3(full[:,:h,:w],q[None],[L],rgb.shape[:2],gain,offset,100)
  return roi[0],dict(added_border=100,canvas_shape=list(z['canvas_shape']),input_shape=list(z['input_shape']),actual_P3_hw=[h,w],gain=gain,offset=offset.tolist(),fullP3_stored=False)
 def changed(o):
  path=check(o['edited_fullRGB_file'])
  return np.load(path,allow_pickle=False) if o['edited_fullRGB_format']=='NPY' else np.array(Image.open(path).convert('RGB'),copy=True)
 output=[];units=[];buffer=[];chunks=[];counts=Counter()
 def flush():
  if not buffer:return
  cp=out/'chunks'/f'changed_{len(chunks):04d}.npz'
  np.savez_compressed(cp,**{k:np.stack([x[k] for _,_,x in buffer]) for k in FIELDS})
  d=bind(cp);chunks.append(dict(count=len(buffer),file=d))
  for j,(oi,ui,x) in enumerate(buffer):
   ref=dict(kind='CHANGED_CHUNK',file=d,index=j);output[oi]['input_ref']=ref;units[ui]['input_ref']=ref
  buffer.clear()
 # Reserve one exact-bound base unit per960 rows. No changed units deduplicated.
 units=[None]*960
 for i,(r,s) in enumerate(zip(rows,source)):
  base=reader.triplet(s);basefacts=dict(reused_before_tensors=True,new_feature_forward=0,**old.input_facts(base));basehash={k:arrsha(base[k]) for k in FIELDS}
  baseref=dict(kind='BASE_ROW',slot_index=i,base_lineage=e['source_lineage'])
  units[i]=dict(input_unit_index=i,slot_index=i,input_ref=baseref,input_facts=basefacts,tensor_value_sha256=basehash)
  for o in occs[33*i:33*i+33]:
   x,isnew,facts=materialize(r,o,base,s['decoded_RGB_sha256'],changed,extract,math,old)
   oi=o['occurrence_index'];assert oi==len(output)
   if isnew:
    ui=len(units);ref=None;hashes={k:arrsha(x[k]) for k in FIELDS}
    units.append(dict(input_unit_index=ui,slot_index=i,input_ref=None,input_facts=facts,tensor_value_sha256=hashes))
   else:ui=i;ref=baseref;hashes=basehash
   output.append(dict(occurrence_index=oi,occurrence_id=o['occurrence_id'],slot_index=i,base_row_index=i,donor_rank=o['donor_rank'],variant=o['variant'],source=s['source'],input_id=r['input_id'],raw_split=r['split'],split=r['split'],original_split=r['split'],original_image_sha256=r['image_sha256'],image_sha256=o['edited_fullRGB_file']['sha256'] if isnew else r['image_sha256'],query_xy=r['query_xy'],L=r['L'],native_slot_logical_only=r['native_slot_logical_only'],original_input_valid=r['original_input_valid'],selected_box_xyxy=r['selected_box_xyxy'],selected_candidate_index=s['selected_candidate_index'],selected_candidate_score=s['selected_candidate_score'],full_rgb_changed=isnew,full_rgb_decoded_sha256=o['full_rgb_decoded_sha256'],input_unit_index=ui,input_ref=ref,input_facts=facts,tensor_value_sha256=hashes))
   counts['changed' if isnew else 'base_reused']+=1
   if isnew:buffer.append((oi,ui,x))
   if len(buffer)==128:flush()
  if (i+1)%100==0:print(json.dumps(dict(base_rows=i+1,occurrences=len(output),source_predict_calls=forwards,seconds=time.monotonic()-start)),flush=True)
 flush();assert len(output)==31680 and all(u is not None for u in units)
 state_after=None if back is None else old.state_sha(back.yolo.model);assert state_after==state_before
 if back is not None:
  assert not any(t.requires_grad for t in back.yolo.model.parameters()) and not any(m.training for m in back.yolo.model.modules());back.close()
 assert forwards==counts['changed'] and repeat==int(forwards>0) and warmup==int(forwards>0)
 for b in p['assets'].values():check(b)
 lp=out/'ALL31680_INPUT_LINEAGE.jsonl';up=out/'INPUT_UNITS_LINEAGE.jsonl';gp=out/'GT_ONLY_PROVENANCE.json'
 for path,rs in [(lp,output),(up,units)]:
  with path.open('x') as f:
   for r in rs:f.write(json.dumps(r,separators=(',',':'),allow_nan=False)+'\n')
 write(gp,dict(completed=True,GT_only_occurrences=e['gt_only_occurrences'],body_read_by_feature_driver=False,GT_fields_may_forward=False,native_expected_part_state='UNKNOWN',human_gold=False,training_admission=False))
 manifest=dict(completed=True,entry=bind(entry_path),protocol=bind(PROTOCOL),base_rows=960,occurrences=31680,query_count=31680,unique_input_units=len(units),summary=summary,counts=dict(counts),original_input_rows=e['input_rows'],base_input_lineage=e['source_lineage'],input_lineage=bind(lp),input_units=bind(up),GT_only_provenance=bind(gp),chunks=chunks,source_predict_calls=forwards,first_repeat_calls=repeat,blank_warmup_calls=warmup,model_state_before=state_before,model_state_after=state_after,model_unchanged=True,q_L_slot_candidate_order_unchanged=True,raw_splits_unchanged=True,changed_inputs_deduplicated=False,GT_fields_may_forward=False,source_only=True,current_allowed=False,head_forward_training_PnP=0,seconds=time.monotonic()-start)
 write(out/'PACK_MANIFEST.json',manifest)
 write(out/'INPUT_PACK_QA.json',dict(completed=True,PASS=True,manifest=bind(out/'PACK_MANIFEST.json'),all31680_retained=True,unique_base_units=960,changed_matching_units=counts['changed'],all_original_q_L_order_candidate_rawsplit_retained=True,fullRGB_exact_bound_reuse_only=True,changed_global_local_equal_still_fresh=True,finite_invalid_not_blanketed_zero=True,model_unchanged=True,source_only=True,no_head_fit_current_PnP=True))
 write(out/'INPUT_PACK_SEAL.json',dict(completed=True,PASS=True,manifest=bind(out/'PACK_MANIFEST.json'),QA=bind(out/'INPUT_PACK_QA.json'),lineage=bind(lp),input_units=bind(up),original_before_GT_only=bind(gp),changed_chunks=[c['file'] for c in chunks],new_current_fit_PnP=0))
 print(json.dumps(dict(completed=True,seal=bind(out/'INPUT_PACK_SEAL.json'))),flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--entry',required=True);run(ap.parse_args().entry)
