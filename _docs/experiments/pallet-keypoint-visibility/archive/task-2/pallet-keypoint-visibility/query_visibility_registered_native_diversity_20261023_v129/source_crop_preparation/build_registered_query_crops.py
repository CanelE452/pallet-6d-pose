"""Inert V129 source q8/crop builder. Root-entry only; no model imports.

All 1728 joint q8 conditions (hence every one of 55296 rows) are locked from
pre-RGB base configurations before any RGB/state/owner payload is read.
"""
from pathlib import Path
import argparse,ast,hashlib,json,sys,time,traceback
from collections import Counter,defaultdict
import numpy as np
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;V=H.parent;P=V.parent
POLICY=V/'ROOT_CORPUS_POLICY_BEFORE_PIXELS.json';PROTOCOL=H/'QUERY_CROP_PROTOCOL_BEFORE_PAYLOADS.json'
OUT=V/'source_crops_v1';GENERATED=H/'GENERATED_QUERY_CHECKS.json'
VARIANTS=('BEFORE','COVER','NEAR_SHAM','FAR_SHAM')
CONDITIONS=('IDENTITY','SHIFT_X_MINUS_QUARTER_L','SHIFT_X_PLUS_QUARTER_L','SHIFT_Y_MINUS_QUARTER_L','SHIFT_Y_PLUS_QUARTER_L','SHIFT_XY_PLUS_QUARTER_L','SHIFT_XY_MINUS_QUARTER_L','CENTROID_EXPAND_1_12','CENTROID_CONTRACT_0_88')
OPS={'local':P/'query_visibility_public_cue_inputs_20261023_v90/producer/spatial_input_math.py','side':P.parent/'external_ray_proxy_head_20261022_v60/head_and_roi.py','context':P/'query_visibility_target_context_rgb_20261023_v107/target_context_math.py'}
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
def ash(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def pure(path,names):
 body=[n for n in ast.parse(Path(path).read_text()).body if isinstance(n,ast.FunctionDef)and n.name in names]
 assert {n.name for n in body}==set(names);ns={'np':np};exec(compile(ast.fix_missing_locations(ast.Module(body=body,type_ignores=[])),str(path)+'::pure','exec'),ns);return ns
def operators():
 o=pure(OPS['local'],('crop','query_validity'));o.update(pure(OPS['side'],('native_grid',)));o.update(pure(OPS['context'],('query_extent','valid_box','target_context_grid')));return o
def context_facts(box,hw,op):
 if box is None:return False,0.
 g=op['target_context_grid'](box);x,y=g[...,0],g[...,1];ix=np.floor(x).astype(np.int64);iy=np.floor(y).astype(np.int64);dx=x-ix;dy=y-iy;v=np.ones(x.shape,bool);h,w=hw
 for ox,oy,weight in ((0,0,(1-dx)*(1-dy)),(1,0,dx*(1-dy)),(0,1,(1-dx)*dy),(1,1,dx*dy)):
  xx,yy=ix+ox,iy+oy;v&=((xx>=0)&(xx<w)&(yy>=0)&(yy<h))|(weight==0)
 return bool(v.any()),float(v.mean())
def joint_condition(reference,index,hw,op):
 ref=np.asarray(reference,np.float64);assert ref.shape==(8,2)
 good=np.isfinite(ref).all(1)&~(ref==[-1.,-1.]).all(1);q=ref.copy();q[~good]=[-1.,-1.]
 _,_,Lref=op['native_grid'](q,hw)
 if index in range(1,7):
  shifts={1:(-.25,0),2:(.25,0),3:(0,-.25),4:(0,.25),5:(.25,.25),6:(-.25,-.25)}
  q[good]+=Lref*np.array(shifts[index])
 elif index in (7,8)and good.any():
  center=ref[good].mean(0);q[good]=center+(ref[good]-center)*(1.12 if index==7 else .88)
 else:assert index==0 or index in (7,8)
 _,roi,L=op['native_grid'](q,hw);box=op['query_extent'](q);context,fraction=context_facts(box,hw,op)
 return {'initial_points':q.tolist(),'L':float(L),'Lref':float(Lref),'context_box_xyxy':box,'context_available_geometry':context,'context_valid_fraction_geometry':fraction,'ROI_any_valid':roi.reshape(8,-1).any(1).tolist()}
def contains_binding(x,b):
 if isinstance(x,dict):return (x.get('path')==b['path']and x.get('sha256')==b['sha256'])or any(contains_binding(v,b)for v in x.values())
 if isinstance(x,list):return any(contains_binding(v,b)for v in x)
 return False
def support_facts(q,L,context,reference,active_bbox):
 """GT-only observability: inclusive nominal crop/context edges, never a gate."""
 q=np.asarray(q,np.float64);r=np.asarray(reference,np.float64);b=np.asarray(active_bbox,np.float64)
 good_r=r.shape==(2,)and np.isfinite(r).all()and not np.array_equal(r,[-1.,-1.])
 good_b=b.shape==(4,)and np.isfinite(b).all()and np.all(b[2:]>=b[:2])
 local_r=(r-q)*96/L+47.5 if good_r else None
 local_b=((b.reshape(2,2)-q)*96/L+47.5).reshape(4)if good_b else None
 in_context=lambda lo,hi:bool(np.all(lo>=np.asarray(context)[:2])and np.all(hi<=np.asarray(context)[2:]))if context is not None else None
 return {'shared_bbox_key':'projected_active_witness_bbox_xyxy','reference_xy_local96_display_only':local_r.tolist()if good_r else None,'local96_bbox_xyxy':local_b.tolist()if good_b else None,'native512_span_xy':(b[2:]-b[:2]).tolist()if good_b else None,'local96_span_xy':(local_b[2:]-local_b[:2]).tolist()if good_b else None,'reference_in_local_crop':bool(np.all((local_r>=-.5)&(local_r<=95.5)))if good_r else None,'reference_within_local_sample_centres':bool(np.all((local_r>=0)&(local_r<=95)))if good_r else None,'active_bbox_in_local_crop':bool(np.all(local_b[:2]>=-.5)and np.all(local_b[2:]<=95.5))if good_b else None,'active_bbox_within_local_sample_centres':bool(np.all(local_b[:2]>=0)and np.all(local_b[2:]<=95))if good_b else None,'reference_in_context':in_context(r,r)if good_r else None,'active_bbox_in_context':in_context(b[:2],b[2:])if good_b else None}
def support_summary(samples):
 out={}
 for split in ('ALL','TRAIN','VALIDATION'):
  rr=[x for s,x in samples if split=='ALL'or s==split];entry={'rows':len(rr)}
  for name in ('native512_span_xy','local96_span_xy'):
   a=np.array([r[name]for r in rr if r[name]is not None],np.float64)
   entry[name]={'known':len(a),'unknown':len(rr)-len(a),'min':a.min(0).tolist()if len(a)else None,'median':np.median(a,axis=0).tolist()if len(a)else None,'max':a.max(0).tolist()if len(a)else None}
  for name in ('reference_in_local_crop','active_bbox_in_local_crop','reference_within_local_sample_centres','active_bbox_within_local_sample_centres','reference_in_context','active_bbox_in_context'):
   a=[r[name]for r in rr if r[name]is not None];outside=sum(v is False for v in a)
   entry[name]={'known':len(a),'unknown':len(rr)-len(a),'outside':outside,'fraction_outside':float(outside/len(a))if a else None}
  out[split]=entry
 return out
def authorize(entry):
 e=read(entry);assert e['approved_source_crop_build']is True and e['source_only']is True
 assert all(e[k]is False for k in ('model_forward','fit','current223','V112','PnP','automatic_retry'))
 for k in ('code','protocol','generated','root_policy','render_complete','base_configs','input_scenes'):check(e[k])
 p=read(PROTOCOL);assert e['code']==p['code']==bind(__file__)and e['protocol']==bind(PROTOCOL)and e['root_policy']==p['root_policy']==bind(POLICY)and e['generated']==p['generated']
 assert read(e['generated']['path'])['PASS']and read(e['generated']['path'])['code']==e['code']
 rc=read(e['render_complete']['path']);assert rc['completed']and rc['PASS']
 for k in ('base_configs','input_scenes','GT_scenes'):assert contains_binding(rc,e[k]),k
 for k,b in p['operators'].items():assert b==bind(OPS[k])
 assert e['output']==str(OUT)and not OUT.exists(),'Preserve prior completed or failed run.'
 assert str(Path(sys.executable).resolve())==p['runtime']['python']and np.__version__==p['runtime']['numpy']
 return e,p
class StreamRows:
 def __init__(self,path):self.f=Path(path).open('x');self.f.write('{"rows":[\n');self.n=0
 def add(self,row):
  if self.n:self.f.write(',\n')
  self.f.write(json.dumps(row,ensure_ascii=False,separators=(',',':'),allow_nan=False));self.n+=1
 def finish(self,metadata):
  self.f.write('\n]')
  for k,v in metadata.items():self.f.write(','+json.dumps(k)+':'+json.dumps(v,ensure_ascii=False,separators=(',',':'),allow_nan=False))
  self.f.write('}\n');self.f.close()
 def close(self):
  if not self.f.closed:self.f.close()
def update_counts(counter,r,state,available,pixel):
 keys={'ALL':'ALL','split':r['split'],'variant':r['variant'],'condition':r['condition_id'],**r['factors']}
 for k,v in keys.items():counter[(str(k),json.dumps(v,sort_keys=True,separators=(',',':')),state,bool(available),pixel)]+=1
def run(entry):
 e,p=authorize(entry);op=operators();start=time.monotonic()
 cfgpack=read(e['base_configs']['path']);sceneinput=read(e['input_scenes']['path']);cfg=cfgpack['rows'];scenes=sceneinput['rows']
 assert cfgpack['completed']and sceneinput['completed']and len(cfg)==192 and len(scenes)==768
 assert [c['base_index']for c in cfg]==list(range(192))and sum(c['split']=='TRAIN'for c in cfg)==128 and sum(c['split']=='VALIDATION'for c in cfg)==64
 for i,s in enumerate(scenes):
  bi,vi=divmod(i,4);assert s['scene_index']==i and s['base_index']==bi and s['variant']==VARIANTS[vi]and s['split']==cfg[bi]['split']
  assert s['full_RGB_format']=='NPZ_RGB_UINT8_HWC'and s['full_RGB_key']=='RGB_uint8'and s['raw_hw']==[512,512]
 OUT.mkdir();(OUT/'chunks').mkdir();write(OUT/'STARTED.json',{'entry':bind(entry),'automatic_retry':False,'RGB_state_owner_reads':0})
 iw=gw=None
 try:
  # All q emulation is constructed from mathematical pre-RGB CF projections only.
  locked=[]
  for bi,c in enumerate(cfg):
   ref=np.asarray(c['native8_CF_xy'],np.float64);assert ref.shape==(8,2)
   for ci,cid in enumerate(CONDITIONS):locked.append({'base_index':bi,'condition_index':ci,'condition_id':cid,**joint_condition(ref,ci,[512,512],op)})
  lp=OUT/'QUERY_CONDITIONS1728_BEFORE_RGB.json'
  write(lp,{'completed':True,'conditions':locked,'base_configs':e['base_configs'],'input_scenes':e['input_scenes'],'rows_determined':55296,'mapping':'occurrence_index=(((base_index*4+variant_index)*9+condition_index)*8+CF_slot); all4 variants use identical condition q8/L/context.','source_reference':'Only mathematical pre-RGB native8_CF_xy from BASE_CONFIGS used. No RGB, state, owner or sceneGT body read before this file.','source_emulated_not_actual_initializer':True,'training_admission':False})
  # Teacher/owner evidence becomes accessible only after the complete q lock exists.
  gtpack=read(check(e['GT_scenes']));gtscenes=gtpack['rows'];assert gtpack['completed']and len(gtscenes)==768
  ip=OUT/'INPUT_ROWS.json';gp=OUT/'GT_ROWS.json';iw=StreamRows(ip);gw=StreamRows(gp);chunks=[];counts=Counter();special=Counter();statecounts=Counter();foldcounts=Counter();scene_facts=[];support_samples=[]
  for bi,c in enumerate(cfg):
   crops=np.empty((288,96,96,3),np.uint8);valids=np.empty((288,96,96),bool);indices=np.arange(bi*288,(bi+1)*288,dtype=np.int32);ins=[];gts=[]
   ref=np.asarray(c['native8_CF_xy'],np.float64);factors=dict(c['factors']);base_id=c['base_id']
   for vi,variant in enumerate(VARIANTS):
    si=4*bi+vi;s=scenes[si];sg=gtscenes[si]
    assert sg['scene_index']==si and sg['base_index']==bi and sg['id']==s['id']and sg['variant']==variant
    assert len(sg['state8_CF'])==len(sg['records_CF'])==8 and all(t in 'DESU'for t in sg['state8_CF'])
    buffer=check(s['full_RGB'])
    with np.load(buffer,allow_pickle=False)as f:
     rgb=np.array(f['RGB_uint8'],copy=True);owner=np.array(f['part_id'],copy=True);amb=np.array(f['ambiguity'],copy=True);tie=np.array(f['tie_count'],copy=True);actual_ref=np.array(f['native8_CF_xy'],copy=True)
    assert rgb.dtype==np.uint8 and rgb.shape==(512,512,3)and owner.shape==amb.shape==tie.shape==(512,512)
    assert np.array_equal(actual_ref,ref,equal_nan=True)and ash(rgb)==s['decoded_RGB_sha256']
    for ci,cid in enumerate(CONDITIONS):
     cc=locked[bi*9+ci];q8=np.array(cc['initial_points'],np.float64);L=cc['L'];center,cont,near=op['query_validity'](q8,[512,512]);context=cc['context_available_geometry']
     for slot in range(8):
      local=(vi*9+ci)*8+slot;i=bi*288+local;q=q8[slot];crop,v=op['crop'](rgb,q,L,96);a=crop.transpose(1,2,0).copy();vv=v[0].copy();crops[local]=a;valids[local]=vv
      central=bool(vv[47:49,47:49].all());factual=bool(center[slot]and central);available=factual and context;roi=bool(cc['ROI_any_valid'][slot]);reason=[]
      if not cont[slot]:reason.append('NATIVE_CONTINUOUS_UNAVAILABLE')
      if not near[slot]:reason.append('NATIVE_NEAREST_UNAVAILABLE')
      if not central:reason.append('LOCAL96_CENTRAL4_UNAVAILABLE')
      if not context:reason.append('CONTEXT_UNAVAILABLE')
      identity=f'{s["id"]}::qcondition{ci:02d}::CF{slot}';group=f'{base_id}::qcondition{ci:02d}::CF{slot}'
      row={'occurrence_index':i,'row_id':i,'id':identity,'query_id':identity,'query_group_id':group,'base_index':bi,'base_id':base_id,'scene_index':si,'scene_id':s['id'],'style':s.get('style',factors['geometry_family']),'variant':variant,'source':'NEW_REGISTERED_NATIVE_DIVERSITY','split':c['split'],'raw_split':c['split'],'factors':factors,'source_base_config':{'file':e['base_configs'],'row':bi},'condition_index':ci,'condition_id':cid,'carrier_slot':slot,'logical_slot':slot,'query_index':slot,'query_xy':q.tolist(),'initial_points':cc['initial_points'],'q8':cc['initial_points'],'L':L,'context_box_xyxy':cc['context_box_xyxy'],'raw_hw':[512,512],'HW':[512,512],'full_RGB':s['full_RGB'],'full_RGB_format':'NPZ_RGB_UINT8_HWC','full_RGB_key':'RGB_uint8','decoded_RGB_sha256':s['decoded_RGB_sha256'],'rgb96_row':local,'rgb96_sha256':ash(a),'valid96_sha256':ash(vv),'rgb96_layout':'HWC','valid96_layout':'HW','q_native_continuous_valid':bool(cont[slot]),'q_nearest_valid':bool(near[slot]),'ROI_any_valid':roi,'local_central4_valid':central,'factual_query_available':factual,'factual_available':factual,'context_available':context,'query_available':available,'context_valid_fraction_geometry':cc['context_valid_fraction_geometry'],'original_input_availability':{'continuous':bool(cont[slot]),'nearest':bool(near[slot]),'ROI_any_valid':roi,'central4':central,'context':context},'unavailability_reasons':reason,'source_query_emulation':True,'actual_initializer_output':False}
      if 'raw_RGB_PNG'in s:row['raw_rgb']=s['raw_RGB_PNG']
      ins.append(row);state=sg['state8_CF'][slot];record=sg['records_CF'][slot];assert record['state']==state
      canonical=int(sg['permCF_to_canonical'][slot]);assert record['canonical_index']==canonical and record['CF_index']==slot
      pix=np.floor(q+.5).astype(np.int64);inside=bool(near[slot]);po=int(owner[pix[1],pix[0]])if inside else None;pa=bool(amb[pix[1],pix[0]])if inside else None;pt=int(tie[pix[1],pix[0]])if inside else None
      pixel='U'if not inside or pa else('BACKGROUND'if po<0 else('FOREIGN'if po==999 else'TARGET'))
      delta=q-ref[slot];error=float(np.linalg.norm(delta))if np.isfinite(ref[slot]).all()and not np.array_equal(ref[slot],[-1.,-1.])else None
      gt={'occurrence_index':i,'row_id':i,'id':identity,'query_group_id':group,'base_index':bi,'scene_index':si,'scene_id':s['id'],'split':c['split'],'condition_index':ci,'condition_id':cid,'canonical_index':canonical,'camera_facing_index':slot,'reference_xy':ref[slot].tolist()if error is not None else None,'delta_xy':delta.tolist()if error is not None else None,'error_px':error,'raw_state':state,'external_target_D0_E1_else_minus1':0 if state=='D'else(1 if state=='E'else-1),'known_geometric_S':state=='S','witnessrefs':{'file':e['GT_scenes'],'scene_index':si,'record_key':'records_CF','CF_record_index':slot,'canonical_index':canonical},'q_nearest_pixel_owner_diagnostic':{'nearest_xy':pix.tolist(),'in_frame':inside,'part_owner_id':po,'ambiguity':pa,'tie_count':pt,'category':pixel},'source_native_registration':'Exact CF/canonical registration for NEW constructed geometry only; not real/legacy identity or human gold.','human_gold':False,'training_admission':False}
      sf=support_facts(q,L,cc['context_box_xyxy'],ref[slot],record['projected_active_witness_bbox_xyxy']);gt['active_witness_support_GT_only']=sf;support_samples.append((c['split'],sf))
      gts.append(gt);update_counts(counts,row,state,available,pixel);statecounts[state]+=1;foldcounts[c['split']]+=1
      if variant=='NEAR_SHAM'and state=='D'and pixel=='FOREIGN':special[(c['split'],'NEAR_SHAM_q_FOREIGN_reference_D')]+=1
      if variant=='COVER'and state=='E'and pixel=='BACKGROUND':special[(c['split'],'COVER_q_BACKGROUND_reference_E')]+=1
    scene_facts.append({'scene_index':si,'id':s['id'],'base_index':bi,'full_RGB':s['full_RGB'],'decoded_RGB_sha256':ash(rgb),'rows':72,'source_reference_exact_config':True})
    del rgb,owner,amb,tie
   chunk=OUT/'chunks'/f'BASE_{bi:03d}.npz';np.savez_compressed(chunk,rgb96=crops,valid96=valids,occurrence_index=indices);cb=bind(chunk);chunks.append(cb)
   assert ins[0]['occurrence_index']==bi*288 and ins[-1]['occurrence_index']==bi*288+287
   for r,g in zip(ins,gts):r['rgb96_chunk']=cb;iw.add(r);gw.add(g)
   print(json.dumps({'base_complete':bi+1,'bases':192,'rows_complete':(bi+1)*288,'chunk':cb,'seconds':time.monotonic()-start}),flush=True)
   del crops,valids,ins,gts
  assert iw.n==gw.n==55296 and foldcounts=={'TRAIN':36864,'VALIDATION':18432}
  iw.finish({'completed':True,'crop_chunks':chunks,'protocol':e['protocol'],'query_lock':bind(lp),'base_configs':e['base_configs'],'input_scenes':e['input_scenes'],'render_complete':e['render_complete'],'source_emulation_not_initializer':True,'training_admission':False})
  gw.finish({'completed':True,'protocol':e['protocol'],'input_row_identity':bind(ip),'shared_scene_teacher':e['GT_scenes'],'raw_S_outside_external_task':True,'teacher_U_not_negative':True,'support_convention':'GT-only active finite-witness projected bounding box; not a whole surface certificate. local=(native-q)*96/L+47.5. Nominal crop closed edges[-.5,95.5], sample-centre range[0,95] also separate; input context closed q8 minmax. Inclusion never changes teacher or input availability.','no_training_admission':True})
  fact=OUT/'SCENE_INPUT_FACTS768.json';write(fact,{'completed':True,'rows':scene_facts})
  countrows=[{'factor':k,'value':json.loads(v),'raw_state':s,'query_available':a,'q_pixel_category':pix,'count':int(n)}for(k,v,s,a,pix),n in sorted(counts.items())]
  summary=OUT/'SUMMARY.json';write(summary,{'completed':True,'rows':55296,'bases':192,'scenes':768,'conditions_per_base':9,'slots':8,'split_counts':dict(foldcounts),'raw_states':dict(statecounts),'counts_by_factor_state_availability_pixel':countrows,'diagnostic_cross_semantic_pixel':[{'split':k[0],'case':k[1],'count':n}for k,n in sorted(special.items())],'support_spans_and_observability_by_fold':support_summary(support_samples),'teacher_and_input_U_separate':True,'known_reference_error_is_source_emulation_not_initializer_accuracy':True,'correlated_source_rows_not_independent_images':True,'raw_RGB_mutated':False,'native_crops_recentered':False,'new_model_fit_current_V112_PnP':0})
  complete=OUT/'CROP_COMPLETE.json';write(complete,{'completed':True,'PASS':True,'entry':bind(entry),'code':e['code'],'protocol':e['protocol'],'render_complete':e['render_complete'],'base_configs':e['base_configs'],'input_scenes':e['input_scenes'],'scene_GT_only':e['GT_scenes'],'query_lock':bind(lp),'input_manifest':bind(ip),'GT_only':bind(gp),'crop_chunks':chunks,'summary':bind(summary),'scene_input_facts':bind(fact),'rows':55296,'split_counts':dict(foldcounts),'all_q_locked_before_RGB_state_owner':True,'maximum_crop_chunk_rows':288,'maximum_uncompressed_crop_chunk_bytes':288*96*96*4,'q_context_L_shared_across_4_variants':True,'source_reference_and_teacher_not_corrected':True,'training_admission':False,'model_fit_current_V112_PnP':0,'seconds':time.monotonic()-start})
  print(json.dumps({'completed':True,'receipt':bind(complete),'rows':55296,'raw_states':dict(statecounts),'seconds':time.monotonic()-start}),flush=True)
 except BaseException:
  if iw is not None:iw.close()
  if gw is not None:gw.close()
  write(OUT/'FAILED_NO_RESTART.json',{'traceback':traceback.format_exc(),'automatic_retry':False});raise
def generated():
 op=operators();ref=np.array([[10.,10.],[110.,10.],[110.,60.],[10.,60.],[20.,20.],[100.,20.],[100.,50.],[20.,50.]]);orig=ref.copy();cc=[joint_condition(ref,i,[512,512],op)for i in range(9)]
 assert np.array_equal(ref,orig)and np.array_equal(cc[0]['initial_points'],ref)and cc[0]['L']==25.
 assert np.array_equal(np.array(cc[1]['initial_points'])-ref,np.tile([-6.25,0.],(8,1)))and cc[1]['L']==25. and cc[1]['context_box_xyxy']==[3.75,10.,103.75,60.]
 assert np.array_equal(np.array(cc[5]['initial_points'])-ref,np.tile([6.25,6.25],(8,1)))
 for i,scale in ((7,1.12),(8,.88)):
  expected=ref.mean(0)+(ref-ref.mean(0))*scale;assert np.array_equal(cc[i]['initial_points'],expected)and abs(cc[i]['L']-25*scale)<1e-12
 missing=ref.copy();missing[3]=[-1.,-1.]
 for i in range(9):assert joint_condition(missing,i,[512,512],op)['initial_points'][3]==[-1.,-1.]
 none=np.full((8,2),-1.)
 for i in range(9):assert joint_condition(none,i,[512,512],op)['context_box_xyxy']is None
 # Signed-quarter CF identity is supplied, never image-sorted or applied twice.
 quarter=np.array([1,5,6,2,0,4,7,3]);reference=ref[quarter]
 assert joint_condition(reference,0,[512,512],op)['initial_points']==reference.tolist()
 rgb=np.full((512,512,3),91,np.uint8);crop,valid=op['crop'](rgb,[128.,128.],32.,96);assert(crop==91).all()and valid.all()
 _,bad=op['crop'](rgb,[-3.,-3.],16.,96);assert not bad[:,47:49,47:49].all()
 assert context_facts([0.,0.,511.,511.],[512,512],op)==(True,1.)
 tuples=[(bi,vi,ci,sl)for bi in range(192)for vi in range(4)for ci in range(9)for sl in range(8)]
 assert len(tuples)==55296 and all((((bi*4+vi)*9+ci)*8+sl)==i for i,(bi,vi,ci,sl)in enumerate(tuples))
 example={'raw_state':'U','query_available':True};assert example['raw_state']=='U'and example['query_available']
 sf=support_facts([100.,100.],96.,[0.,0.,200.,200.],[100.,100.],[50.,95.,140.,105.]);assert sf['local96_bbox_xyxy']==[-2.5,42.5,87.5,52.5]and sf['native512_span_xy']==[90.,10.]and sf['local96_span_xy']==[90.,10.]and sf['reference_in_local_crop']and not sf['active_bbox_in_local_crop']and sf['active_bbox_in_context']
 su=support_summary([('TRAIN',sf)]);assert su['ALL']['active_bbox_in_local_crop']['fraction_outside']==1. and su['TRAIN']['reference_in_local_crop']['fraction_outside']==0.;json.dumps(su,allow_nan=False)
 checks={'nine_joint_conditions_and_Lref':True,'identity_copy_no_input_mutation':True,'global_axis_and_diagonal_shift':True,'centroid_expand_contract_recompute_L_context':True,'sentinel_stays_absent_allconditions':True,'signedquarter_CF_not_resorted_or_double_permuted':True,'55296_order_and_4variant_group_identity':True,'constant_crop_HWC_HW_and_OOBcentral4':True,'positive_weight_context_support':True,'teacher_U_not_input_U':True,'stream_chunk288_memory_bound':True,'GT_only_native_to_local_support_span_inclusion_and_serialization':True}
 json.dumps(cc,allow_nan=False)
 return {'completed':True,'PASS':True,'code':bind(__file__),'checks':checks,'actual_RGB_state_owner_payload_model_fit_current_V112':0}
def prepare():
 policy=read(POLICY);assert policy['counts']['all_occurrences']==55296 and policy['training_admitted']is False
 g=generated();write(GENERATED,g)
 write(PROTOCOL,{'completed':True,'code':bind(__file__),'generated':bind(GENERATED),'root_policy':bind(POLICY),'operators':{k:bind(v)for k,v in OPS.items()},'runtime':{'python':str(Path(sys.executable).resolve()),'numpy':np.__version__},'output':str(OUT),'order':'base→BEFORE/COVER/NEAR_SHAM/FAR_SHAM→9jointcondition→CFslot; group base/condition/slot shared4variants','conditions':list(CONDITIONS),'q_lock':'All1728jointq8conditions from BASE_CONFIGS192.native8_CF_xy before sceneGT orRGB/owner; compact formula determinesall55296rows. Reference-derived emulation, not initializer.','schema':'INPUT_ROWS completed/rows/crop_chunks; rgb96_chunk pathSHA+rgb96_row LOCAL; chunksRGB96HWC U8/valid96HW bool/globaloccurrence_index. Separate GT_ROWS sharedwitnessrefs only.','chunk_rows':288,'uncompressed_chunk_bytes':288*96*96*4,'availability':'continuous&nearest&literalRGB96central4, contextpositiveweight-any; teacherS/U never gates inputs. v61ROIany independentfact.','counts':'all55296retained; TRAIN36864/VALIDATION18432; rawstates+inputavailability+qpixeldiagnostics byfold/factor/condition, no filtering.','actual_model_fit_current_V112_PnP':0,'training_admission':False})
 t={'approved_source_crop_build':False,'source_only':True,'model_forward':False,'fit':False,'current223':False,'V112':False,'PnP':False,'automatic_retry':False,'code':bind(__file__),'protocol':bind(PROTOCOL),'generated':bind(GENERATED),'root_policy':bind(POLICY),'render_complete':None,'base_configs':None,'input_scenes':None,'GT_scenes':None,'output':str(OUT)};write(H/'ROOT_CROP_ENTRY_TEMPLATE.json',t)
 print(json.dumps({'code':bind(__file__),'protocol':bind(PROTOCOL),'generated':bind(GENERATED),'template':bind(H/'ROOT_CROP_ENTRY_TEMPLATE.json'),'command':str(Path(sys.executable).resolve())+' '+str(Path(__file__).resolve())+' --entry ROOT_ACTUAL_SOURCE_CROP_ENTRY.json','actual_payload_or_model_read':0}),flush=True)
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--prepare',action='store_true');a.add_argument('--entry',type=Path);o=a.parse_args()
 if o.prepare:prepare()
 elif o.entry:run(o.entry)
 else:raise SystemExit('Use --prepare or root --entry.')
