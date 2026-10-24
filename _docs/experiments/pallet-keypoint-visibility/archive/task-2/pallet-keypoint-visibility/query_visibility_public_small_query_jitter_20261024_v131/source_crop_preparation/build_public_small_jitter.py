"""V131 input-only native-pixel small-jitter crop builder; root entry only.

Preparation locks all918 q8/L/context rows from parent centre INPUT metadata.
No RGB or GT teacher body/hash is read until actual --entry and lock replay.
"""
from pathlib import Path
import argparse,ast,collections,copy,hashlib,json,sys,time,traceback
import numpy as np
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;V=H.parent;P=V.parent
POLICY=V/'ROOT_SMALL_QUERY_DIAGNOSIS_POLICY_BEFORE_INPUTS.json'
PROTOCOL=H/'CROP_PROTOCOL_BEFORE_PAYLOADS.json';LOCK=H/'QUERY_INPUT_LOCK918_BEFORE_RGB_GT.json';OUT=V/'source_crops_v1'
OPS={'local':P/'query_visibility_public_cue_inputs_20261023_v90/producer/spatial_input_math.py','side':P.parent/'external_ray_proxy_head_20261022_v60/head_and_roi.py','context':P/'query_visibility_target_context_rgb_20261023_v107/target_context_math.py'}
OFFSETS=[(dx,dy)for dy in (-3.,0.,3.)for dx in (-3.,0.,3.)]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return {'path':str(Path(p).resolve()),'sha256':sha(p)}
def check(b):assert bind(b['path'])==b,b['path'];return Path(b['path'])
def read(p):return json.loads(Path(p).read_text())
def write(p,x):
 with Path(p).open('x')as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def ash(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def pure(path,names):
 nodes=[n for n in ast.parse(Path(path).read_text()).body if isinstance(n,ast.FunctionDef)and n.name in names]
 assert {n.name for n in nodes}==set(names);ns={'np':np};exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(path)+'::pure','exec'),ns);return ns
def operators():
 x=pure(OPS['local'],('crop','query_validity'));x.update(pure(OPS['side'],('native_grid',)));x.update(pure(OPS['context'],('query_extent','valid_box','target_context_grid')));return x
def context_any(box,hw,op):
 if box is None:return False
 g=op['target_context_grid'](box);x,y=g[...,0],g[...,1];ix=np.floor(x).astype(int);iy=np.floor(y).astype(int);dx=x-ix;dy=y-iy;v=np.ones(x.shape,bool);h,w=hw
 for ox,oy,weight in ((0,0,(1-dx)*(1-dy)),(1,0,dx*(1-dy)),(0,1,(1-dx)*dy),(1,1,dx*dy)):xx=ix+ox;yy=iy+oy;v&=((xx>=0)&(xx<w)&(yy>=0)&(yy<h))|(weight==0)
 return bool(v.any())
def condition(center,offset,hw,op):
 q8=np.array(center,np.float64,copy=True);assert q8.shape==(8,2)
 good=np.isfinite(q8[0]).all()and not np.array_equal(q8[0],[-1.,-1.])
 if good:q8[0]+=np.array(offset,np.float64)
 else:q8[0]=[-1.,-1.]
 _,rv,L=op['native_grid'](q8,hw);box=op['query_extent'](q8)
 return {'initial_points':q8.tolist(),'query_xy':q8[0].tolist(),'L':float(L),'context_box_xyxy':box,'ROI_any_valid':bool(rv[0].any()),'context_available_geometry':context_any(box,hw,op)}
def make_plan(parent,binding,op):
 assert parent['completed']and len(parent['rows'])==918;old=parent['rows'];assert [r['occurrence_index']for r in old]==list(range(918))
 rows=[];drop={'rgb96_chunk','crop_chunk','rgb96_row','crop_row','rgb96_sha256','valid96_sha256','q_native_continuous_valid','q_nearest_valid','local_central4_valid','factual_query_available','factual_available','context_available','query_available','original_input_availability','unavailability_reasons'}
 for cue in range(6):
  for ci,offset in enumerate(OFFSETS):
   for vi in range(17):
    i=cue*153+ci*17+vi;pc=cue*153+4*17+vi;c=old[pc];wide=old[i]
    assert c['condition_index']==4 and c['carrier_slot']==0 and c['scene_id']==wide['scene_id']and c['full_RGB']==wide['full_RGB']and c['split']==wide['split']
    info=condition(c['initial_points'],offset,c['raw_hw'],op);r={k:copy.deepcopy(v)for k,v in c.items()if k not in drop}
    identity=f'V131_PUBLIC6:C{cue:02d}:J{ci}:R{vi:02d}';kind='CENTRE_0PX'if ci==4 else('AXIS_3PX'if offset[0]==0 or offset[1]==0 else'DIAGONAL_SQRT18PX')
    r.update(info);r.update(occurrence_index=i,row_id=i,id=identity,occurrence_id=identity,query_id=identity,query_group_id=f'V131_PUBLIC6_C{cue:02d}_J{ci}',condition_index=ci,jitter_index=ci,condition_id=f'NATIVE_PIXEL_DX{offset[0]:+.0f}_DY{offset[1]:+.0f}',condition_kind=kind,offset_xy=list(offset),normalized_offset_xy=[float(v/c['L'])for v in offset],q8=info['initial_points'],carrier_slot=0,logical_slot=0,query_index=0,parent_input=binding,parent_occurrence_index=i,parent_center_occurrence_index=pc,source_query_emulation=True,actual_initializer_output=False)
    assert r['initial_points'][1:]==c['initial_points'][1:]
    if ci==4:
     for k in ('query_xy','initial_points','L','context_box_xyxy'):assert r[k]==c[k]
    rows.append(r)
 assert len(rows)==918 and sum(r['split']=='TRAIN'for r in rows)==612 and sum(r['split']=='VALIDATION'for r in rows)==306
 return rows
def authorize(entry):
 e=read(entry);assert e['approved_public_small_jitter_crop_build']is True and e['source_only']is True
 assert all(e[k]is False for k in ('model_forward','fit','current223','V112','PnP','automatic_retry'))
 for k in ('code','protocol','generated','policy','query_lock'):check(e[k])
 p=read(PROTOCOL);assert e['code']==p['code']==bind(__file__)and e['protocol']==bind(PROTOCOL)and e['policy']==p['policy']==bind(POLICY)and e['query_lock']==p['query_lock']==bind(LOCK)and e['generated']==p['generated']
 assert read(e['generated']['path'])['PASS']and read(e['generated']['path'])['code']==e['code']
 for b in p['operators'].values():check(b)
 check(p['parent_complete']);check(p['parent_input']);done=read(p['parent_complete']['path']);assert done['completed']and done['PASS']and done['input_manifest']==p['parent_input']
 assert not OUT.exists(),'Preserve earlier completed/failed attempt, never automatically retry.'
 return e,p
def run(entry):
 e,p=authorize(entry);op=operators();plan=read(LOCK);rows=copy.deepcopy(plan['rows']);old=read(p['parent_input']['path'])['rows'];assert plan['completed']and len(rows)==918
 OUT.mkdir();(OUT/'chunks').mkdir();outlock=OUT/LOCK.name;outlock.write_bytes(LOCK.read_bytes());write(OUT/'STARTED.json',{'entry':bind(entry),'query_lock':bind(outlock),'RGB_or_GT_reads_before_query_lock_replay':0,'automatic_retry':False});start=time.monotonic()
 try:
  # This is the first parent teacher body/hash access in this builder.
  check(p['parent_GT']);parentgt=read(p['parent_GT']['path']);assert parentgt['completed']and len(parentgt['rows'])==918
  rgbs=collections.OrderedDict();parentcrop=collections.OrderedDict();checked=set();input_rows=[];gt_rows=[];chunks=[];a=np.empty((576,96,96,3),np.uint8);vv=np.empty((576,96,96),bool);ix=np.empty(576,np.int32);pending=[];centres=0
  def cached_rgb(r):
   b=r['full_RGB'];key=(b['path'],b['sha256'])
   if key not in rgbs:
    if key not in checked:check(b);checked.add(key)
    with np.load(b['path'],allow_pickle=False)as f:rgb=np.array(f['RGB_uint8'],copy=True)
    assert rgb.dtype==np.uint8 and rgb.shape==tuple(r['raw_hw'])+(3,)and ash(rgb)==r['decoded_RGB_sha256'];rgbs[key]=rgb
    if len(rgbs)>17:rgbs.popitem(last=False)
   rgbs.move_to_end(key);return rgbs[key]
  def old_crop(r):
   b=r['rgb96_chunk'];key=(b['path'],b['sha256'])
   if key not in parentcrop:
    check(b)
    with np.load(b['path'],allow_pickle=False)as f:parentcrop[key]=(np.array(f['rgb96']),np.array(f['valid96']),np.array(f['occurrence_index']))
    if len(parentcrop)>2:parentcrop.popitem(last=False)
   parentcrop.move_to_end(key);x,v,n=parentcrop[key];j=r['rgb96_row'];assert n[j]==r['occurrence_index']and ash(x[j])==r['rgb96_sha256']and ash(v[j])==r['valid96_sha256'];return x[j],v[j]
  for i,r in enumerate(rows):
   rgb=cached_rgb(r);q=np.array(r['query_xy']);crop,valid=op['crop'](rgb,q,r['L'],96);c=crop.transpose(1,2,0).copy();v=valid[0].copy();center,continuous,near=op['query_validity'](q[None],r['raw_hw']);c4=bool(v[47:49,47:49].all());context=context_any(r['context_box_xyxy'],r['raw_hw'],op);factual=bool(center[0]and c4);available=factual and context
   r.update(q_native_continuous_valid=bool(continuous[0]),q_nearest_valid=bool(near[0]),local_central4_valid=c4,factual_query_available=factual,factual_available=factual,context_available=context,query_available=available,original_input_availability={'continuous':bool(continuous[0]),'nearest':bool(near[0]),'ROI_any_valid':r['ROI_any_valid'],'central4':c4,'context':context},rgb96_sha256=ash(c),valid96_sha256=ash(v),rgb96_layout='HWC',valid96_layout='HW')
   pc=r['parent_center_occurrence_index'];og=parentgt['rows'][pc];assert og['id']==old[pc]['id']and og['scene_id']==r['scene_id']and og['error_px']==0.and np.array_equal(old[pc]['query_xy'],og['reference_xy'])
   g=copy.deepcopy(og);delta=q-np.array(og['reference_xy']);error=float(np.linalg.norm(delta));g.update(row_id=i,occurrence_index=i,id=r['id'],scene_id=r['scene_id'],jitter_index=r['condition_index'],condition_index=r['condition_index'],condition_kind=r['condition_kind'],parent_GT=p['parent_GT'],parent_center_occurrence_index=pc,parent_center_id=og['id'],delta_xy=delta.tolist(),error_px=error,error_bin='CENTRE_0PX'if error==0 else('AXIS_3PX'if abs(error-3)<1e-12 else'DIAGONAL_SQRT18PX'),nearest_q_inside_image_diagnostic=bool(near[0]),parent_center_nearest_q_foreground_alpha_diagnostic=og.get('nearest_q_foreground_alpha_diagnostic'),nearest_q_foreground_alpha_diagnostic=None,nearest_q_alpha_status='Parent centre diagnostic retained separately; new query alpha not independently replayed. This does not affect copied scene weak teacher.',source_query_emulation=True,actual_initializer_output=False)
   if r['condition_index']==4:
    prev=old[pc];oc,ov=old_crop(prev);assert np.array_equal(c,oc)and np.array_equal(v,ov)and r['rgb96_sha256']==prev['rgb96_sha256']and r['valid96_sha256']==prev['valid96_sha256']
    for k in ('initial_points','query_xy','L','context_box_xyxy','query_available','factual_query_available','context_available'):assert r[k]==prev[k],k
    centres+=1
   gt_rows.append(g);j=len(pending);a[j]=c;vv[j]=v;ix[j]=i;r['rgb96_row']=r['crop_row']=j;pending.append(r)
   if len(pending)==576 or i==917:
    path=OUT/'chunks'/f'QUERIES_{i-len(pending)+1:04d}_{i+1:04d}.npz';np.savez_compressed(path,rgb96=a[:len(pending)],valid96=vv[:len(pending)],occurrence_index=ix[:len(pending)]);b=bind(path);chunks.append(b)
    for item in pending:item['rgb96_chunk']=item['crop_chunk']=b;input_rows.append(item)
    pending=[];print(json.dumps({'rows_complete':i+1,'total':918,'elapsed_seconds':time.monotonic()-start}),flush=True)
  assert centres==102 and len(input_rows)==len(gt_rows)==918 and [r['occurrence_index']for r in input_rows]==list(range(918))
  ip=OUT/'INPUT_ROWS.json';gp=OUT/'GT_ROWS.json';write(ip,{'completed':True,'rows':input_rows,'crop_chunks':chunks,'query_lock':bind(outlock),'parent_input':p['parent_input'],'source_query_emulation_not_initializer':True,'training_admission':False});write(gp,{'completed':True,'rows':gt_rows,'input_row_identity':bind(ip),'parent_GT':p['parent_GT'],'teacher_copied_only_after_q_lock':True,'original_weak_uncertainty_unchanged':True,'native_mapping':'UNKNOWN','human_gold':False,'no_training_admission':True})
  summary={'completed':True,'rows':918,'scenes':102,'split_counts':dict(collections.Counter(r['split']for r in input_rows)),'teacher_counts':dict(collections.Counter(g['raw_state']for g in gt_rows)),'input_available':sum(r['query_available']for r in input_rows),'centre_rows_exact_parent':centres,'condition_kind_counts':dict(collections.Counter(g['condition_kind']for g in gt_rows)),'native_error_values':sorted(set(g['error_px']for g in gt_rows)),'all918_are_not_within3EuclideanPx':True,'weak_region_identity_not_qpixel_label':True,'no_GT_input_U_gate':True};write(OUT/'SUMMARY.json',summary)
  done=OUT/'CROP_COMPLETE.json';write(done,{'completed':True,'PASS':True,'entry':bind(entry),'code':e['code'],'protocol':e['protocol'],'policy':e['policy'],'query_lock':bind(outlock),'parent_complete':p['parent_complete'],'parent_input':p['parent_input'],'parent_GT':p['parent_GT'],'input_manifest':bind(ip),'GT_only':bind(gp),'crop_chunks':chunks,'summary':bind(OUT/'SUMMARY.json'),'rows':918,'split_counts':summary['split_counts'],'centre102_q8_q_L_context_RGB96_valid_exact_parent':True,'full_RGB_references_unchanged':True,'no_source_native_labels_created':True,'training_admission':False,'models_fit_current_V112_PnP':0,'seconds':time.monotonic()-start});print(json.dumps({'completed':True,'receipt':bind(done),'rows':918,'centre_parity':102,'seconds':time.monotonic()-start}),flush=True)
 except BaseException:write(OUT/'FAILED_NO_RESTART.json',{'traceback':traceback.format_exc(),'automatic_retry':False});raise
def prepare():
 policy=read(POLICY);assert bind(POLICY)['sha256']=='1ddb6f69270478089e6cd89e23c9476154c58b6e127769c6a3ebcec076999c65';parent=policy['parent_source'];check(parent['complete']);check(parent['inputs']);op=operators();old=read(parent['inputs']['path']);rows=make_plan(old,parent['inputs'],op)
 # Invented math checks exercise extent changes, missing carrier and error norms.
 q=np.array([[100.,80.],[0.,0.],[100.,0.],[100.,80.],[0.,80.],[50.,0.],[50.,80.],[0.,40.]])
 c=condition(q,(3.,3.),[256,256],op);assert c['initial_points'][1:]==q[1:].tolist()and c['query_xy']==[103.,83.]and c['L']==25.75 and c['context_box_xyxy']==[0.,0.,103.,83.]
 missing=q.copy();missing[0]=[-1.,-1.];assert condition(missing,(3.,3.),[256,256],op)['query_xy']==[-1.,-1.]
 assert [float(np.linalg.norm(d))for d in OFFSETS]==[np.sqrt(18),3.,np.sqrt(18),3.,0.,3.,np.sqrt(18),3.,np.sqrt(18)]
 write(LOCK,{'completed':True,'rows':rows,'parent_input':parent['inputs'],'policy':bind(POLICY),'source':'Only parent centre condition4 input q8, noGT/RGB read. Carrier0 native-pixel offsets, other7 unchanged.','parent_GT_body_or_hash_read':False})
 g=H/'GENERATED_QUERY_CHECKS.json';write(g,{'completed':True,'PASS':True,'code':bind(__file__),'checks':{'actual_parent918_input_order102centres':True,'carrier0_only_and_extent_recompute':True,'sentinel_carrier_unchanged':True,'axis3_diagonal_sqrt18_actualnorm':True,'centre102_input_metadata_exact_parent':True,'TR612_VAL306_preserved':True,'all918_lock_JSON_serializable':True},'actual_RGB_GT_teacher_model_weight_current_V112_reads':0})
 write(PROTOCOL,{'completed':True,'code':bind(__file__),'generated':bind(g),'policy':bind(POLICY),'query_lock':bind(LOCK),'parent_complete':parent['complete'],'parent_input':parent['inputs'],'parent_GT':parent['GT'],'operators':{k:bind(v)for k,v in OPS.items()},'output':str(OUT),'order':'cue6→condition9→scenevariant17','offsets_native_pixels':OFFSETS,'input_schema':'SameV129aliases,rgb96_chunk/localrgb96_row,HWC/HW/globaloccurrence_index; parent_input+parent_occurrence_index+parent_center_occurrence_index.','GT_boundary':'Preparation/lock uses parentINPUT only; rootactualGT hash/body after qlockreplay. Copy same scene weakstate/unc from parentcentre; recomputeactualerror; parentcentre alpha retainedseparately,newalpha unknown.','parity':'All102centre rows exact q8/q/L/context/RGB96/valid against boundV129centre metadata/crop; no tolerance.','root_entry_required':True,'actual_model_RGB_GT_calls_by_preparation':0})
 template={'approved_public_small_jitter_crop_build':False,'source_only':True,'model_forward':False,'fit':False,'current223':False,'V112':False,'PnP':False,'automatic_retry':False,'code':bind(__file__),'protocol':bind(PROTOCOL),'generated':bind(g),'policy':bind(POLICY),'query_lock':bind(LOCK)};write(H/'ROOT_CROP_ENTRY_TEMPLATE.json',template);print(json.dumps({'code':bind(__file__),'protocol':bind(PROTOCOL),'generated':bind(g),'template':bind(H/'ROOT_CROP_ENTRY_TEMPLATE.json'),'query_lock':bind(LOCK),'command':str(Path(sys.executable).resolve())+' '+str(Path(__file__).resolve())+' --entry ROOT_ACTUAL_SMALL_JITTER_CROP_ENTRY.json','actual_RGB_GT_model_execution':0}),flush=True)
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--prepare',action='store_true');a.add_argument('--entry',type=Path);o=a.parse_args()
 if o.prepare:prepare()
 elif o.entry:run(o.entry)
 else:raise SystemExit('Use --prepare or root --entry.')
