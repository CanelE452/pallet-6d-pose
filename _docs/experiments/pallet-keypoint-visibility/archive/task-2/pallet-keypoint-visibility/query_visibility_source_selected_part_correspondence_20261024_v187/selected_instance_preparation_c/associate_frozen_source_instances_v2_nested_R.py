"""Source TRAIN only, frozen detection-box to provided-instance metadata diagnostic.
Preparation: existing input-only metadata + code and invented rectangles only.
Actual GT/camera/GT_info JSON body reads require a separately bound root entry.
No pixels/NPZ headers/poses fitting/native permutation/labels/model calls.
"""
from pathlib import Path
from collections import Counter
import argparse, hashlib, json, math, os, stat, traceback

H=Path(__file__).absolute().parent;V=H.parent;P=V.parent
V181=P/'query_visibility_local_support_raw_head_20261024_v181'
V153=P/'query_visibility_mr6d_full_public_source_20261024_v153'
V185=P/'query_visibility_public_part_supervision_readiness_20261024_v185'
FRAMES=V181/'sam_input_preparation_c/PUBLIC185_FRAMES_METADATA_ONLY.jsonl'
SLOTS=V181/'sam_input_preparation_c/PUBLIC1480_ORIGINAL_INPUTS_METADATA_ONLY.jsonl'
ASSETS=V185/'local_train_asset_inventory_b/MR185_SOURCE_TRAIN_FRAME_PATHS.jsonl'
ASSET_INDEX=V185/'local_train_asset_inventory_b/LOCAL_SOURCE_TRAIN_ASSET_STAT_INVENTORY.jsonl'
R0_SEAL=V153/'actual_full230_initializer_v1/PREDICTION_SEAL.json'
PROPOSALS=V153/'actual_full230_initializer_v1/FROZEN_PROPOSALS_ALL230_INPUT_ONLY.jsonl'
SOURCE_CODES=[V181/'sam_input_preparation_c/build_public_source_manifest.py',V153/'initializer_feature_preparation/encode_full230.py',V153/'native_reference_preparation/project_all_source_native_references.py']
PROTOCOL=H/'PROTOCOL_BEFORE_TRAIN_INSTANCE_METADATA_BODIES_V2_NESTED_R.json'
CHECKS=H/'GENERATED_INVENTED_CHECKS_V2_NESTED_R.json'
FRAME_PLAN=H/'ALL185_FROZEN_DETECTION_AND_ASSET_BINDINGS.jsonl'
ROW_PLAN=H/'ALL1480_ORIGINAL_SOURCE_INPUTS_ONLY.jsonl'
OUT=V/'selected_instance_metadata_diagnostic_v2_nested_R'
SCENES={'000000','000002','000005','000006'}

def require(value,message):
    if not bool(value):raise AssertionError(message)
def permitted(path):
    p=Path(path)
    # Check names lexically BEFORE stat/open/resolve. Never resolve symlinks.
    require(p.is_absolute() and '..' not in p.parts,'Absolute explicit path')
    require(all('v112' not in s.casefold() for s in p.parts),'Forbidden V112 name')
    require(p==P or P in p.parents,'Outside source project')
    return p
def open_safe(path):
    p=permitted(path);parts=p.relative_to(P).parts
    fd=os.open(P,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:
        for part in parts[:-1]:
            nd=os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd);os.close(fd);fd=nd
        final=os.open(parts[-1],os.O_RDONLY|os.O_NOFOLLOW,dir_fd=fd)
    finally:os.close(fd)
    require(stat.S_ISREG(os.fstat(final).st_mode),'Ordinary file only')
    return os.fdopen(final,'rb')
def sha(path):
    h=hashlib.sha256()
    with open_safe(path) as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def bind(path):return {'path':str(permitted(path)),'sha256':sha(path)}
def read(path):
    with open_safe(path) as f:return json.load(f)
def lines(path):
    with open_safe(path) as f:return [json.loads(s) for s in f if s.strip()]
def checked(b):
    require(isinstance(b.get('sha256'),str) and len(b['sha256'])==64,'Actual metadata SHA required')
    require(sha(b['path'])==b['sha256'],'Metadata binding drift: '+b['path']);return Path(b['path'])
def write(path,obj):
    with Path(path).open('x') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def writelines(path,rows):
    with Path(path).open('x') as f:
        for r in rows:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
def valid_box(box):return isinstance(box,(list,tuple)) and len(box)==4 and all(isinstance(v,(int,float)) and math.isfinite(v) for v in box) and box[2]>box[0] and box[3]>box[1]
def xywh_box(box):
    if not isinstance(box,(list,tuple)) or len(box)!=4:return None
    x,y,w,h=box
    if not all(isinstance(v,(int,float)) and math.isfinite(v) for v in box) or w<=0 or h<=0:return None
    return [float(x),float(y),float(x+w),float(y+h)]
def iou(a,b):
    if not valid_box(a) or not valid_box(b):return None
    intersection=max(0.,min(a[2],b[2])-max(a[0],b[0]))*max(0.,min(a[3],b[3])-max(a[1],b[1]))
    union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection
    return intersection/union
def associate(det_box,instances):
    """Exact FP64 IoU max across ALL supplied GT entries; no fitted cutoff/tie break."""
    candidates=[]
    for inst in instances:
        b=xywh_box(inst.get('bbox_obj'))
        candidates.append({'gt_instance_index':inst['gt_instance_index'],'obj_id':inst['obj_id'],'provided_bbox_obj_xywh':inst.get('bbox_obj'),'provided_bbox_obj_xyxy':b,'IoU':iou(det_box,b),'provided_bbox_visib_xywh':inst.get('bbox_visib'),'visible_bbox_IoU_diagnostic_only':iou(det_box,xywh_box(inst.get('bbox_visib')))})
    known=[c for c in candidates if c['IoU'] is not None]
    ranked=sorted(known,key=lambda c:(-c['IoU'],c['gt_instance_index']))
    result={'status':'UNKNOWN_NO_VALID_GT_BOX','selected_gt_instance_index':None,'selected_obj_id':None,'selected_IoU':None,'top_IoU':None,'second_IoU':None,'top_minus_second':None,'top_tied_indices':[],'all_GT_bbox_candidates':candidates,'association_is_weak_identity_diagnostic':True,'selected_instance_identity_certified':False}
    if not valid_box(det_box):result['status']='UNKNOWN_INVALID_OR_NO_DETECTION_BOX';return result
    if not ranked:return result
    top=ranked[0]['IoU'];second=ranked[1]['IoU'] if len(ranked)>1 else None
    ties=[c['gt_instance_index'] for c in ranked if c['IoU']==top]
    result.update(top_IoU=top,second_IoU=second,top_minus_second=None if second is None else top-second,top_tied_indices=ties)
    if top<=0:result['status']='UNKNOWN_NO_POSITIVE_BBOX_OVERLAP'
    elif len(ties)>1:result['status']='UNKNOWN_EXACT_MAX_IOU_TIE'
    else:result.update(status='UNIQUE_POSITIVE_MAX_BBOX_IOU_WEAK_IDENTITY',selected_gt_instance_index=ranked[0]['gt_instance_index'],selected_obj_id=ranked[0]['obj_id'],selected_IoU=top)
    return result

def valid_provided_R(value):
    """Accept only original finite flat9 or nested3x3 notation; never modify it."""
    def finite_scalar(x):return isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x)
    if not isinstance(value,(list,tuple)):return False
    if len(value)==9:return all(finite_scalar(x) for x in value)
    return len(value)==3 and all(isinstance(row,(list,tuple)) and len(row)==3 and all(finite_scalar(x) for x in row) for row in value)

def invented():
    def inst(i,box,obj=1):return {'gt_instance_index':i,'obj_id':obj,'bbox_obj':box,'bbox_visib':box}
    checks=[]
    checks.append(('xywh_halfopen_no_plus1',xywh_box([-2,3,5,7])==[-2.,3.,3.,10.]))
    checks.append(('exact_single_IoU',iou([0.,0.,10.,10.],[0.,0.,10.,10.])==1.))
    r=associate([0.,0.,10.,10.],[inst(4,[0,0,10,10]),inst(1,[20,20,2,2])]);checks.append(('unique_original_GT_index_not_list_rank',r['selected_gt_instance_index']==4))
    r=associate([0.,0.,10.,10.],[inst(4,[0,0,10,10]),inst(1,[0,0,10,10])]);checks.append(('tie_UNKNOWN_no_lowest_index_choice',r['selected_gt_instance_index'] is None and r['status']=='UNKNOWN_EXACT_MAX_IOU_TIE'))
    r=associate([0.,0.,1.,1.],[inst(0,[2,2,1,1])]);checks.append(('no_overlap_UNKNOWN',r['selected_gt_instance_index'] is None and r['status']=='UNKNOWN_NO_POSITIVE_BBOX_OVERLAP'))
    r=associate([0.,0.,10.,10.],[inst(0,[0,0,2,2]),inst(3,[0,0,10,10],obj=7)]);checks.append(('do_not_force_obj1_when_other_best',r['selected_obj_id']==7 and len(r['all_GT_bbox_candidates'])==2))
    checks.append(('invalid_box_UNKNOWN',associate(None,[inst(0,[0,0,1,1])])['status']=='UNKNOWN_INVALID_OR_NO_DETECTION_BOX'))
    checks.append(('boundary_touch_zero_area',iou([0.,0.,1.,1.],[1.,0.,2.,1.])==0.))
    original=[{'native_slot':k,'q':[float(k)+.25,-.5],'L':12.5,'available':k<6} for k in range(8)]
    associated=[dict(row,association={'status':'UNKNOWN_EXACT_MAX_IOU_TIE'}) for row in original]
    checks.append(('all8_U_and_qL_preserved',len(associated)==8 and all({k:v for k,v in z.items() if k!='association'}==r for z,r in zip(associated,original)) and sum(not r['available'] for r in associated)==2))
    try:permitted(P/'do_not_stat_v112'/'nonexistent');ok=False
    except AssertionError:ok=True
    checks.append(('V112_lexical_reject_before_stat',ok))
    nested=[[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]];flat=[1.,0.,0.,0.,1.,0.,0.,0.,1.]
    checks.append(('provided_R_nested3x3_finite_accepted',valid_provided_R(nested)))
    checks.append(('provided_R_flat9_finite_still_accepted',valid_provided_R(flat)))
    before=json.dumps(nested,separators=(',',':'));inst={'provided_cam_R_m2c':nested};valid_provided_R(inst['provided_cam_R_m2c'])
    checks.append(('provided_R_literal_structure_values_not_rewritten',json.dumps(inst['provided_cam_R_m2c'],separators=(',',':'))==before and inst['provided_cam_R_m2c'] is nested))
    invalids=[[[1.,0.],[0.,1.],[0.,0.]],[[1.,0.,0.],[0.,1.,0.]],flat[:8],[[1.,0.,0.],[0.,float('nan'),0.],[0.,0.,1.]],['1']*9,[True]*9]
    checks.append(('provided_R_invalid_shape_nonfinite_nonnumeric_rejected',all(not valid_provided_R(x) for x in invalids)))
    return {'completed':True,'PASS':all(v for _,v in checks),'invented_only':True,'checks':[{'name':k,'PASS':v} for k,v in checks],'actual_source_GT_RGB_NPZ_or_payload_reads':0}

def prepare():
    # Reuse immutable V1 metadata plans and the already-root-bound exact 12 assets.
    # No source GT/camera/info/pixel/NPZ body or payload hash is read here.
    import ast
    old_protocol_path=H/'PROTOCOL_BEFORE_TRAIN_INSTANCE_METADATA_BODIES.json'
    old_code_path=H/'associate_frozen_source_instances.py'
    old_checks_path=H/'GENERATED_INVENTED_CHECKS.json'
    old_entry_path=V/'ROOT_ACTUAL_SELECTED_INSTANCE_METADATA_ENTRY.json'
    old=read(old_protocol_path);old_entry=read(old_entry_path)
    require(old['code']==bind(old_code_path),'V1 code unchanged')
    for key in ('frame_plan','row_plan','checks'):require(old_entry[key]==old[key],'Original immutable plan '+key);checked(old[key])
    def functions(path):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(Path(path).read_text()).body if isinstance(n,ast.FunctionDef)}
    before,after=functions(old_code_path),functions(__file__)
    unchanged=['valid_box','xywh_box','iou','associate']
    require(all(before[k]==after[k] for k in unchanged),'Matcher/box/IoU AST unchanged')
    checks=invented();original=read(old_checks_path)['checks']
    require(checks['checks'][:10]==original and len(checks['checks'])==14,'Original10+fournotationchecks unchanged')
    checks.update(code=bind(__file__),original_checks=bind(old_checks_path),original_matcher_functions_AST_exact=True,original_R_values_or_structure_modified=False)
    write(CHECKS,checks);require(checks['PASS'],'Invented fixture PASS')
    p=dict(old);p.update(code=bind(__file__),checks=bind(CHECKS),source_scene_metadata=old_entry['source_scene_metadata'],output=str(OUT),parent_V1_protocol=bind(old_protocol_path),original_failed_V1_entry=bind(old_entry_path),root_scope=old_entry['root_scope'],notation_adapter={'allowed_R':'original flat9 or nested3x3, all numeric finite','allowed_t':'original flat3 numeric finite','mutation':'none; g.cam_R_m2c and g.cam_t_m2c copied literally as provided; no reshape/repair/orthogonalization/math change','scope':'R notation validation only; source assets, GT values, IoU/matcher and native/query metadata unchanged'},GT_info_hash_authority='All12 exact provided metadata bindings inherited from root-approved V1 entry; four GT_info SHA now resolved, no source body re-read in V2 preparation.')
    write(PROTOCOL,p)
    template=dict(old_entry);template.update(approved_actual_source_instance_metadata_diagnostic=False,code=p['code'],protocol=bind(PROTOCOL),checks=p['checks'],output=str(OUT))
    write(H/'ROOT_ACTUAL_ENTRY_TEMPLATE_V2_NESTED_R_APPROVED_FALSE.json',template)
    ready={'completed':True,'prepared_only':True,'code':p['code'],'protocol':bind(PROTOCOL),'checks':p['checks'],'frame_plan':p['frame_plan'],'row_plan':p['row_plan'],'template':bind(H/'ROOT_ACTUAL_ENTRY_TEMPLATE_V2_NESTED_R_APPROVED_FALSE.json'),'counts':p['expected'],'source_asset_values_or_matcher_changed':False,'original_V1_preserved':True,'actual_source_body_reads_or_hashes_in_V2_preparation':0,'automatic_restart':False}
    write(H/'PREPARATION_COMPLETE_V2_NESTED_R.json',ready);print(json.dumps(ready),flush=True)

def run(entry):
    e=read(entry);require(e['approved_actual_source_instance_metadata_diagnostic'] is True and e['source_TRAIN_only'] is True,'Root actual entry required')
    for key in ('RGB_depth_mask_mesh_NPZ_or_model_reads','label_or_training_admission','automatic_retry'):require(e[key] is False,'No extra action '+key)
    p=read(checked(e['protocol']));require(e['protocol']==bind(PROTOCOL) and e['code']==p['code']==bind(__file__),'Exact frozen code/protocol')
    for key in ('frame_plan','row_plan','checks'):require(e[key]==p[key],'Entry binding '+key);checked(p[key])
    scope=read(checked(e['root_scope']));require(bool(scope),'Bound root scope')
    scene_bindings=e['source_scene_metadata'];require(set(scene_bindings)==SCENES,'Exactly four TRAIN scenes')
    for s in SCENES:
        require(set(scene_bindings[s])=={'GT','camera','GT_info'},'Explicit12 metadata files')
        for k,b in scene_bindings[s].items():
            require(b['path']==p['source_scene_metadata'][s][k]['path'],'No replacement asset')
            old=p['source_scene_metadata'][s][k]['sha256']
            require(old is None or b['sha256']==old,'Inherited metadata SHA unchanged')
    require(e['output']==p['output']==str(OUT),'Fixed output')
    OUT.mkdir(exist_ok=False);write(OUT/'STARTED.json',{'completed':False,'entry':bind(entry),'automatic_retry':False})
    try:
        frames=lines(checked(p['frame_plan']));slots=lines(checked(p['row_plan']));source={}
        for s,b in scene_bindings.items():source[s]={k:read(checked(v)) for k,v in b.items()}
        frame_rows=[];allinstances=[];byfi={}
        for frame in frames:
            s,key=frame['scene'],frame['frame_key'];raw=source[s];gt=raw['GT'][key];info=raw['GT_info'][key];camera=raw['camera'][key]
            require(len(gt)==len(info),'Original GT and info instance-order alignment')
            require([i for i,g in enumerate(gt) if g['obj_id']==1]==frame['all_obj1_GT_instance_indices'],'All original obj1 observations preserved')
            require(len(camera['cam_K'])==9 and all(math.isfinite(v) for v in camera['cam_K']),'Provided K finite9')
            full={r['gt_instance_index']:r['binding'] for r in frame['source_asset_bindings']['full_masks']};vis={r['gt_instance_index']:r['binding'] for r in frame['source_asset_bindings']['visible_masks']}
            instances=[]
            for ix,(g,inf) in enumerate(zip(gt,info)):
                require(valid_provided_R(g['cam_R_m2c']) and isinstance(g['cam_t_m2c'],(list,tuple)) and len(g['cam_t_m2c'])==3 and all(isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v) for v in g['cam_t_m2c']),'Provided original R flat9 or nested3x3 and flat t3 finite notation')
                inst={'frame_index':frame['frame_index'],'scene':s,'frame_key':key,'gt_instance_index':ix,'obj_id':g['obj_id'],'bbox_obj':inf.get('bbox_obj'),'bbox_visib':inf.get('bbox_visib'),'provided_cam_K':camera['cam_K'],'provided_cam_R_m2c':g['cam_R_m2c'],'provided_cam_t_m2c_mm':g['cam_t_m2c'],'provided_depth_scale':camera.get('depth_scale'),'full_mask_binding':full.get(ix),'visible_mask_binding':vis.get(ix),'GT_mask_payload_read':False,'camera_GT_binding':scene_bindings[s],'native_reference_or_label_generated':False}
                instances.append(inst);allinstances.append(inst)
            assoc=associate(frame['original_selected_detection_bbox_xyxy'],instances)
            selected=next((r for r in instances if r['gt_instance_index']==assoc['selected_gt_instance_index']),None)
            assoc.update(selected_obj1_mask_paths_available=selected is not None and selected['obj_id']==1 and selected['full_mask_binding'] is not None and selected['visible_mask_binding'] is not None,selected_instance_metadata=selected)
            result=dict(frame,diagnostic_association=assoc,all_GT_instance_count=len(instances),new_training_admission=False)
            frame_rows.append(result);byfi[frame['frame_index']]=result
        outrows=[]
        for r in slots:
            original=dict(r);new=dict(r,selected_instance_diagnostic=byfi[r['original_frame_index']]['diagnostic_association'],new_native_visibility_target=-1,new_point_quality_target=-1,new_end_support_target=[-1,-1],new_training_admission=False)
            require(all(new[k]==v for k,v in original.items()),'Original row invariants');outrows.append(new)
        require(len(frame_rows)==185 and len(outrows)==1480 and sum(not r['original_query_available'] for r in outrows)==194,'All original denominators')
        require(sum(r['obj_id']==1 for r in allinstances)==258,'All258 obj1 instance observations')
        writelines(OUT/'ALL185_FRAME_INSTANCE_ASSOCIATIONS.jsonl',frame_rows);writelines(OUT/'ALL1480_QUERY_INSTANCE_DIAGNOSTICS.jsonl',outrows);writelines(OUT/'ALL_GT_OBSERVATIONS_IN185_SOURCE_FRAMES.jsonl',allinstances)
        complete={'completed':True,'PASS':True,'entry':bind(entry),'code':bind(__file__),'protocol':bind(PROTOCOL),'counts':{'frames':185,'rows':1480,'inputU':194,'input_available':1286,'all_GT_instance_observations':len(allinstances),'obj1_GT_instance_observations':258,'frame_association_states':dict(Counter(r['diagnostic_association']['status'] for r in frame_rows)),'selected_obj_ids':dict(Counter(str(r['diagnostic_association']['selected_obj_id']) for r in frame_rows))},'source_scene_metadata':scene_bindings,'outputs':{n:bind(OUT/n) for n in ['ALL185_FRAME_INSTANCE_ASSOCIATIONS.jsonl','ALL1480_QUERY_INSTANCE_DIAGNOSTICS.jsonl','ALL_GT_OBSERVATIONS_IN185_SOURCE_FRAMES.jsonl']},'new_labels_or_training_admission':False,'physical_or_native_identity_certification':False,'RGB_depth_mask_mesh_NPZ_body_or_header_reads':0,'model_initializer_fit_PnP_qL_changes':0,'automatic_retry':False}
        write(OUT/'COMPLETE.json',complete);print(json.dumps({'completed':True,'PASS':True,'counts':complete['counts'],'complete':bind(OUT/'COMPLETE.json')}),flush=True)
    except Exception:
        write(OUT/'FAILED_EXECUTION_PRESERVED.json',{'completed':False,'automatic_retry':False,'traceback':traceback.format_exc(),'entry':bind(entry)});raise

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');ap.add_argument('--entry',type=Path);args=ap.parse_args()
    if args.prepare:prepare()
    elif args.entry:run(args.entry)
    else:ap.error('--prepare or --entry required')
