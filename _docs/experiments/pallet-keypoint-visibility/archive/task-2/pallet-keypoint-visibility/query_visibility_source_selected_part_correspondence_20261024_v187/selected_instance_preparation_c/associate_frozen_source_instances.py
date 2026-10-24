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
PROTOCOL=H/'PROTOCOL_BEFORE_TRAIN_INSTANCE_METADATA_BODIES.json'
CHECKS=H/'GENERATED_INVENTED_CHECKS.json'
FRAME_PLAN=H/'ALL185_FROZEN_DETECTION_AND_ASSET_BINDINGS.jsonl'
ROW_PLAN=H/'ALL1480_ORIGINAL_SOURCE_INPUTS_ONLY.jsonl'
OUT=V/'selected_instance_metadata_diagnostic_v1'
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
    return {'completed':True,'PASS':all(v for _,v in checks),'invented_only':True,'checks':[{'name':k,'PASS':v} for k,v in checks],'actual_source_GT_RGB_NPZ_or_payload_reads':0}

def prepare():
    frames=lines(FRAMES);slots=lines(SLOTS);assets=lines(ASSETS)
    require(len(frames)==len(assets)==185 and len(slots)==1480,'Original TRAIN185/1480')
    require(sum(not r['original_query_available'] for r in slots)==194,'Original U194')
    require(all(r['split']=='SOURCE_DEVELOPMENT' for r in frames+slots),'No held metadata rows')
    byfi={r['original_frame_index']:r for r in frames};byasset={r['frame_index']:r for r in assets}
    require(len(byfi)==185 and set(byfi)==set(byasset),'Exact frame ID inventory')
    require(sum(len(r['all_obj1_GT_instance_indices']) for r in assets)==258,'Original all258 obj1 instance observations')
    seal=read(R0_SEAL);proposal_binding=next(b for b in seal['outputs'] if Path(b['path']).name==PROPOSALS.name)
    require(proposal_binding['path']==str(PROPOSALS),'Exact frozen proposal authority')
    # Shared input-only JSONL container: only explicit TRAIN line indices are decoded.
    # No held RGB/NPZ/GT/semantic label or new score file is read.
    props={}
    with open_safe(PROPOSALS) as f:
        for line_index,line in enumerate(f):
            if line_index not in byfi:continue
            r=json.loads(line);require(r['frame_index']==line_index and r['split']=='SOURCE_DEVELOPMENT','Frozen proposal order/split');props[line_index]=r
    require(set(props)==set(byfi),'All185 frozen detections')
    groups=[]
    for fi in sorted(byfi):
        fr=byfi[fi];asset=byasset[fi];pr=props[fi];rr=[r for r in slots if r['original_frame_index']==fi]
        require(len(rr)==8 and [r['native_slot'] for r in rr]==list(range(8)),'All native8 original order')
        require(asset['scene'] in SCENES and asset['internal_split']=='SOURCE_DEVELOPMENT','Explicit four TRAIN sequences')
        require(fr['frame_id']==pr['id']==asset['id'] and fr['source_id']==pr['source_id']==asset['source_id'],'Frame identity exact')
        ix=pr['selected_index'];selected=None if ix is None else pr['candidates'][ix]
        if selected is not None:
            require(selected['candidate_index']==ix and selected['keypoints_xy'][:8]==fr['original_initial_points8'],'Original selected candidate q8')
        for k,r in enumerate(rr):
            require(r['original_global_index']==fi*8+k and r['original_query_xy']==fr['original_initial_points8'][k] and r['original_L']==fr['original_L8'][k],'q/L/global/native exact')
            require(r['initializer_association_status']=='UNASSIGNED_BY_INPUT_PRODUCER','No original GT instance identity')
        groups.append({'frame_index':fi,'frame_id':fr['frame_id'],'source_id':fr['source_id'],'split':fr['split'],'scene':asset['scene'],'frame_key':str(asset['frame_key']),'raw_hw':fr['image']['raw_hw'],'original_selected_detection_candidate_index':ix,'original_selected_detection_bbox_xyxy':None if selected is None else selected['box_xyxy'],'all_original_detection_candidate_indices':[r['candidate_index'] for r in pr['candidates']],'original_global_indices':fr['original_global_indices'],'original_query_ids':fr['original_query_ids'],'original_q8':fr['original_initial_points8'],'original_L8':fr['original_L8'],'original_available8':[r['original_query_available'] for r in rr],'all_obj1_GT_instance_indices':asset['all_obj1_GT_instance_indices'],'source_asset_bindings':asset,'original_factual96_packet':fr['saved_rgb96_packet'],'original_input_flags8':[r['original_input_flags'] for r in rr],'selected_gt_instance_id_originally_certified':False})
    index=lines(ASSET_INDEX);scene_bindings={s:{} for s in sorted(SCENES)}
    wanted={'MR_GT_metadata':'GT','MR_camera_metadata':'camera','MR_GT_info_metadata':'GT_info'}
    for r in index:
        category=next((c for c in wanted if c in r['categories']),None)
        if category is None:continue
        path=permitted(r['path']);scene=path.parent.name;require(scene in SCENES,'No held GT path')
        scene_bindings[scene][wanted[category]]={'path':str(path),'sha256':r['inherited_sha256'],'body_read_in_preparation':False}
    require(all(set(b)=={'GT','camera','GT_info'} for b in scene_bindings.values()),'Exact GT/camera/info12 paths')
    writelines(FRAME_PLAN,groups);writelines(ROW_PLAN,slots);checks=invented();checks['code']=bind(__file__);write(CHECKS,checks);require(checks['PASS'],'Invented fixtures')
    protocol={'completed':True,'prepared_only':True,'code':bind(__file__),'input_metadata':{n:bind(p) for n,p in [('PUBLIC185',FRAMES),('PUBLIC1480',SLOTS),('TRAIN_asset_paths',ASSETS),('asset_stat_inventory',ASSET_INDEX),('frozen_R0_complete',R0_SEAL)]},'source_producer_code':[bind(p) for p in SOURCE_CODES],'original_frozen_proposals_inherited_binding':proposal_binding,'frame_plan':bind(FRAME_PLAN),'row_plan':bind(ROW_PLAN),'checks':bind(CHECKS),'expected':{'frames':185,'native_rows':1480,'input_available':1286,'inputU':194,'native_each':185,'obj1_GT_instance_observations':258,'TRAIN_scene_count':4},'source_scene_metadata':scene_bindings,'GT_info_hash_authority':'Existing stat inventory has no inherited SHA for four GT_info JSON files. Root must explicitly bind each SHA in actual entry before semantic parsing; preparation does not read/hash those bodies.','calculation':{'det_box':'Unmodified selected candidate box_xyxy from frozen original R0; not q8 minmax or GT crop.','GT_box':'Provided bbox_obj=[x,y,w,h] -> [x,y,x+w,y+h], no clip/expand/+1.','IoU':'intersection/(det_area+GT_area-intersection), positive widths/heights, FP64.','selection':'ALL supplied GT entries, unique exactly maximum IoU >0; exact max tie, no positive overlap, invalid/missing boxes => UNKNOWN. No confidence/ref/native/pose/mask/visibility criterion or fitted threshold.','identity_authority':'Weak bbox-overlap diagnostic only, never a certified physical/native instance. obj_id retained; an object other than1 is never forced to1.','bbox_visib':'Secondary diagnostic values only; no effect on choice.'},'execution_asset_scope':'After root entry only 12 explicit TRAIN GT/camera/GT_info JSON files; no RGB/depth/mask/mesh/NPZ/header/feature/model/annotation/score reads. All185 and1480 retained; all GT candidates recorded.','semantic_limits':['Original query/native order/inputU unchanged; selected detection candidate_index is not GT instance index.','Provided K,R,t copied unchanged for downstream source nominal geometry only; no GT-best permutation or q/L correction.','A unique IoU max is not proof of target identity, native-part association, visibility/external/self state or coordinate correctness.','No native/end/owner/quality target generation, no training admission; UNKNOWN and multiinstance evidence remain visible.','Physical exact mesh gold is not a new blocker for the existing semantic goal.'],'output':str(OUT),'automatic_retry':False}
    write(PROTOCOL,protocol)
    template={'approved_actual_source_instance_metadata_diagnostic':False,'source_TRAIN_only':True,'code':protocol['code'],'protocol':bind(PROTOCOL),'frame_plan':protocol['frame_plan'],'row_plan':protocol['row_plan'],'checks':protocol['checks'],'root_scope':{'path':None,'sha256':None},'source_scene_metadata':scene_bindings,'RGB_depth_mask_mesh_NPZ_or_model_reads':False,'label_or_training_admission':False,'automatic_retry':False,'output':str(OUT)}
    write(H/'ROOT_ACTUAL_ENTRY_TEMPLATE_APPROVED_FALSE.json',template)
    ready={'completed':True,'prepared_only':True,'code':protocol['code'],'protocol':bind(PROTOCOL),'checks':protocol['checks'],'frame_plan':protocol['frame_plan'],'row_plan':protocol['row_plan'],'template':bind(H/'ROOT_ACTUAL_ENTRY_TEMPLATE_APPROVED_FALSE.json'),'counts':protocol['expected'],'metadata_only':True,'actual_GT_camera_info_body_reads':0,'RGB_NPZ_header_body_hash_reads':0,'model_initializer_PnP_fit_calls':0,'readiness':'Executable after root scope/entry and explicit four pending GT_info SHA bindings; no new labels or admission.'}
    write(H/'PREPARATION_COMPLETE.json',ready);print(json.dumps(ready),flush=True)

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
                require(len(g['cam_R_m2c'])==9 and len(g['cam_t_m2c'])==3 and all(math.isfinite(v) for v in g['cam_R_m2c']+g['cam_t_m2c']),'Provided R/t finite shapes')
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
