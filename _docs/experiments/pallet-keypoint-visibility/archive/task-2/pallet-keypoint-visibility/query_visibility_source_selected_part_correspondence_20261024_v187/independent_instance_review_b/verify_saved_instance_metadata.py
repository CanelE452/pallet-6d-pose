"""Independent saved instance-metadata audit. No producer/array/model imports.

Preparation reads only producer code/protocol and invented inputs. A separately
bound root entry is required before the 12 whitelisted TRAIN JSON bodies, plans,
and serialized producer results can be opened.
"""
from pathlib import Path
from collections import Counter
import argparse,hashlib,json,math,os,stat,traceback
from independent_box_kernel import all_gt_unique_max,invented_checks

H=Path(__file__).absolute().parent;V=H.parent;P=V.parent
PROTOCOL=H/'PROTOCOL_BEFORE_ACTUAL_INSTANCE_METADATA_REVIEW.json'
KERNEL=H/'independent_box_kernel.py';CHECKS=H/'GENERATED_INVENTED_AUDIT_CHECKS.json'
SCENES={'000000','000002','000005','000006'}
OUTPUT_NAMES=('ALL185_FRAME_INSTANCE_ASSOCIATIONS.jsonl','ALL1480_QUERY_INSTANCE_DIAGNOSTICS.jsonl','ALL_GT_OBSERVATIONS_IN185_SOURCE_FRAMES.jsonl')

def require(value,message):
    if not value:raise ValueError(message)

def lexical(path):
    p=Path(path)
    require(p.is_absolute() and '..' not in p.parts and all('v112' not in part.lower() for part in p.parts),'Forbidden path name before access')
    require(P in p.parents and p.suffix in ('.json','.jsonl','.py'),'Only explicit project metadata/source files')
    return p

def safe_open(path):
    p=lexical(path);parts=p.relative_to(P).parts
    fd=os.open(P,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:
        for part in parts[:-1]:
            new=os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd);os.close(fd);fd=new
        final=os.open(parts[-1],os.O_RDONLY|os.O_NOFOLLOW,dir_fd=fd)
    finally:os.close(fd)
    try:require(stat.S_ISREG(os.fstat(final).st_mode),'Metadata ordinary file only')
    except BaseException:os.close(final);raise
    return os.fdopen(final,'rb')

def bind(path):
    digest=hashlib.sha256()
    with safe_open(path) as handle:
        for block in iter(lambda:handle.read(1<<20),b''):digest.update(block)
    return dict(path=str(lexical(path)),sha256=digest.hexdigest())

def check(binding):
    require(isinstance(binding.get('sha256'),str) and len(binding['sha256'])==64,'Explicit actual SHA256 required')
    require(bind(binding['path'])==dict(path=binding['path'],sha256=binding['sha256']),'Frozen metadata SHA changed')
    return Path(binding['path'])

def read(path):
    with safe_open(path) as f:return json.load(f)

def lines(path):
    with safe_open(path) as f:return [json.loads(line) for line in f if line.strip()]

def write(path,value):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')

def compare(expected,actual,where='root',float_tolerance=0.):
    require(type(expected) is type(actual),'JSON type disagreement at '+where)
    if isinstance(expected,dict):
        require(set(expected)==set(actual),'JSON fields disagreement at '+where)
        for key,value in expected.items():compare(value,actual[key],where+'/'+str(key),float_tolerance)
    elif isinstance(expected,list):
        require(len(expected)==len(actual),'JSON list length at '+where)
        for index,(a,b) in enumerate(zip(expected,actual)):compare(a,b,where+'/'+str(index),float_tolerance)
    elif isinstance(expected,float):
        require(math.isfinite(expected) and math.isfinite(actual) and abs(expected-actual)<=float_tolerance,'Scalar float disagreement at '+where)
    else:require(expected==actual,'Scalar metadata disagreement at '+where)

def flat_finite(value,count):
    def flatten(x):
        if isinstance(x,list):
            out=[]
            for y in x:out.extend(flatten(y))
            return out
        require(type(x) in (int,float) and math.isfinite(x),'Finite numeric camera/pose metadata');return [x]
    numbers=flatten(value);require(len(numbers)==count,'Provided camera/pose element count')
    return numbers

def audit_output_path(value):
    p=Path(value)
    require(p==H/'actual_review_v1' and all('v112' not in part.lower() for part in p.parts),'Fixed explicit new audit directory')
    return p

def validate_original_population(frames,rows,expected):
    require(len(frames)==expected['frames'] and len(rows)==expected['native_rows'],'All original frame/row counts')
    fids=[r['frame_index'] for r in frames]
    require(len(set(fids))==len(fids),'Unique source frame IDs')
    require(len({r['original_global_index'] for r in rows})==len(rows),'Unique original slot IDs')
    require(all(r['split']=='SOURCE_DEVELOPMENT' for r in rows),'Source TRAIN role only')
    require(sum(not r['original_query_available'] for r in rows)==expected['inputU'],'Original inputU count')
    for frame in frames:
        rr=[r for r in rows if r['original_frame_index']==frame['frame_index']]
        require([r['native_slot'] for r in rr]==list(range(8)),'Original all8 order')
        require([r['original_global_index'] for r in rr]==frame['original_global_indices'],'Original frame/global mapping')
        require(frame['original_global_indices']==[frame['frame_index']*8+k for k in range(8)],'Original frozen frame/native index arithmetic')
        require([r['original_query_xy'] for r in rr]==frame['original_q8'] and [r['original_L'] for r in rr]==frame['original_L8'],'Original q/L frame/slot match')
        require([r['original_query_available'] for r in rr]==frame['original_available8'],'Original availability mapping')
    require(Counter(r['native_slot'] for r in rows)==Counter({k:expected['native_each'] for k in range(8)}),'Every native denominator')

def make_instance(frame,ix,g,info,camera,scene_binding):
    flat_finite(camera['cam_K'],9);flat_finite(g['cam_R_m2c'],9);flat_finite(g['cam_t_m2c'],3)
    full={r['gt_instance_index']:r['binding'] for r in frame['source_asset_bindings']['full_masks']}
    visible={r['gt_instance_index']:r['binding'] for r in frame['source_asset_bindings']['visible_masks']}
    return dict(frame_index=frame['frame_index'],scene=frame['scene'],frame_key=frame['frame_key'],gt_instance_index=ix,obj_id=g['obj_id'],
        bbox_obj=info.get('bbox_obj'),bbox_visib=info.get('bbox_visib'),provided_cam_K=camera['cam_K'],provided_cam_R_m2c=g['cam_R_m2c'],
        provided_cam_t_m2c_mm=g['cam_t_m2c'],provided_depth_scale=camera.get('depth_scale'),full_mask_binding=full.get(ix),visible_mask_binding=visible.get(ix),
        GT_mask_payload_read=False,camera_GT_binding=scene_binding,native_reference_or_label_generated=False)

def self_tests():
    result=invented_checks();tests=list(result['checks'])
    flat_finite([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]],9);flat_finite([0.,0.,1.],3)
    tests.append('Provided_nested3x3_R_flat_t_validated_without_value_or_shape_rewrite')
    try:compare(dict(rows=8),dict(rows=7))
    except ValueError:tests.append('Injected_false_count_rejected')
    else:raise AssertionError('False count accepted')
    try:compare(dict(q=[1.,2.],available=False),dict(q=[1.1,2.],available=False))
    except ValueError:tests.append('Injected_q_drift_rejected')
    else:raise AssertionError('q drift accepted')
    try:compare(dict(available=False),dict(available=True))
    except ValueError:tests.append('Injected_inputU_resurrection_rejected')
    else:raise AssertionError('U resurrection accepted')
    frames=[dict(frame_index=0,original_global_indices=list(range(8)),original_q8=[[float(k),0.] for k in range(8)],original_L8=[12.]*8,original_available8=[False]*8)]
    rows=[dict(original_frame_index=0,original_global_index=k,native_slot=k,original_query_xy=[float(k),0.],original_L=12.,original_query_available=False,split='SOURCE_DEVELOPMENT') for k in range(8)]
    expected=dict(frames=1,native_rows=8,inputU=8,native_each=1);validate_original_population(frames,rows,expected)
    tests.append('Actual_population_validator_allU_and_native8_invented_fixture')
    rows[7]['original_global_index']=6
    try:validate_original_population(frames,rows,expected)
    except ValueError:tests.append('Injected_duplicate_missing_original_slot_rejected')
    else:raise AssertionError('Duplicate original slot accepted')
    require(audit_output_path(str(H/'actual_review_v1'))==H/'actual_review_v1','Output directory is not an input file')
    tests.append('Fixed_output_directory_checked_without_file_suffix_or_source_access')
    return dict(completed=True,PASS=True,invented_only=True,count=len(tests),checks=tests,actual_source_metadata_or_result_bodies_read=0)

def run(entry_path):
    entry=read(entry_path)
    require(entry.get('approved_independent_source_instance_audit') is True and entry.get('source_TRAIN_only') is True,'Root actual audit entry')
    require(entry['code']==bind(__file__) and entry['kernel']==bind(KERNEL) and entry['protocol']==bind(PROTOCOL),'Independent frozen audit binding')
    p=read(check(entry['protocol']));require(entry['checks']==p['checks'] and read(check(p['checks']))['PASS'],'Invented checks')
    for key in ('producer_code','producer_protocol','producer_complete'):
        require(entry[key]==p[key],'Final producer code/protocol/complete frozen before audit '+key)
    policy=read(check(p['producer_original_policy']))
    complete=read(check(entry['producer_complete']));require(complete['completed'] is True and complete['PASS'] is True,'Actual producer completed')
    require(complete['code']==entry['producer_code'] and complete['protocol']==entry['producer_protocol'],'Exact final producer binding')
    check(entry['producer_code']);actual_policy=read(check(entry['producer_protocol']))
    for field in ('calculation','frame_plan','row_plan','expected'):
        compare(policy[field],actual_policy[field],'frozen policy/'+field)
    scene_bindings=complete['source_scene_metadata'];require(set(scene_bindings)==SCENES,'Exactly four TRAIN scenes')
    compare(p['source_scene_metadata'],scene_bindings,'Exact prepared twelve asset whitelist')
    require(complete['new_labels_or_training_admission'] is False and complete['physical_or_native_identity_certification'] is False
            and complete['RGB_depth_mask_mesh_NPZ_body_or_header_reads']==0 and complete['model_initializer_fit_PnP_qL_changes']==0
            and complete['automatic_retry'] is False,'Producer source metadata-only scope')
    for scene in SCENES:
        require(set(scene_bindings[scene])=={'GT','camera','GT_info'},'Exactly12 metadata assets')
        for kind,b in scene_bindings[scene].items():
            original=policy['source_scene_metadata'][scene][kind]
            require(b['path']==original['path'],'No substituted source metadata path')
            require(original['sha256'] is None or b['sha256']==original['sha256'],'Inherited source SHA drift')
            require(Path(b['path']).parent.name==scene,'Whitelist scene pathname')
    output=audit_output_path(entry['output_directory'])
    output.mkdir(exist_ok=False);write(output/'STARTED.json',dict(entry=bind(entry_path),completed=False,automatic_retry=False))
    try:
        frames=lines(check(policy['frame_plan']));rows=lines(check(policy['row_plan']));validate_original_population(frames,rows,policy['expected'])
        raw={s:{k:read(check(b)) for k,b in scene_bindings[s].items()} for s in sorted(SCENES)}
        require(set(complete['outputs'])==set(OUTPUT_NAMES),'All three serialized outputs')
        require(all(Path(complete['outputs'][n]['path'])==Path(actual_policy['output'])/n for n in OUTPUT_NAMES),'Exact final producer output directory/names')
        produced={n:lines(check(complete['outputs'][n])) for n in OUTPUT_NAMES}
        require(len(produced[OUTPUT_NAMES[0]])==185 and len(produced[OUTPUT_NAMES[1]])==1480,'Serialized full denominators')
        expected_instances=[];expected_frames=[];associations={}
        for frame in frames:
            scene,key=frame['scene'],frame['frame_key'];require(scene in SCENES and frame['split']=='SOURCE_DEVELOPMENT','TRAIN frame role')
            gt,info,camera=raw[scene]['GT'][key],raw[scene]['GT_info'][key],raw[scene]['camera'][key]
            require(len(gt)==len(info),'Original instance-order GT/info')
            require([i for i,g in enumerate(gt) if g['obj_id']==1]==frame['all_obj1_GT_instance_indices'],'All original obj1 metadata observations')
            instances=[make_instance(frame,i,g,z,camera,scene_bindings[scene]) for i,(g,z) in enumerate(zip(gt,info))]
            expected_instances.extend(instances)
            association=all_gt_unique_max(frame['original_selected_detection_bbox_xyxy'],instances)
            selected=next((z for z in instances if z['gt_instance_index']==association['selected_gt_instance_index']),None)
            association.update(selected_obj1_mask_paths_available=selected is not None and selected['obj_id']==1 and selected['full_mask_binding'] is not None and selected['visible_mask_binding'] is not None,selected_instance_metadata=selected)
            associations[frame['frame_index']]=association
            expected_frames.append(dict(frame,diagnostic_association=association,all_GT_instance_count=len(instances),new_training_admission=False))
        expected_rows=[dict(row,selected_instance_diagnostic=associations[row['original_frame_index']],new_native_visibility_target=-1,new_point_quality_target=-1,new_end_support_target=[-1,-1],new_training_admission=False) for row in rows]
        # All original q/L/flags are exact; tolerance applies only independently calculated IoU scalars.
        for original,actual in zip(frames,produced[OUTPUT_NAMES[0]]):
            compare(original,{k:actual[k] for k in original},'original frame')
        for original,actual in zip(rows,produced[OUTPUT_NAMES[1]]):
            compare(original,{k:actual[k] for k in original},'original slot')
        for name,expected_values in zip(OUTPUT_NAMES,(expected_frames,expected_rows,expected_instances)):
            compare(expected_values,produced[name],name,float_tolerance=1e-12)
        obj1=sum(r['obj_id']==1 for r in expected_instances);require(obj1==258,'All258 obj1 preserved')
        counts=dict(frames=len(frames),rows=len(rows),inputU=sum(not r['original_query_available'] for r in rows),
            input_available=sum(r['original_query_available'] for r in rows),all_GT_instance_observations=len(expected_instances),obj1_GT_instance_observations=obj1,
            frame_association_states=dict(Counter(r['diagnostic_association']['status'] for r in expected_frames)),
            selected_obj_ids=dict(Counter(str(r['diagnostic_association']['selected_obj_id']) for r in expected_frames)))
        compare(counts,complete['counts'],'Serialized summary counts')
        rechecked=[policy['frame_plan'],policy['row_plan'],entry['producer_complete'],entry['producer_code'],entry['producer_protocol'],*complete['outputs'].values(),*[b for s in scene_bindings.values() for b in s.values()]]
        for binding in rechecked:check(binding)
        review=dict(completed=True,PASS=True,entry=bind(entry_path),code=bind(__file__),kernel=bind(KERNEL),protocol=bind(PROTOCOL),producer_complete=entry['producer_complete'],counts=counts,
            all_other_GT_observations=len(expected_instances)-obj1,all185_frames_all1480_rows_all12_metadata_assets_verified=True,
            raw_camera_pose_literal_values_and_shapes_preserved=True,original_q_L_native_flags_and_U_exact=True,
            independently_recomputed_allGT_amodal_and_visible_IoUs=True,all_unique_max_ties_unknown_other_object_winners_preserved=True,
            source_metadata_arrays_or_RGB_NPZ_models_evaluation_reads=0,source_GT_camera_info_JSON_files_read=12,
            new_native_end_owner_quality_labels_or_training_admission=False,producer_imports=0,automatic_retry=False,
            scope='Arithmetic/metadata integrity of weak bbox identity diagnostic only. Other-object unique max remains a matcher result, not obj1/native support; semantic support remains separate. No physical/native identity certification or new semantic blocker.',bindings_rechecked=rechecked)
        write(output/'REVIEW.json',review);print(json.dumps(dict(completed=True,PASS=True,counts=counts,review=bind(output/'REVIEW.json'))))
    except Exception:
        write(output/'FAILED_REVIEW_PRESERVED.json',dict(completed=False,automatic_retry=False,entry=bind(entry_path),traceback=traceback.format_exc()));raise

if __name__=='__main__':
    a=argparse.ArgumentParser();g=a.add_mutually_exclusive_group(required=True);g.add_argument('--self-test',action='store_true');g.add_argument('--entry',type=Path);a.add_argument('--report',type=Path);args=a.parse_args()
    if args.self_test:
        r=self_tests()
        if args.report:write(args.report,r)
        print(json.dumps(r))
    else:run(args.entry)
