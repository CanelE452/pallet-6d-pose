"""V217 nominal candidate diagnostics; never select GT or change original q/native/input roles."""
import argparse
import copy
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import numpy as np

H=Path(__file__).absolute().parent;V=H.parent;P=V.parent
PROTO=H/'PROTOCOL_BEFORE_ACTUAL_NOMINAL_CANDIDATE_PROJECTION.json'
OUT=V/'nominal_candidate_query_projection_v1'
INPUTS={
 'scope':{'path':str(V/'ROOT_PROSPECTIVE_FIXED12_PUBLIC_GT_CORRESPONDENCE_AUDIT_SCOPE.json'),'sha256':'b396012313065b36781a927dc51222c8d6acf64d0fd19186f8a80d3e6ea1a4c4'},
 'selected_complete':{'path':str(V/'fixed12_selected_GT_metadata_v1/COMPLETE.json'),'sha256':'ef3b92c0563955504e8a6d8a5caad594f05b8b9f5c7a7e78d18334cc7324e10e'},
 'selected_GT':{'path':str(V/'fixed12_selected_GT_metadata_v1/ALL12_SELECTED_TRAIN_FRAME_GT_METADATA.json'),'sha256':'e9a92956153de1a8fa82786898ba89efee41768cc5222ac573eef611e50786e5'},
 'original96':{'path':str(V/'fixed12_GT_payload_preparation_a/ALL96_ORIGINAL_INPUTS_AND_GT_POINTERS.json'),'sha256':'22803511e5abb9c9eb07d809bb5796c112fcdcd63ffe75c65bffb878549aa031'},
 'contract':{'path':str(V/'fixed12_GT_payload_preparation_a/INSTANCE_NATIVE_AND_EVIDENCE_CONTRACT.json'),'sha256':'4bfabc65c499d9044bec1176befd3e4fa634ff546b8fffa2dc55c5ea2375c9d4'},
 'definition':{'path':str(P/'query_visibility_native_part_support_feasibility_20261024_v159/native_definition_audit/DEFINITION_AND_FEASIBILITY.json'),'sha256':'76c0a83888f3c93445a8c216ceb4c6af069a939af97451fefab34bdae38d4ddc'},
}
FIXED=np.array([[-400,-600,72],[400,-600,72],[400,-600,-72],[-400,-600,-72],[-400,600,72],[400,600,72],[400,600,-72],[-400,600,-72]],dtype=np.float64)
FACES=[[0,1,2,3],[4,5,6,7],[0,4,7,3],[1,5,6,2]]

def need(value,message):
 if not value:raise AssertionError(message)

def safe(path):
 p=Path(path);need(p.is_absolute()and'..'not in p.parts and'v112'not in str(p).casefold(),'V112/relative path rejection before filesystem')
 need(P in p.parents,'Only explicit project metadata/owned output')
 return p

def parentfd(path):
 p=safe(path);fd=os.open('/',os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
 try:
  for part in p.parts[1:-1]:
   n=os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd);os.close(fd);fd=n
  return p,fd
 except BaseException:os.close(fd);raise

def rb(path):
 p=safe(path);allowed={Path(x['path'])for x in INPUTS.values()}|{Path(__file__),PROTO}
 need(p in allowed or(p.parent==V and p.name.startswith('ROOT_')and p.suffix=='.json'),'Exact generated metadata/code or explicit root entry/scope only; never data binding')
 p,fd=parentfd(p)
 try:n=os.open(p.name,os.O_RDONLY|os.O_NOFOLLOW,dir_fd=fd)
 finally:os.close(fd)
 need(stat.S_ISREG(os.fstat(n).st_mode),'Regular file/no symlink')
 with os.fdopen(n,'rb')as f:return f.read()

def new(path,data):
 p=safe(path);need(p.parent in(H,OUT),'New files only in owned prep/fresh output')
 p,fd=parentfd(p)
 try:n=os.open(p.name,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o644,dir_fd=fd)
 finally:os.close(fd)
 with os.fdopen(n,'wb')as f:f.write(data)

def mkdir_new(path):
 p,fd=parentfd(path)
 try:os.mkdir(p.name,dir_fd=fd)
 finally:os.close(fd)

def sha(data):return hashlib.sha256(data).hexdigest()
def enc(value):return(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()
def bind(path):return dict(path=str(safe(path)),sha256=sha(rb(path)))
def identity(b):
 need(isinstance(b,dict)and set(b)in({'path','sha256'},{'path','sha256','size_bytes'}),'Only path/SHA and optional size metadata')
 need(type(b['sha256'])is str and len(b['sha256'])==64 and all(c in'0123456789abcdef'for c in b['sha256']),'Exact SHA format');safe(b['path'])
 if 'size_bytes'in b:need(type(b['size_bytes'])is int and b['size_bytes']>=0,'Exact nonnegative integer byte size')
 return(b['path'],b['sha256'])
def read(b):
 identity(b);data=rb(b['path']);need(sha(data)==b['sha256'],'Pinned metadata SHA')
 if 'size_bytes'in b:need(len(data)==b['size_bytes'],'Pinned optional metadata size')
 return json.loads(data)

def metadata():
 done=read(INPUTS['selected_complete']);selected=read(INPUTS['selected_GT']);original=read(INPUTS['original96']);contract=read(INPUTS['contract']);definition=read(INPUTS['definition'])
 need(done['completed']and done['PASS']and identity(done['selected_GT_metadata'])==identity(INPUTS['selected_GT']),'Generated selected SOURCE metadata closure')
 need(contract['exact_coordinate_and_order_contract']==definition['exact_coordinate_and_order_contract'],'Exact prior nominal convention copied without changes')
 c=contract['exact_coordinate_and_order_contract'];need(np.array_equal(np.asarray(c['fixed_canonical_XYZ_mm']),FIXED)and c['front_candidates_fixed_indices']==FACES,'Frozen fixed canonical/face authority')
 need(c['permutation_direction']=='dynamic[k] = fixed[perm[k]]; never reapply to already dynamic stored arrays.','Forward permutation direction only')
 frames=selected['frames12'];rows=original['rows96']
 need(len(frames)==12 and len(rows)==96 and len({r['row_id']for r in rows})==96 and sum(not r['original_query_available']for r in rows)==11,'Original fixed12/native96/U11')
 need(sum(r['internal_source_role']=='SOURCE_FIT'for r in rows)==64 and sum(r['internal_source_role']=='SOURCE_DEV'for r in rows)==32,'Original FIT64/DEV32 separate from TRAIN')
 candidates=0
 for position,frame in enumerate(frames):
  rr=[r for r in rows if r['frame_position']==position];indices=[g['gt_list_index']for g in frame['obj1_candidates']]
  need(frame['frame_position']==position and len(rr)==8 and[r['native_slot']for r in rr]==list(range(8))and len(indices)==len(set(indices)),'Original frame and all8 native order; unique candidate indices')
  need(frame['selected_gt_instance_index']is None and frame['native_target_admission']is False,'No selected/best GT instance or target admission')
  for r in rr:need(r['frame_key']==frame['frame_key']and r['original_role']=='TRAIN'and r['internal_source_role']==frame['internal_source_role']and r['obj1_candidate_gt_list_indices']==indices and r['selected_gt_instance_index']is None and r['training_admission']is False,'Original input/candidate metadata join, no changes')
  need(all(g['GT']['obj_id']==1 for g in frame['obj1_candidates']),'All recorded obj1 candidates, no filtering by outcome');candidates+=len(indices)
 need(candidates==15,'All15 predeclared GT candidates preserved')
 return frames,rows,c

def project_fixed(points,R,t,K):
 camera=points@R.T+t;homogeneous=camera@K.T
 xy=[]
 for xyz,h in zip(camera,homogeneous):
  value=h[:2]/xyz[2]if np.all(np.isfinite(xyz))and xyz[2]>0 else None
  xy.append(value.tolist()if value is not None and np.all(np.isfinite(value))else None)
 return camera,xy

def nominal_mapping(R_value,t_value,K_value):
 out=dict(status='UNKNOWN',reason=None,perm_native_to_fixed=None,camera_object_mm=None,front_face_index=None,front_scores=None,front_score_margin=None,front_exact_tie=None,top_left_right_X_margin=None,top_left_right_exact_tie=None,fixed_projection8=None,R_determinant=None,R_orthogonality_maxabs_error=None)
 try:R=np.asarray(R_value,dtype=np.float64).reshape(3,3);t=np.asarray(t_value,dtype=np.float64).reshape(3);K=np.asarray(K_value,dtype=np.float64).reshape(3,3)
 except(TypeError,ValueError):out['reason']='MISSING_OR_MALFORMED_NOMINAL_POSE_CAMERA';return out
 if not(np.all(np.isfinite(R))and np.all(np.isfinite(t))and np.all(np.isfinite(K))):out['reason']='NONFINITE_NOMINAL_POSE_CAMERA';return out
 out['R_determinant']=float(np.linalg.det(R));out['R_orthogonality_maxabs_error']=float(np.max(np.abs(R@R.T-np.eye(3))))
 camera,xy=project_fixed(FIXED,R,t,K);out['fixed_projection8']=[dict(canonical_index=i,fixed_XYZ_mm=FIXED[i].tolist(),camera_Z_mm=float(p[2])if np.isfinite(p[2])else None,reference_xy=xy[i],projection_status='POSITIVE_Z_FINITE'if xy[i]is not None else'UNKNOWN_NONPOSITIVE_Z_OR_NONFINITE')for i,p in enumerate(camera)]
 try:camobj=np.linalg.solve(R,-t)
 except np.linalg.LinAlgError:out['reason']='SINGULAR_R';return out
 out['camera_object_mm']=camobj.tolist();distance=float(np.linalg.norm(camobj))
 if not np.isfinite(distance)or distance==0:out['reason']='CAMERA_AT_MODEL_CENTER_OR_NONFINITE';return out
 normals=[]
 for face in FACES:
  p=FIXED[face];n=np.cross(p[1]-p[0],p[2]-p[0]);n=n/np.linalg.norm(n)
  if np.dot(n,p.mean(axis=0))<0:n=-n
  normals.append(n)
 scores=np.asarray(normals)@(camobj/distance);choice=int(np.argmax(scores));ordered=np.sort(scores)
 out.update(front_face_index=choice,front_scores=scores.tolist(),front_score_margin=float(ordered[-1]-ordered[-2]),front_exact_tie=bool(np.sum(scores==scores[choice])>1))
 top=[i for i in FACES[choice]if FIXED[i,2]==72]
 if any(xy[i]is None for i in top):out['reason']='FRONT_TOP_PROJECTION_UNKNOWN';return out
 left,right=sorted(top,key=lambda i:xy[i][0])  # Stable declared-face order only on exact projected-X ties.
 margin=abs(xy[right][0]-xy[left][0]);out.update(top_left_right_X_margin=float(margin),top_left_right_exact_tie=bool(margin==0))
 lookup={tuple(p):i for i,p in enumerate(FIXED.tolist())}
 def bottom(index):p=FIXED[index].copy();p[2]=-72;return lookup[tuple(p)]
 front=[left,right,bottom(right),bottom(left)];constant_axes=[a for a in(0,1)if np.ptp(FIXED[FACES[choice],a])==0];need(len(constant_axes)==1,'Vertical face has one depth axis');axis=constant_axes[0]
 rear=[]
 for i in front:p=FIXED[i].copy();p[axis]=-p[axis];rear.append(lookup[tuple(p)])
 perm=front+rear;need(sorted(perm)==list(range(8)),'One geometry-derived forward permutation')
 out.update(status='NOMINAL_MAPPING_AVAILABLE_WITH_UNCERTAINTY',reason='DECLARED_NOMINAL_ONLY_TIES_AND_MARGINS_NOT_SEMANTIC_GATES',perm_native_to_fixed=perm)
 return out

def query_diagnostic(original,reference):
 result=dict(status='ORIGINAL_INPUT_U'if not original['original_query_available']else'UNKNOWN_REFERENCE',delta_reference_minus_q_px=None,delta_over_original_L=None,RAW96_reference_address=None,image_inbounds='UNKNOWN_NO_PIXEL_DIMENSIONS',coordinate_error_gold=False)
 if not original['original_query_available']or reference is None:return result
 try:q=np.asarray(original['query_xy'],dtype=np.float64).reshape(2);L=float(original['L']);ref=np.asarray(reference,dtype=np.float64)
 except(TypeError,ValueError):result['status']='UNKNOWN_ORIGINAL_SCALAR';return result
 if not np.all(np.isfinite(q))or not np.isfinite(L)or L<=0:result['status']='UNKNOWN_ORIGINAL_SCALAR';return result
 delta=ref-q;result.update(status='NOMINAL_REFERENCE_DIAGNOSTIC_ONLY',delta_reference_minus_q_px=delta.tolist(),delta_over_original_L=(delta/L).tolist(),RAW96_reference_address=(delta*(96./L)+47.5).tolist());return result

def candidate_rows(original8,candidate,camera):
 gt=candidate['GT'];mapping=nominal_mapping(gt.get('cam_R_m2c'),gt.get('cam_t_m2c'),camera.get('cam_K'));rows=[]
 for original in original8:
  k=original['native_slot'];perm=mapping['perm_native_to_fixed'];ci=None if perm is None else perm[k];projection=None if ci is None else mapping['fixed_projection8'][ci]
  reference=None if projection is None else projection['reference_xy']
  rows.append(dict(original_input=copy.deepcopy(original),diagnostic_candidate_gt_list_index=candidate['gt_list_index'],obj_id=gt['obj_id'],selected_gt_instance_index=None,native_slot=k,canonical_index=ci,nominal_fixed_XYZ_mm=None if ci is None else FIXED[ci].tolist(),nominal_camera_Z_mm=None if projection is None else projection['camera_Z_mm'],nominal_reference_xy=reference,nominal_mapping_status=mapping['status'],query_diagnostic=query_diagnostic(original,reference),nominal_not_physical_native_gold=True,new_native_visibility_label=None,training_admission=False))
 return rows,mapping

def invented_tests():
 I=np.eye(3);K=np.array([[1000.,0,320],[0,1000,240],[0,0,1]])
 m=nominal_mapping(I,[0,3000,3000],K);need(m['perm_native_to_fixed']==list(range(8))and m['front_face_index']==0,'Front negative-Y and fixed/native direction')
 expected=[320-400000/3072,240+2400000/3072];need(np.allclose(m['fixed_projection8'][0]['reference_xy'],expected,rtol=0,atol=1e-12),'Independent pinhole fixture axes/Z')
 flip=np.diag([-1.,-1.,1.]);f=nominal_mapping(flip,[0,-3000,3000],K);need(f['perm_native_to_fixed']==[1,0,3,2,5,4,7,6],'Projected image-left/right, opposite-depth rear partners')
 tied=nominal_mapping(I,[0,0,3000],K);need(tied['front_exact_tie']and tied['front_face_index']==0,'Exact face ties reported and declared-first deterministic')
 side=nominal_mapping(I,[3000,0,3000],K);need(side['front_face_index']==2 and side['top_left_right_exact_tie']and side['perm_native_to_fixed']==[0,4,7,3,1,5,6,2],'Top-X tie retained; rear not independently image-sorted')
 R=np.diag([1.01,.99,1.]);t=np.array([10.,20.,3000.]);inverse=nominal_mapping(R,t,K);need(np.allclose(inverse['camera_object_mm'],np.linalg.solve(R,-t))and not np.allclose(inverse['camera_object_mm'],-R.T@t),'Solve inverse, not silent transpose of rounded R')
 behind=nominal_mapping(I,[0,0,-3000],K);need(behind['perm_native_to_fixed']is None and all(p['reference_xy']is None for p in behind['fixed_projection8']),'Nonpositive Z stays UNKNOWN without division/recenter')
 singular=nominal_mapping(np.zeros((3,3)),[0,0,3000],K);need(singular['reason']=='SINGULAR_R'and singular['perm_native_to_fixed']is None,'Singular nominal pose explicit UNKNOWN')
 original=[dict(row_id='INVENTED'+str(k),native_slot=k,query_xy=[10.+k,20.],L=192.,original_query_available=k!=7,original_role='TRAIN',internal_source_role='SOURCE_DEV')for k in range(8)]
 candidate=dict(gt_list_index=2,GT=dict(obj_id=1,cam_R_m2c=I.tolist(),cam_t_m2c=[0,3000,3000]));a,ma=candidate_rows(original,candidate,dict(cam_K=K.tolist()));other=copy.deepcopy(original)
 for r in other:r['query_xy']=[99999.,-99999.]
 b,mb=candidate_rows(other,candidate,dict(cam_K=K.tolist()));need(ma==mb and a[7]['query_diagnostic']['status']=='ORIGINAL_INPUT_U'and a[7]['nominal_reference_xy']is not None and original[7]['original_query_available']is False,'Geometry independent of q, U remains U despite available geometry')
 second=copy.deepcopy(candidate);second['gt_list_index']=7;aa,_=candidate_rows(original,second,dict(cam_K=K.tolist()));need(len(a+aa)==16 and all(r['selected_gt_instance_index']is None and r['new_native_visibility_label']is None and r['training_admission']is False for r in a+aa),'Every GT candidate retained, no best/nearest/label')
 need(query_diagnostic(dict(original_query_available=True,query_xy=[100.,200.],L=192.),[120.,180.])['RAW96_reference_address']==[57.5,37.5],'Original96 address algebra diagnostic only')
 return dict(completed=True,PASS=True,invented_only=True,count=10,checks=['Fixed_front_native_forward_direction_and_independent_pinhole','Image_LR_and_opposite_depth_rear','Face_exact_tie_flag','Top_LR_tie_flag','Camera_solve_not_transpose','Negative_Z_UNKNOWN','Singular_R_UNKNOWN','Query_independent_geometry_original_U_retained','All_candidates_no_GT_best_or_labels','RAW96_original_qL_algebra'],actual_SOURCE_projection_calls=0)

def prepare():
 frames,rows,c=metadata()  # Closure/identity/counts only; do not call nominal_mapping on SOURCE matrices.
 checks=invented_tests();new(H/'INVENTED_NOMINAL_PROJECTION_CHECKS.json',enc(checks))
 template=dict(approved_actual_nominal_candidate_projection=False,root_actual_scope=None,code=bind(__file__),protocol=bind(PROTO),inputs=INPUTS,output_directory=str(OUT),original_payloads_or_models_or_scores_allowed=False,no_GT_candidate_selection=True,training_admission=False,automatic_retry=False)
 new(H/'ROOT_ACTUAL_NOMINAL_PROJECTION_ENTRY_TEMPLATE_APPROVED_FALSE.json',enc(template))
 ready=dict(completed=True,PASS=True,prepared_only=True,code=bind(__file__),protocol=bind(PROTO),checks=dict(path=str(H/'INVENTED_NOMINAL_PROJECTION_CHECKS.json'),sha256=sha(enc(checks))),false_entry=dict(path=str(H/'ROOT_ACTUAL_NOMINAL_PROJECTION_ENTRY_TEMPLATE_APPROVED_FALSE.json'),sha256=sha(enc(template))),inputs=INPUTS,source_frames=12,original_rows=96,original_U=11,all_candidates=15,future_expanded_rows=120,actual_SOURCE_projection_executed=False,raw_GT_camera_mask_depth_CAD_RGB_NPZ_access=0,models_scores_Git_reads=0,frozen_no_further_edits=True)
 new(H/'FINAL_READY.json',enc(ready));print(json.dumps(dict(completed=True,PASS=True,ready=dict(path=str(H/'FINAL_READY.json'),sha256=sha(enc(ready))),code=ready['code'],protocol=ready['protocol'])))

def run(entry_path):
 e=json.loads(rb(entry_path));need(e['approved_actual_nominal_candidate_projection']is True and e['original_payloads_or_models_or_scores_allowed']is False and e['no_GT_candidate_selection']is True and e['training_admission']is False and e['automatic_retry']is False,'Separate root actual projection authority')
 need(identity(e['code'])==identity(bind(__file__))and identity(e['protocol'])==identity(bind(PROTO))and e['inputs']==INPUTS and e['output_directory']==str(OUT),'Frozen exact closure/fresh output')
 actual=read(e['root_actual_scope']);need(actual['approved_actual_nominal_candidate_projection']is True,'Separate root actual scope')
 frames,original,c=metadata();mkdir_new(OUT);new(OUT/'STARTED.json',enc(dict(completed=False,entry=bind(entry_path),automatic_retry=False)))
 try:
  rows=[];candidates=[]
  for frame in frames:
   rr=[r for r in original if r['frame_position']==frame['frame_position']]
   for candidate in frame['obj1_candidates']:
    expanded,mapping=candidate_rows(rr,candidate,frame['camera']);rows.extend(expanded);candidates.append(dict(frame_position=frame['frame_position'],frame_key=frame['frame_key'],gt_list_index=candidate['gt_list_index'],mapping=mapping,selected_instance=False,training_admission=False))
  need(len(rows)==120 and len(candidates)==15 and{r['original_input']['row_id']for r in rows}=={r['row_id']for r in original},'All15×8, every original96 preserved')
  for r in rows:need(r['original_input']==next(x for x in original if x['row_id']==r['original_input']['row_id']),'Original full metadata/qL/U/native/roles exact')
  summary=dict(completed=True,PASS=True,source_frames=12,original_unique_rows=96,original_inputU=11,original_SOURCE_FIT_rows=64,original_SOURCE_DEV_rows=32,candidate_instances=15,candidate_native_rows=120,expanded_inputU_rows=sum(not r['original_input']['original_query_available']for r in rows),mapping_status_counts={s:sum(g['mapping']['status']==s for g in candidates)for s in sorted({g['mapping']['status']for g in candidates})},face_exact_tie_candidates=sum(g['mapping']['front_exact_tie']is True for g in candidates),top_LR_exact_tie_candidates=sum(g['mapping']['top_left_right_exact_tie']is True for g in candidates),image_inbounds='UNKNOWN_NO_DIMENSIONS_OPENED',candidate_selection=None,native_labels_created=0,training_admission=False,nominal_point_not_physical_surface_or_native_gold=True,source_input_qL_native_order_or_roles_changed=0,original_payload_or_model_score_reads=0)
  new(OUT/'ALL120_CANDIDATE_NATIVE_QUERY_DIAGNOSTICS.json',enc(dict(completed=True,rows120=rows,training_admission=False)));new(OUT/'ALL15_NOMINAL_CANDIDATE_MAPPINGS.json',enc(dict(completed=True,candidates=candidates)));new(OUT/'SUMMARY.json',enc(summary))
  flat=[]
  for r in rows:
   o=r['original_input'];d=r['query_diagnostic'];flat.append(dict(row_id=o['row_id'],frame_position=o['frame_position'],frame_key=o['frame_key'],component_id=o['component_id'],original_role=o['original_role'],internal_source_role=o['internal_source_role'],native_slot=o['native_slot'],original_query_available=o['original_query_available'],original_query_xy=json.dumps(o['query_xy']),original_L=o['L'],diagnostic_gt_list_index=r['diagnostic_candidate_gt_list_index'],selected_gt_instance_index='',canonical_index=r['canonical_index'],fixed_XYZ_mm=json.dumps(r['nominal_fixed_XYZ_mm']),reference_xy=json.dumps(r['nominal_reference_xy']),camera_Z_mm=r['nominal_camera_Z_mm'],mapping_status=r['nominal_mapping_status'],query_diagnostic_status=d['status'],delta_ref_minus_q_px=json.dumps(d['delta_reference_minus_q_px']),delta_over_original_L=json.dumps(d['delta_over_original_L']),RAW96_reference_address=json.dumps(d['RAW96_reference_address']),image_inbounds=d['image_inbounds'],native_label='',training_admission=False))
  out=io.StringIO(newline='');w=csv.DictWriter(out,fieldnames=list(flat[0]),lineterminator='\n');w.writeheader();w.writerows(flat);csvdata=out.getvalue().encode();new(OUT/'ALL120_NOMINAL_CANDIDATE_QUERY_DIAGNOSTICS.csv',csvdata)
  artifacts={name:dict(path=str(OUT/name),sha256=sha(data))for name,data in [('ALL120_CANDIDATE_NATIVE_QUERY_DIAGNOSTICS.json',enc(dict(completed=True,rows120=rows,training_admission=False))),('ALL15_NOMINAL_CANDIDATE_MAPPINGS.json',enc(dict(completed=True,candidates=candidates))),('SUMMARY.json',enc(summary)),('ALL120_NOMINAL_CANDIDATE_QUERY_DIAGNOSTICS.csv',csvdata)]}
  complete=dict(completed=True,PASS=True,code=e['code'],protocol=e['protocol'],entry=bind(entry_path),inputs=INPUTS,artifacts=artifacts,summary=summary,source_projection_executed=True,original_payload_reads=0,labels_or_admission_created=0,automatic_retry=False);new(OUT/'COMPLETE.json',enc(complete));print(json.dumps(dict(completed=True,PASS=True,complete=dict(path=str(OUT/'COMPLETE.json'),sha256=sha(enc(complete))))))
 except BaseException as error:new(OUT/'FAILURE_PRESERVED_NO_AUTOMATIC_RETRY.json',enc(dict(completed=False,PASS=False,error_type=type(error).__name__,error=str(error),automatic_retry=False)));raise

if __name__=='__main__':
 parser=argparse.ArgumentParser();group=parser.add_mutually_exclusive_group(required=True);group.add_argument('--prepare',action='store_true');group.add_argument('--entry');args=parser.parse_args();prepare()if args.prepare else run(args.entry)
