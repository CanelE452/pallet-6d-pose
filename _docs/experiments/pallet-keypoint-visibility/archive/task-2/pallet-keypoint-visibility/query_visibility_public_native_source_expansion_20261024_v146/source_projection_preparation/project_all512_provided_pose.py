"""GT-only public64 references and per-camera LOWER role continuity.

Prepare reads existing metadata/code only. New T/K bodies require root entry AND
completed RGB-only R0 seal. Never reads R0 q, source RGB/depth/masks or models.
"""
from pathlib import Path
from collections import Counter
import argparse,ast,hashlib,json,sys,time,traceback
import numpy as np
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;V=H.parent;P=V.parent
POLICY=V/'ROOT_PUBLIC64_NATIVE_EXPANSION_POLICY_BEFORE_NEW_BODIES.json';PSHA='592ab0be3c8635cb75cbbd93c3457daf463892e7dfe34c5029c63213746d247e'
SELECTION=V/'FIXED64_SELECTION_BEFORE_NEW_SOURCE_BODIES.json';SSHA='cf33c72e0d5f547d0ecfcf8849fbbfd30beda80724587bfef9972e9227aa26f2'
TRANSPORT=V/'canonical_RGB_transport_v1/TRANSPORT_COMPLETE.json'
OLD=P/'query_visibility_public_native_reference_feasibility_20261024_v140'
AXIS=OLD/'pallox_reference_preparation/AXIS_REGISTRATION_PROPOSAL_BEFORE_NAMED_NATIVE_LABELS.json';ASHA='74802a2f7663e75695a9ca1d44870357e3e7b5914c4dc28572fceab3fc3b756e'
NAMED=OLD/'pallox_named_virtual_reference_v1/NAMED_CF32_REFERENCES.json'
CONTRACT=P/'query_visibility_loco_native_context_20261023_v74/source_contract/native_context_contract_v2.py';CSHA='4ddd134aa26da1d676628bd6987a161ab36f09328dfdaa89bedd213e397755ab'
LOWER=P/'query_visibility_public_selected_native_labels_20261024_v144/pallox_base_role_geometry_review/LOWER_ROLE_INVARIANCE_CERTIFICATE.json';LSHA='7e891ff3f831a72f640fff4f2c9eefc11e0ace690ef85d458fdb08ad44f83bc1'
SOURCE=P/'query_visibility_controlled_real_20261022_v66/public_pallet';INVENTORY=SOURCE/'ALL_DOWNLOAD_INVENTORY.jsonl'
PROTOCOL=H/'PROTOCOL_BEFORE_NEW_POSE_K_READS.json';GENERATED=H/'GENERATED_GEOMETRY_CHECKS.json';BINDINGS=H/'POSE_K64_BINDINGS_METADATA_ONLY.json'
OUT=V/'provided_pose_references_v1';FORBIDDEN=('RGB_depth_mask_reads','initializer_q_score_reads','model_fit_or_forward','current223','V112','PnP_pose_or_q_L_optimization','base_upper_cut_selection','semantic_label_by_geometry','row_replacement_or_drop')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def read(p):return json.loads(Path(p).read_text())
def write(p,x):
 with Path(p).open('x')as f:json.dump(x,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
def check(b):
 assert '_v112'not in b['path'].lower()and '/private_evaluation/'not in b['path'];assert sha(b['path'])==b['sha256'],b['path'];return Path(b['path'])
def native_math():
 assert sha(CONTRACT)==CSHA;wanted={'FIXED_ZUP_SIGNS','VERTICAL_FIXED_FACES','NATIVE_NAMES','outward_normals','normal_facing_permutation'};nodes=[]
 for n in ast.parse(CONTRACT.read_text()).body:
  if isinstance(n,ast.FunctionDef)and n.name in wanted:nodes.append(n)
  elif isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id in wanted for t in n.targets):nodes.append(n)
 ns={'np':np};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(CONTRACT)+'::pure_static_geometry','exec'),ns);assert wanted.issubset(ns);return ns

def geometry_metadata():
 assert sha(AXIS)==ASHA and sha(LOWER)==LSHA
 assert bind(NAMED)==read(LOWER)['original_named32']
 axis=read(AXIS);named=read(NAMED);n=native_math()
 obj=np.array(named['registered_fixed_object_corners_xyz_m'],np.float64);B=np.array(axis['source_specific_registration']['world_from_centered_object_rotation']);C=np.array(axis['source_specific_registration']['C_raw_object_AABB_center_m'])
 assert obj.shape==(8,3)and np.allclose(B@B.T,np.eye(3))and np.linalg.det(B)==1
 return n,obj,B,C

def project(obj,K,T):
 camera=obj@T[:3,:3].T+T[:3,3];hp=camera@K.T
 with np.errstate(divide='ignore',invalid='ignore'):xy=hp[:,:2]/hp[:,2,None]
 return xy,camera[:,2]

def camera_geometry(T,K,n,obj,B,C):
 """Deterministic supplied-matrix math. Failure keeps frame, never optimizes."""
 reason=[];p=None;xy=np.full((8,2),np.nan);z=np.full(8,np.nan);cos=None;proof=None
 if T.shape!=(4,4)or K.shape!=(3,3)or not np.isfinite(T).all()or not np.isfinite(K).all():
  return dict(PASS=False,reasons=['PROVIDED_MATRIX_SHAPE_OR_FINITE_INVALID'],perm=None,xy=xy,z=z,cos=None,proof=None)
 orth=float(np.max(np.abs(T[:3,:3].T@T[:3,:3]-np.eye(3))));det=float(np.linalg.det(T[:3,:3]));last=float(np.max(np.abs(T[3]-[0,0,0,1])))
 if orth>1e-8 or abs(det-1)>1e-8 or last>1e-12:reason.append('PROVIDED_T_NOT_CERTIFIED_SE3')
 if not np.array_equal(K[2],np.array([0.,0.,1.])):reason.append('K_NOT_STANDARD_PINHOLE_DENOMINATOR')
 xy,z=project(obj,K,T);world=(obj-C)@B.T;camera=(-T[:3,:3].T@T[:3,3]-C)@B.T
 if not np.isfinite(xy).all():reason.append('NONFINITE_FIXED_AABB_PROJECTION')
 else:
  try:p,cos=n['normal_facing_permutation'](world,xy,camera)
  except ValueError:reason.append('CF_CAMERA_ORDER_UNRESOLVED')
 if p is not None:
  normals,_=n['outward_normals'](world,n['VERTICAL_FIXED_FACES']);margin=float(np.sort(cos)[-1]-np.sort(cos)[-2]);gap=float(xy[p[1],0]-xy[p[0],0])
  if margin<=1e-10:reason.append('FRONT_NORMAL_TIE_OR_NEAR_TIE')
  if gap<=1e-10:reason.append('FRONT_PROJECTED_X_TIE_OR_NEAR_TIE')
  if np.max(np.abs(normals[:,2]))>1e-12:reason.append('VERTICAL_NORMAL_CONTRACT_MISMATCH')
  top=np.flatnonzero(n['FIXED_ZUP_SIGNS'][:,2]>0);bottom=np.flatnonzero(n['FIXED_ZUP_SIGNS'][:,2]<0);lo=float(obj[top[0],1]);hi=float(obj[bottom[0],1]);assert hi>lo
  # u_b-u_a numerator is linear in upper raw-Y; t² cancels exactly.
  a,b=obj[p[0]].copy(),obj[p[1]].copy();a[1]=b[1]=0;ca=a@T[:3,:3].T+T[:3,3];cb=b@T[:3,:3].T+T[:3,3]
  alpha=float(K[0]@T[:3,1]);gamma=float(T[2,1]);ba=float(K[0]@ca);bb=float(K[0]@cb);da=float(ca[2]);db=float(cb[2]);slope=alpha*(da-db)+gamma*(bb-ba);intercept=bb*da-ba*db
  ends=[]
  for t in(lo,hi):
   qq=obj.copy();qq[top,1]=t;pp,zz=project(qq,K,T);ends.append(dict(raw_upper_Y_m=t,all8_positive_Z=bool(np.isfinite(zz).all()and(zz>0).all()),minimum_Z_m=float(zz.min()),front_top_X_gap_px=float(pp[p[1],0]-pp[p[0],0])if np.isfinite(pp[p[[0,1]]]).all()else None,numerator=float(slope*t+intercept)))
  if not all(e['all8_positive_Z']for e in ends):reason.append('UPPER_HEIGHT_INTERVAL_CAMERA_Z_NOT_ALL_POSITIVE')
  if not all(e['front_top_X_gap_px']is not None and e['front_top_X_gap_px']>1e-10 and e['numerator']>0 for e in ends):reason.append('UPPER_HEIGHT_INTERVAL_PROJECTED_X_ORDER_NOT_CERTIFIED')
  proof=dict(upper_raw_Y_interval_m=[lo,hi],upper_endpoint_is_degenerate_limit_not_selected_basecut=True,endpoints=ends,linear_projectedX_difference_numerator_slope=slope,linear_projectedX_difference_numerator_intercept=intercept,front_normal_top2_margin=margin,fullheight_front_top_image_X_gap_px=gap,front_priority_height_invariant='Vertical normals have zero registered-Z; changing upperheight only changes common positive cosine denominator.',lower_XZ_and_bottomplane_fixed=True,raw_global_AABB_unchanged=True)
 return dict(PASS=not reason,reasons=reason,perm=p,xy=xy,z=z,cos=cos,proof=proof,T_orthogonality_max_abs=orth,T_determinant=det,T_lastrow_max_abs=last)

def generated():
 n,obj,B,C=geometry_metadata();K=np.array([[500.,0,320],[0,500,240],[0,0,1]]);T=np.eye(4);T[:3,:3]=np.diag([-1.,1,-1]);T[2,3]=5
 good=camera_geometry(T,K,n,obj,B,C);assert good['PASS']and good['perm']is not None
 close=T.copy();close[2,3]=0;bad=camera_geometry(close,K,n,obj,B,C);assert not bad['PASS']and 'UPPER_HEIGHT_INTERVAL_CAMERA_Z_NOT_ALL_POSITIVE'in bad['reasons']
 tied=T.copy();tied[:3,3]=-tied[:3,:3]@(C+np.array([3.,0,3.]));bad=camera_geometry(tied,K,n,obj,B,C);assert not bad['PASS']and 'FRONT_NORMAL_TIE_OR_NEAR_TIE'in bad['reasons']
 flat=K.copy();flat[0]=[0,0,320];bad=camera_geometry(T,flat,n,obj,B,C);assert not bad['PASS']
 nonstandard=K.copy();nonstandard[2,0]=.01;bad=camera_geometry(T,nonstandard,n,obj,B,C);assert not bad['PASS']and 'K_NOT_STANDARD_PINHOLE_DENOMINATOR'in bad['reasons']
 invalid=camera_geometry(np.zeros((2,2)),K,n,obj,B,C);assert not invalid['PASS']and invalid['perm']is None
 # Continuum proof supplemented with invented 67 height replays, not actual new camera reads.
 p=good['perm'];xy0,z0=project(obj,K,T);top=np.flatnonzero(n['FIXED_ZUP_SIGNS'][:,2]>0)
 for y in np.linspace(obj[:,1].min(),obj[:,1].max(),67,endpoint=False):
  qq=obj.copy();qq[top,1]=y;xx,zz=project(qq,K,T);per,_=n['normal_facing_permutation']((qq-C)@B.T,xx,(-T[:3,:3].T@T[:3,3]-C)@B.T);assert np.array_equal(per,p)and np.array_equal(xx[per[[2,3]]],xy0[p[[2,3]]])
 write(GENERATED,dict(completed=True,PASS=True,code=bind(__file__),invented_K_T_arrays_only=True,new64_pose_K_RGB_q_body_reads=0,positive_continuous_height_proof_and67_parity=True,negativeZ_fronttie_collapsedX_badshape_nonstandardK_UNRESOLVED=True,no_geometry_to_semantic_labels=True,no_model_or_PnP=True))

def prepare():
 assert sha(POLICY)==PSHA and sha(SELECTION)==SSHA and not PROTOCOL.exists()
 selection=read(SELECTION);assert selection['completed']and len(selection['rows'])==64 and selection['policy']==bind(POLICY)
 transport=read(TRANSPORT);assert transport['completed']and transport['PASS']and transport['frames']==64 and transport['selection']==bind(SELECTION)
 rgb=read(check(transport['manifest']));assert len(rgb['rows'])==64 and all(a['id']==b['id']and a['source_id']==b['source_id']and a['split']==b['split']for a,b in zip(selection['rows'],rgb['rows']))
 inv={r['relative_path']:r for r in (json.loads(s)for s in INVENTORY.read_text().splitlines()if s)};files=[]
 for i,r in enumerate(selection['rows']):
  assert r['frame_index']==i and r['source']=='PALLOX';item={k:r[k]for k in('frame_index','id','source_id','sequence','frame_stem','split')}
  for field in('provided_pose','provided_K'):
   path=Path(r[field]);meta=inv[str(path.relative_to(SOURCE/'data'))];item[field]=dict(path=str(path),sha256=meta['sha256'])
  files.append(item)
 assert len({r['source_id']for r in files})==64 and Counter(r['split']for r in files)==Counter(TRAIN_LAB=32,VAL_HALL=32)
 write(BINDINGS,dict(completed=True,metadata_only=True,selection=bind(SELECTION),source_download_inventory=bind(INVENTORY),rows=files,new_pose_K_body_hash_reads=0))
 generated();n,obj,B,C=geometry_metadata()
 p=dict(completed=True,prepared_only=True,code=bind(__file__),policy=bind(POLICY),selection=bind(SELECTION),transport_seal=bind(TRANSPORT),RGB_manifest=transport['manifest'],pose_K_bindings=bind(BINDINGS),generated=bind(GENERATED),axis_proposal=bind(AXIS),old_named_geometry=bind(NAMED),V144_lower_certificate=bind(LOWER),original_CF_code=bind(CONTRACT),output=str(OUT),frames=64,rows=512,semantic_states_all_U=True,model_q_or_score_or_RGB_depth_mask_reads=0,
  reference_scope='Original whole-assembly full scanAABB and bottomplane; UP=-OBJY provisional; no standalone baseupper or basefootprint recovery. Original34.5mm isolatedscan Xoutlier unchanged.',
  failure='Each failed camera remains with all8 rows and lower_role_continuity_status UNRESOLVED; no selection/replacement/cut/point snapping.',
  mandatory_order='Completed original RGB-only initializer seal metadata -> explicit root projection entry -> bound direct T/K projection only. No R0 query/prediction body reads.',
  schema='ALL512_SOURCE_REFERENCES_GT_ONLY.json {completed,PASS,frames:[...],rows:[{frame_index,id,source_frame_id,split,native_slot,native_name,reference_object_xyz_m,reference_xy,camera_Z_m,continuous_inframe,role_status,lower_role_continuity_status,semantic_state:U}]}')
 write(PROTOCOL,p);template=dict(approved_source_GT_projection=False,**{k:p[k]for k in('code','policy','selection','transport_seal','RGB_manifest','pose_K_bindings','generated')},protocol=bind(PROTOCOL),initializer_prediction_seal={'path':'ROOT_FILL_AFTER_COMPLETED_R0','sha256':'ROOT_FILL_AFTER_COMPLETED_R0'},initializer_input_manifest={'path':'ROOT_FILL_EXACT_R0_INPUT_MANIFEST','sha256':'ROOT_FILL_EXACT_R0_INPUT_MANIFEST'},**{k:False for k in FORBIDDEN})
 write(H/'ROOT_PROJECTION_ENTRY_TEMPLATE.json',template);write(H/'PREPARATION_READY.json',dict(completed=True,prepared_only=True,code=bind(__file__),protocol=bind(PROTOCOL),generated=bind(GENERATED),template=bind(H/'ROOT_PROJECTION_ENTRY_TEMPLATE.json'),new64_pose_K_RGB_q_body_reads=0,model_calls=0,command='/home/minjae/anaconda3/envs/pallet-yolo26/bin/python '+str(Path(__file__).resolve())+' --entry ROOT_EXACT_ENTRY.json'))
 print(json.dumps(read(H/'PREPARATION_READY.json')),flush=True)

def load_input_metadata(path):
 text=Path(path).read_text()
 try:a=json.loads(text)
 except json.JSONDecodeError:a=[json.loads(s)for s in text.splitlines()if s]
 return a['rows']if isinstance(a,dict)else a

def run(path):
 e=read(path);assert e['approved_source_GT_projection']is True and all(e[k]is False for k in FORBIDDEN)
 assert sha(POLICY)==PSHA and sha(SELECTION)==SSHA and not OUT.exists();p=read(PROTOCOL)
 for key in('code','policy','selection','transport_seal','RGB_manifest','pose_K_bindings','generated'):
  assert e[key]==p[key];check(p[key])
 for key in('axis_proposal','old_named_geometry','V144_lower_certificate','original_CF_code'):check(p[key])
 assert e['code']==bind(__file__)and e['protocol']==bind(PROTOCOL)and read(GENERATED)['PASS']
 init=read(check(e['initializer_prediction_seal']));assert init['completed']and init['PASS']and init['frames']==64 and init['rows']==512 and init['GT_read']is False and init['actual_original_initializer']is True and init['source_emulation']is False
 assert init['inputs']==e['initializer_input_manifest'];metadata=load_input_metadata(check(e['initializer_input_manifest']));sel=read(SELECTION)['rows'];assert len(metadata)==64
 for i,(a,b)in enumerate(zip(sel,metadata)):assert a['frame_index']==b['frame_index']==i and a['id']==b['id']and a['source_id']==b['source_id']and a['split']==b['split']
 files=read(BINDINGS)['rows'];n,obj,B,C=geometry_metadata();OUT.mkdir();write(OUT/'BEFORE_NEW_POSE_K_READ.json',dict(completed=True,entry=bind(path),initializer_prediction_seal=e['initializer_prediction_seal'],initializer_q_score_body_read=False,new_pose_K_body_read_so_far=0,all64_fixed=True));start=time.monotonic()
 try:
  rows=[];frames=[]
  for frame in files:
   T=np.loadtxt(check(frame['provided_pose']));K=np.loadtxt(check(frame['provided_K']));g=camera_geometry(T,K,n,obj,B,C);perm=g['perm'];role='PASS'if g['PASS']else'UNRESOLVED';ff=dict(frame_index=frame['frame_index'],id=frame['id'],source_frame_id=frame['source_id'],split=frame['split'],provided_pose=frame['provided_pose'],provided_K=frame['provided_K'],provided_T_values=T.tolist(),provided_K_values=K.tolist(),whole_CF_perm=None if perm is None else perm.tolist(),lower_role_continuity_status=role,unresolved_reasons=g['reasons'],continuous_height_proof=g['proof'],semantic_states=['U']*8)
   frames.append(ff)
   for k in range(8):
    index=None if perm is None else int(perm[k]);xy=None if index is None else g['xy'][index];z=None if index is None else float(g['z'][index]);valid=bool(xy is not None and np.isfinite(xy).all()and z is not None and np.isfinite(z)and z>0);inframe=bool(valid and 0<=xy[0]<640 and 0<=xy[1]<480)
    rows.append(dict(frame_index=frame['frame_index'],id=frame['id'],source_frame_id=frame['source_id'],split=frame['split'],native_slot=k,native_name=n['NATIVE_NAMES'][k],registered_fixed_corner_index=index,reference_object_xyz_m=None if index is None else obj[index].tolist(),reference_xy=None if xy is None or not np.isfinite(xy).all()else xy.tolist(),camera_Z_m=z if z is not None and np.isfinite(z)else None,continuous_projection_valid=valid,continuous_inframe=inframe,role_status='PROVISIONAL_WHOLEASSEMBLY_CF'if perm is not None else'UNRESOLVED',lower_role_continuity_status=role if k in(2,3)else'NOT_LOWER_BRIDGE_ROLE',lower_role_bridge_usable=bool(k in(2,3)and g['PASS']),semantic_state='U',semantic_annotation_status='UNREVIEWED_SOURCE_STRUCTURAL_WEAK_LABEL_PENDING',human_gold=False,exact_mesh_surface_membership_required=False,initializer_q_or_error_used=False))
  assert len(frames)==64 and len(rows)==512 and all(sum(r['frame_index']==i for r in rows)==8 for i in range(64))
  result=dict(completed=True,PASS=True,code=bind(__file__),policy=p['policy'],protocol=bind(PROTOCOL),entry=bind(path),selection=p['selection'],initializer_prediction_seal=e['initializer_prediction_seal'],pose_K_bindings=p['pose_K_bindings'],original_CF_code=p['original_CF_code'],axis_proposal=p['axis_proposal'],V144_lower_certificate=p['V144_lower_certificate'],reference_scope=p['reference_scope'],whole_fixed_object_corners_xyz_m=obj.tolist(),frames=frames,rows=rows,all64_512_preserved=True,lower128_preserved=True,lower_continuity_camera_counts=dict(Counter(f['lower_role_continuity_status']for f in frames)),semantic_state_counts={'U':512},RGB_depth_mask_R0_q_score_model_current_V112_PnP_reads_or_calls=0,no_pose_fit_basecut_or_q_L_change=True)
  target=OUT/'ALL512_SOURCE_REFERENCES_GT_ONLY.json';write(target,result);write(OUT/'PROJECTION_COMPLETE.json',dict(completed=True,PASS=True,code=bind(__file__),policy=p['policy'],protocol=bind(PROTOCOL),entry=bind(path),selection=p['selection'],initializer_prediction_seal=e['initializer_prediction_seal'],results=bind(target),frames=64,rows=512,lower_rows=128,all_original_rows_kept=True,lower_continuity_camera_counts=result['lower_continuity_camera_counts'],semantic_states_all_U=True,GT_geometry_only_not_model_input=True,no_q_RGB_model_or_PnP_reads=True,seconds=time.monotonic()-start))
  print(json.dumps(dict(completed=True,PASS=True,seal=bind(OUT/'PROJECTION_COMPLETE.json'),counts=result['lower_continuity_camera_counts'])),flush=True)
 except BaseException:
  write(OUT/'FAILED_EXECUTION.json',dict(completed=False,traceback=traceback.format_exc(),automatic_retry=False));raise
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');ap.add_argument('--entry',type=Path);a=ap.parse_args()
 if a.prepare and not a.entry:prepare()
 elif a.entry and not a.prepare:run(a.entry)
 else:raise SystemExit('Use --prepare or explicit root --entry')
